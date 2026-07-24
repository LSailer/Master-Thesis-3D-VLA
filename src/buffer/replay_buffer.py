"""Replay buffer that stores transitions in preallocated parallel arrays.

The public interface is batch-first: ``ReplayBuffer.add`` accepts one
``ReplayTransition`` and ``sample`` returns a ``ReplayBatch`` whose leaves have
``(batch, time)`` leading axes. Internally, the ring is a structure-of-arrays:
one preallocated NumPy array per observation field plus scalar arrays for
actions, rewards, and flags. Sampling is a vectorized fancy-index gather —
one host copy and one device transfer per field — instead of a per-transition
Python object walk (which cost ~119 ms per 16x64 batch at production shape;
see docs/notes/house-points-pose-profiling.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.typing import DTypeLike

from src.environments.observation import ObservationFrame

ReplayObservationBatch: TypeAlias = jax.Array | dict[str, jax.Array]
ObservationLeaf: TypeAlias = jnp.ndarray | np.ndarray

#TODO Can remove the dataclass not need more
@dataclass
class HybridObservation:
    """Hybrid image plus compact world-point observation for one transition.

    Args:
        image: Image observation for one environment step.
        wp_cp: Compact world-point / camera-pose feature for one environment
            step.
    """

    image: ObservationLeaf
    wp_cp: ObservationLeaf


ObservationInput: TypeAlias = (
    ObservationLeaf | HybridObservation | Mapping[str, ObservationLeaf]
)

# Internal storage key for unstructured (single-array) observations.
_ARRAY_FIELD = "__array__"


@struct.dataclass
class ReplayTransition:
    """One transition accepted by the replay buffer.

    Args:
        obs: Replay observation for one stored frame.
        action: Discrete action that produced ``obs``.
        reward: Reward returned with ``obs``.
        is_first: Whether this transition should reset sequence state. Sampled
            batches also mark the first transition in every sampled window as
            ``True``.
        is_episode_end: Whether this transition ended an episode.
        encoders: Optional ``field -> encoder id`` routing for mapping
            observations. Opaque ints so the buffer stays decoupled from any
            encoder registry; the buffer treats this as schema — locked in by
            the first added transition and required to stay identical after.
        global_feature: Optional live, batch-wide feature (e.g. the current
            house-context map). The buffer keeps only the latest value across
            all adds — one element, not a per-step series — and returns it on
            every sampled batch.
    """

    obs: ObservationInput
    action: int | np.integer
    reward: float | np.floating
    is_first: bool | np.bool_
    is_episode_end: bool | np.bool_
    encoders: Mapping[str, int] | None = struct.field(
        pytree_node=False, default=None
    )
    global_feature: ObservationLeaf | None = None

    @classmethod
    def from_frame(
        cls,
        obs: ObservationInput,
        frame: ObservationFrame,
        encoders: Mapping[str, int] | None = None,
        global_feature: ObservationLeaf | None = None,
    ) -> "ReplayTransition":
        """Build one transition from prepared replay observation and env frame.

        Args:
            obs: Replay observation prepared for ``frame``.
            frame: Environment observation frame returned by ``env.step``.
            encoders: Optional ``field -> encoder id`` routing for ``obs``.
            global_feature: Optional live batch-wide feature for this step.

        Returns:
            A replay transition whose scalar fields come from ``frame``.

        Raises:
            ValueError: If ``frame`` is a reset frame without a previous action.
        """
        if frame.previous_action is None:
            raise ValueError("cannot build ReplayTransition from reset frame")
        return cls(
            obs=obs,
            action=frame.previous_action,
            reward=frame.reward,
            is_first=frame.is_first,
            is_episode_end=frame.is_episode_end,
            encoders=encoders,
            global_feature=global_feature,
        )


@struct.dataclass
class ReplayBatch:
    """Sampled replay batch with ``(batch, time)`` leading axes.

    Args:
        obs: Observation leaves stacked to ``(B, T, *obs_shape)``. Structured
            observations are stored as ``dict[field, array]`` with the same
            ``(B, T)`` prefix on every field.
        actions: Actions that produced the stored observations, one-hot encoded
            with the configured floating dtype and shape ``(B, T, num_actions)``.
        rewards: Rewards returned with the stored observations as configured
            floating arrays of shape ``(B, T)``.
        is_first: RSSM reset mask as configured floating arrays of shape
            ``(B, T)``. This is true at each sampled sequence start and after
            episode ends.
        is_episode_end: Episode-end flags returned with the stored observations
            as configured floating arrays of shape ``(B, T)``.
        global_feature: Live, batch-wide feature (e.g. the current
            house-context map). Unlike the replay fields it has no ``(B, T)``
            prefix — one array is shared by every sequence in the batch. The
            buffer fills it with the latest value carried by an added
            transition (``None`` when no transition carried one); callers can
            still swap it after sampling via ``batch.replace(...)``.
        encoders: ``field -> encoder id`` routing captured from the first
            added transition (``None`` when transitions carried none). Static
            metadata, not a pytree leaf.
    """

    obs: ReplayObservationBatch
    actions: jax.Array
    rewards: jax.Array
    is_first: jax.Array
    is_episode_end: jax.Array
    global_feature: jax.Array | None = None
    encoders: Mapping[str, int] | None = struct.field(
        pytree_node=False, default=None
    )


ReplayTransitionBatch: TypeAlias = list[list[ReplayTransition]]


class ReplayBuffer:
    """Structure-of-arrays ring buffer for ``ReplayTransition`` frames.

    Args:
        capacity: Maximum number of transitions to store.
        num_actions: Number of discrete actions for sampled one-hot batches.
        float_dtype: Floating dtype for sampled action, reward, and mask arrays.

    Observation storage is allocated lazily on the first ``add`` (one array
    per field, shaped ``(capacity, *leaf_shape)``); the observation structure
    (plain array, ``HybridObservation``, or mapping with fixed keys) is locked
    in by that first transition. Backing arrays come from ``np.empty``, so
    physical pages are committed only as slots are written.
    """

    def __init__(
        self,
        capacity: int,
        num_actions: int,
        *,
        float_dtype: DTypeLike = jnp.float32,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}")
        self.capacity = capacity
        self.num_actions = num_actions
        self.float_dtype = float_dtype
        self.idx = 0
        self.size = 0
        self._obs_kind: str | None = None  # "array" | "hybrid" | "mapping"
        self._obs_store: dict[str, np.ndarray] = {}
        self._encoders: dict[str, int] | None = None
        self._global_feature: np.ndarray | None = None
        self._actions = np.zeros(capacity, dtype=np.int32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._is_first = np.zeros(capacity, dtype=np.bool_)
        self._is_episode_end = np.zeros(capacity, dtype=np.bool_)

    @staticmethod
    def _split_observation(
        obs: ObservationInput,
    ) -> tuple[str, dict[str, np.ndarray]]:
        """Normalize one observation into ``(kind, field -> host array)``.

        Args:
            obs: Plain array, ``HybridObservation``, or mapping observation.

        Returns:
            The structure kind (``"array"``/``"hybrid"``/``"mapping"``) and a
            dict of host NumPy leaves keyed by storage field name.
        """
        if isinstance(obs, HybridObservation):
            return "hybrid", {
                "image": np.asarray(obs.image),
                "wp_cp": np.asarray(obs.wp_cp),
            }
        if isinstance(obs, Mapping):
            return "mapping", {key: np.asarray(value) for key, value in obs.items()}
        return "array", {_ARRAY_FIELD: np.asarray(obs)}

    def _ensure_store(self, kind: str, fields: dict[str, np.ndarray]) -> None:
        """Allocate storage on first add and enforce a stable structure after.

        Args:
            kind: Observation structure kind from ``_split_observation``.
            fields: Normalized observation leaves for the incoming transition.

        Raises:
            TypeError: If ``kind`` differs from the stored structure.
            KeyError: If mapping keys differ from the stored field set.
            ValueError: If a leaf's shape or dtype differs from its store.
        """
        if self._obs_kind is None:
            self._obs_kind = kind
            self._obs_store = {
                key: np.empty((self.capacity, *value.shape), dtype=value.dtype)
                for key, value in fields.items()
            }
            return
        if kind != self._obs_kind:
            raise TypeError(
                f"cannot mix {kind} and {self._obs_kind} observations in one buffer"
            )
        if set(fields) != set(self._obs_store):
            raise KeyError(
                "replay observation keys changed between added transitions: "
                f"expected={sorted(self._obs_store)}, got={sorted(fields)}"
            )
        for key, value in fields.items():
            store = self._obs_store[key]
            if value.shape != store.shape[1:] or value.dtype != store.dtype:
                raise ValueError(
                    f"replay observation field {key!r} changed layout: stored "
                    f"{store.shape[1:]}/{store.dtype}, got {value.shape}/{value.dtype}"
                )

    def add(self, replay_transition: ReplayTransition) -> None:
        """Append one replay transition.

        Args:
            replay_transition: Transition whose leaves are copied into the
                current ring slot (the caller keeps ownership of its arrays).
        """
        kind, fields = self._split_observation(replay_transition.obs)
        self._ensure_store(kind, fields)
        if replay_transition.encoders is not None:
            incoming = dict(replay_transition.encoders)
            if self._encoders is None:
                self._encoders = incoming
            elif incoming != self._encoders:
                raise ValueError(
                    "encoder routing changed between added transitions: "
                    f"stored={self._encoders}, got={incoming}"
                )
        if replay_transition.global_feature is not None:
            # Latest wins: one live element, not a per-step series.
            self._global_feature = np.asarray(replay_transition.global_feature)
        for key, value in fields.items():
            self._obs_store[key][self.idx] = value
        self._actions[self.idx] = int(replay_transition.action)
        self._rewards[self.idx] = float(replay_transition.reward)
        self._is_first[self.idx] = bool(replay_transition.is_first)
        self._is_episode_end[self.idx] = bool(replay_transition.is_episode_end)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch:
        """Sample fixed-length frame-aligned replay batches.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Number of consecutive transitions per window.

        Returns:
            A ``ReplayBatch`` with ``(batch_size, seq_len)`` leading axes.
        """
        ring = self._sample_ring_indices(batch_size, seq_len)

        episode_ends = self._is_episode_end[ring]
        # A window resets sequence state at its start, at stored episode
        # starts, and on the frame after an episode end.
        is_first = self._is_first[ring].copy()
        is_first[:, 0] = True
        is_first[:, 1:] |= episode_ends[:, :-1]

        # Gather every sampled leaf host-side, then move it to the device with
        # a single ``device_put`` instead of one ``jnp.asarray`` per field. The
        # per-field variant paid fixed transfer/dispatch overhead once per leaf
        # (six times for a hybrid-image adapter); fusing them into one pytree
        # transfer cuts that to a single dispatch on the hot sampling path.
        payload = jax.device_put(
            {
                "obs": {key: store[ring] for key, store in self._obs_store.items()},
                "actions": self._actions[ring],
                "rewards": self._rewards[ring],
                "is_first": is_first,
                "is_episode_end": episode_ends,
            }
        )

        obs_fields = payload["obs"]
        obs: ReplayObservationBatch = (
            obs_fields[_ARRAY_FIELD] if self._obs_kind == "array" else obs_fields
        )
        return ReplayBatch(
            obs=obs,
            actions=jax.nn.one_hot(
                payload["actions"], self.num_actions, dtype=self.float_dtype
            ),
            rewards=payload["rewards"].astype(self.float_dtype),
            is_first=payload["is_first"].astype(self.float_dtype),
            is_episode_end=payload["is_episode_end"].astype(self.float_dtype),
            global_feature=(
                None
                if self._global_feature is None
                else jax.device_put(self._global_feature)
            ),
            encoders=self._encoders,
        )

    def sample_transitions(
        self,
        batch_size: int,
        seq_len: int,
    ) -> ReplayTransitionBatch:
        """Sample copied transition windows for transition-level checks.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Number of consecutive transitions per window.

        Returns:
            A list with ``batch_size`` sampled transition sequences. Each
            sequence contains ``seq_len`` copied transitions; the first
            transition of every window is marked ``is_first=True``.
        """
        ring = self._sample_ring_indices(batch_size, seq_len)
        return [
            [
                self._transition_at(int(ring_index), first=offset == 0)
                for offset, ring_index in enumerate(window)
            ]
            for window in ring
        ]

    def _sample_ring_indices(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Draw window starts and return their ``(B, T)`` physical ring indices.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Number of consecutive transitions per window.

        Returns:
            An int64 array of ring positions in chronological window order.
        """
        self._validate_sample_request(batch_size, seq_len)
        n_valid = self.size - seq_len + 1
        starts = np.asarray(np.random.randint(0, n_valid, size=batch_size))
        oldest_index = 0 if self.size < self.capacity else self.idx
        offsets = np.arange(seq_len, dtype=np.int64)
        return (oldest_index + starts[:, None] + offsets[None, :]) % self.capacity

    def _validate_sample_request(self, batch_size: int, seq_len: int) -> None:
        """Validate common sampling arguments before drawing windows."""
        if self.size == 0:
            raise RuntimeError("cannot sample before adding at least one transition")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if self.size < seq_len:
            raise ValueError(
                f"not enough data in replay buffer: size={self.size}, seq_len={seq_len}"
            )

    def _transition_at(self, index: int, *, first: bool = False) -> ReplayTransition:
        """Rebuild the stored transition at a physical ring index.

        Args:
            index: Physical ring position of the stored transition.
            first: Whether to mark the rebuilt transition as a window start.

        Returns:
            A ``ReplayTransition`` whose observation leaves are fresh copies
            (mutating them cannot affect replay storage).

        Raises:
            RuntimeError: If the slot has not been written yet.
        """
        if index >= self.size and self.size < self.capacity:
            raise RuntimeError(f"ring slot {index} does not contain a transition")
        leaves = {key: store[index].copy() for key, store in self._obs_store.items()}
        obs: ObservationInput
        if self._obs_kind == "array":
            obs = leaves[_ARRAY_FIELD]
        elif self._obs_kind == "hybrid":
            obs = HybridObservation(image=leaves["image"], wp_cp=leaves["wp_cp"])
        else:
            obs = leaves
        return ReplayTransition(
            obs=obs,
            action=int(self._actions[index]),
            reward=float(self._rewards[index]),
            is_first=True if first else bool(self._is_first[index]),
            is_episode_end=bool(self._is_episode_end[index]),
        )
