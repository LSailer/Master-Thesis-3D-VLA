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
    """

    obs: ObservationInput
    action: int | np.integer
    reward: float | np.floating
    is_first: bool | np.bool_
    is_episode_end: bool | np.bool_

    @classmethod
    def from_frame(
        cls,
        obs: ObservationInput,
        frame: ObservationFrame,
    ) -> "ReplayTransition":
        """Build one transition from prepared replay observation and env frame.

        Args:
            obs: Replay observation prepared for ``frame``.
            frame: Environment observation frame returned by ``env.step``.

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
        global_feature: Batch-wide feature array attached *after* sampling
            via :meth:`change_global_feature` (e.g. the live house-context map
            in ``live_house_context`` mode). Unlike the replay fields it has
            no ``(B, T)`` prefix — one array is shared by every sequence in
            the batch. The buffer itself never stores or fills it; ``sample``
            always returns ``None``.
    """

    obs: ReplayObservationBatch
    actions: jax.Array
    rewards: jax.Array
    is_first: jax.Array
    is_episode_end: jax.Array
    global_feature: jax.Array | None = None

    def change_global_feature(self, value: jax.Array):
        """Change the batch's global feature in place to ``value``.

        Use this for a live, batch-wide array that is computed outside the
        replay buffer (e.g. the house-context map). The value is stored as
        given — the caller is responsible for device placement and dtype.

        Args:
            value: Feature array; replaces any previously set global feature.

        Raises:
            dataclasses.FrozenInstanceError: Always, as long as
                ``ReplayBatch`` is a frozen ``flax.struct`` dataclass —
                in-place assignment is blocked by the class decorator.
        """
        self.global_feature = value


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

        obs_fields = {
            key: jnp.asarray(store[ring]) for key, store in self._obs_store.items()
        }
        obs: ReplayObservationBatch = (
            obs_fields[_ARRAY_FIELD] if self._obs_kind == "array" else obs_fields
        )

        episode_ends = self._is_episode_end[ring]
        # A window resets sequence state at its start, at stored episode
        # starts, and on the frame after an episode end.
        is_first = self._is_first[ring].copy()
        is_first[:, 0] = True
        is_first[:, 1:] |= episode_ends[:, :-1]

        actions = jax.nn.one_hot(
            jnp.asarray(self._actions[ring]), self.num_actions, dtype=self.float_dtype
        )
        return ReplayBatch(
            obs=obs,
            actions=actions,
            rewards=jnp.asarray(self._rewards[ring], dtype=self.float_dtype),
            is_first=jnp.asarray(is_first, dtype=self.float_dtype),
            is_episode_end=jnp.asarray(episode_ends, dtype=self.float_dtype),
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
