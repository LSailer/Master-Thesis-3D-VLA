"""Numpy replay buffer that stores observations exactly as received.

This scratch version removes replay-level observation normalization and dtype
switching. ``ReplayBuffer.add`` accepts a prepared observation array/tree plus the
``ObservationFrame`` returned by ``env.step``; transition metadata is copied from
that frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import jax.numpy as jnp
import numpy as np

from src.environments.observation import ObservationFrame

# TODO: Why does we ObservationStorage and ObservationInput it would nice to have only one Obsveration. I want to have Observation Input and Dataclass for the Obsstorage which Replay batch is array from dataclass.
ObservationInput: TypeAlias = np.ndarray | Mapping[str, np.ndarray]
ObservationStorage: TypeAlias = np.ndarray | dict[str, np.ndarray]
JaxObservationBatch: TypeAlias = jnp.ndarray | dict[str, jnp.ndarray]
ReplayBatch: TypeAlias = dict[str, jnp.ndarray | dict[str, jnp.ndarray]]


@dataclass(frozen=True)
class ReplayTransition:
    obs: np.ndarray | dict[str, np.ndarray]
    action: np.int8
    reward: np.float16
    is_first: np.bool_
    is_episode_end: np.bool_


@dataclass(frozen=True)
class _FieldSpec:
    """Shape and dtype inferred from one replay observation field."""

    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass
class _ReplayStorage:
    """Arrays backing replay transitions and lazy observation storage."""

    capacity: int
    obs: ObservationStorage | None
    actions: np.ndarray
    rewards: np.ndarray
    episode_ends: np.ndarray

    @classmethod
    def empty(cls, capacity: int) -> "_ReplayStorage":
        """Allocate transition arrays without observation storage."""
        # TODO: Why do we need this cls?
        return cls(
            capacity=capacity,
            obs=None,
            actions=np.empty(capacity, dtype=np.int32),
            rewards=np.empty(capacity, dtype=np.float32),
            episode_ends=np.empty(capacity, dtype=np.bool_),
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        episode_ends: np.ndarray,
    ) -> "_ReplayStorage":
        """Wrap loaded arrays in replay-storage form for validation datasets."""
        return cls(
            capacity=len(actions),
            obs=obs,
            actions=actions.astype(np.int32, copy=False),
            rewards=rewards.astype(np.float32, copy=False),
            episode_ends=episode_ends.astype(np.bool_, copy=False),
        )


# TODO why we need this function. Looks like boilercode. It can be removed.
def _field_spec(array: np.ndarray) -> _FieldSpec:
    """Return the exact shape and dtype to enforce for later inserts."""
    return _FieldSpec(shape=tuple(array.shape), dtype=np.dtype(array.dtype))


def _validate_array(array: np.ndarray, spec: _FieldSpec, label: str) -> None:
    """Raise if a replay observation field changes shape or dtype."""
    shape = tuple(array.shape)
    if shape != spec.shape:
        raise ValueError(
            f"observation{label} shape changed after replay initialization: "
            f"expected {spec.shape}, got {shape}"
        )
    dtype = np.dtype(array.dtype)
    if dtype != spec.dtype:
        raise TypeError(
            f"observation{label} dtype changed after replay initialization: "
            f"expected {spec.dtype}, got {dtype}"
        )


def _gather_obs(obs: ObservationStorage, indices: np.ndarray) -> JaxObservationBatch:
    """Gather observation windows without normalization or dtype conversion."""
    if isinstance(obs, Mapping):
        return {key: jnp.asarray(value[indices]) for key, value in obs.items()}
    return jnp.asarray(obs[indices])


def _gather_sequence_batch(
    starts: np.ndarray,
    seq_len: int,
    storage: _ReplayStorage,
) -> ReplayBatch:
    """Gather fixed-length replay windows from explicit start indices."""
    if storage.obs is None:
        raise RuntimeError("ReplayBuffer observation storage is not initialized")

    indices = starts[:, None] + np.arange(seq_len)[None, :]
    episode_ends_b = storage.episode_ends[indices]
    is_first = np.zeros_like(episode_ends_b, dtype=np.bool_)
    is_first[:, 0] = True
    is_first[:, 1:] = episode_ends_b[:, :-1]
    return {
        "obs": _gather_obs(storage.obs, indices),
        "actions": jnp.asarray(storage.actions[indices]),
        "rewards": jnp.asarray(storage.rewards[indices]),
        "is_episode_end": jnp.asarray(episode_ends_b),
        "is_first": jnp.asarray(is_first),
    }


class ReplayBuffer:
    """Lazy ring buffer for prepared observations and ``ObservationFrame`` data.

    Args:
        capacity: Maximum number of transitions to store.

    The first ``add`` call defines the observation schema. Later observations must
    have the same keys, shapes, and dtypes. Sampling returns the stored
    observation dtypes unchanged; any normalization or casting belongs at the
    observation-packing/training boundary.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._storage = _ReplayStorage.empty(capacity)
        self._obs_specs: _FieldSpec | dict[str, _FieldSpec] | None = None
        self.idx = 0
        self.size = 0

    @property
    def capacity(self) -> int:
        """Maximum number of transitions in the ring buffer."""
        return self._storage.capacity

    # TODO: Change the property to the dataclass ObservationStorage.
    @property
    def obs(self) -> ObservationStorage | None:
        """Observation storage, allocated lazily on the first ``add``."""
        return self._storage.obs

    @property
    def actions(self) -> np.ndarray:
        """Discrete action storage, copied from ``ObservationFrame.previous_action``."""
        return self._storage.actions

    @property
    def rewards(self) -> np.ndarray:
        """Reward storage, copied from ``ObservationFrame.reward``."""
        return self._storage.rewards

    @property
    def episode_ends(self) -> np.ndarray:
        """Episode-end storage, copied from ``ObservationFrame.is_episode_end``."""
        return self._storage.episode_ends

    def add(self, inputs: ObservationInput, frame: ObservationFrame) -> None:
        """Append one transition from prepared inputs and the resulting frame.

        Args:
            inputs: Prepared replay observation for the state before the action.
                This can be one ndarray or a mapping of named ndarray fields.
            frame: ``ObservationFrame`` returned by ``env.step``. Its
                ``previous_action``, ``reward``, and ``is_episode_end`` fields
                become the transition metadata.

        Raises:
            ValueError: If a reset frame without ``previous_action`` is added.
            TypeError: If later observations change dtype or single/dict form.
        """
        action = getattr(frame, "previous_action", None)
        if action is None:
            raise ValueError(
                "ReplayBuffer.add requires an ObservationFrame from env.step with "
                "previous_action set"
            )
        if self._storage.obs is None:
            self._initialize_obs(inputs)

        self._store_obs(inputs)
        self._storage.actions[self.idx] = int(action)
        self._storage.rewards[self.idx] = float(frame.reward)
        self._storage.episode_ends[self.idx] = bool(frame.is_episode_end)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch:
        """Sample fixed-length transition sequences from valid ring positions.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Number of consecutive transitions per window.

        Returns:
            A replay-sequence batch with raw observations and transition fields.
            ``is_episode_end`` and ``is_first`` are boolean arrays; convert them
            at the training boundary if a model expects floats.
        """
        if self._storage.obs is None:
            raise RuntimeError("cannot sample before adding at least one transition")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if self.size < seq_len:
            raise ValueError(
                f"not enough data in replay buffer: size={self.size}, seq_len={seq_len}"
            )
        starts = self._sample_starts(batch_size, seq_len)
        return _gather_sequence_batch(starts, seq_len, self._storage)

    def _initialize_obs(self, inputs: ObservationInput) -> None:
        """Allocate observation arrays from the first replay observation."""
        if isinstance(inputs, Mapping):
            self._initialize_mapping_obs(inputs)
            return
        self._initialize_array_obs(inputs)

    def _initialize_mapping_obs(self, inputs: Mapping[str, np.ndarray]) -> None:
        """Allocate named observation-field arrays from the first mapping."""
        if not inputs:
            raise ValueError("mapping observations must contain at least one field")

        specs = {key: _field_spec(np.asarray(value)) for key, value in inputs.items()}
        self._storage.obs = {
            key: np.empty((self.capacity, *spec.shape), dtype=spec.dtype)
            for key, spec in specs.items()
        }
        self._obs_specs = specs

    def _initialize_array_obs(self, inputs: np.ndarray) -> None:
        """Allocate single-array observation storage from the first ndarray."""
        array = np.asarray(inputs)
        spec = _field_spec(array)
        self._storage.obs = np.empty((self.capacity, *spec.shape), dtype=spec.dtype)
        self._obs_specs = spec

    def _store_obs(self, inputs: ObservationInput) -> None:
        """Write one observation into the current ring-buffer slot."""
        obs_storage = self._storage.obs
        obs_specs = self._obs_specs
        if obs_storage is None or obs_specs is None:
            raise RuntimeError("ReplayBuffer observation storage is not initialized")

        if isinstance(obs_storage, Mapping):
            if not isinstance(inputs, Mapping):
                raise TypeError("mapping replay buffer requires mapping inputs")
            if not isinstance(obs_specs, Mapping):
                raise TypeError("mapping replay specs missing")
            self._store_mapping_obs(inputs, obs_storage, obs_specs)
            return

        if isinstance(inputs, Mapping):
            raise TypeError("single replay buffer requires ndarray inputs")
        if not isinstance(obs_specs, _FieldSpec):
            raise TypeError("single replay spec missing")
        array = np.asarray(inputs)
        _validate_array(array, obs_specs, "")
        obs_storage[self.idx] = array

    def _store_mapping_obs(
        self,
        inputs: Mapping[str, np.ndarray],
        obs_storage: Mapping[str, np.ndarray],
        obs_specs: Mapping[str, _FieldSpec],
    ) -> None:
        """Write one mapping observation after exact key/shape/dtype checks."""
        expected_keys = set(obs_storage)
        actual_keys = set(inputs)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise KeyError(
                "observation keys changed after replay initialization: "
                f"missing={missing}, extra={extra}"
            )

        for key, storage in obs_storage.items():
            array = np.asarray(inputs[key])
            _validate_array(array, obs_specs[key], f"[{key!r}]")
            storage[self.idx] = array

    def _sample_starts(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Pick valid sequence starts without crossing the ring-buffer write head."""
        # TODO: I believe that this function can simply written because if i understand correclty. Why we need the valid?
        if self.size < self.capacity:
            n_valid = self.size - seq_len + 1
            return np.random.randint(0, n_valid, size=batch_size)

        n_new = max(0, self.idx - seq_len + 1)
        n_old = max(0, self.capacity - seq_len - self.idx + 1)
        n_valid = n_new + n_old
        if n_valid <= 0:
            raise ValueError(
                "not enough contiguous data in replay buffer: "
                f"capacity={self.capacity}, idx={self.idx}, seq_len={seq_len}"
            )
        raw = np.random.randint(0, n_valid, size=batch_size)
        return np.where(raw < n_new, raw, raw - n_new + self.idx)


# TODO: Why do we need for the Dataset a buffer? This should be in inference time. No need to save it in the storage.
class ValReplayDataset:
    """Static replay dataset loaded from a pre-collected ``.npz`` file.

    This helper mirrors ``ReplayBuffer.sample`` for validation loss computation.
    It also preserves the stored observation dtype; no replay-level normalization
    is applied.
    """

    def __init__(self, path: str) -> None:
        data = np.load(path)
        self._storage = _ReplayStorage.from_arrays(
            obs=data["obs"],
            actions=data["actions"],
            rewards=data["rewards"],
            episode_ends=data["episode_ends"],
        )
        episode_end_indices = np.where(self._storage.episode_ends)[0]
        self._ep_starts = np.concatenate([[0], episode_end_indices + 1])
        self._ep_starts = self._ep_starts[self._ep_starts < len(self)]
        ends = np.concatenate([episode_end_indices + 1, [len(self)]])
        self._ep_lengths = ends[: len(self._ep_starts)] - self._ep_starts
        print(
            f"ValReplayDataset: {len(self)} steps, "
            f"{self.episode_count()} episodes from {path}"
        )

    def __len__(self) -> int:
        """Return the number of stored validation transitions."""
        return self._storage.capacity

    def episode_count(self) -> int:
        """Return the number of reconstructed validation episodes."""
        return len(self._ep_starts)

    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch:
        """Sample random fixed-length subsequences from reconstructed episodes."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        valid = np.where(self._ep_lengths >= seq_len)[0]
        if len(valid) == 0:
            raise ValueError(
                f"no episodes with length >= {seq_len} "
                f"(max length: {self._ep_lengths.max()})"
            )

        ep_idx = np.random.choice(valid, size=batch_size)
        ep_starts = self._ep_starts[ep_idx]
        ep_lens = self._ep_lengths[ep_idx]
        offsets = np.array(
            [np.random.randint(0, ep_len - seq_len + 1) for ep_len in ep_lens]
        )
        starts = ep_starts + offsets
        return _gather_sequence_batch(starts, seq_len, self._storage)
