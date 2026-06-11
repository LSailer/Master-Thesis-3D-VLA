"""Unified numpy ring buffer with sequence sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class BufferConfig:
    """Configuration for ReplayBuffer.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of a single observation, e.g. (3, 64, 64) or (4116,).
        obs_dtype: Storage dtype — "uint8" for images, "float32" for features.
        normalize_obs: If True and obs_dtype is "uint8", divide by 255.0 on sample.
    """
    capacity: int
    obs_shape: tuple[int, ...] | Mapping[str, tuple[int, ...]]
    obs_dtype: str | Mapping[str, str] = "uint8"
    normalize_obs: bool | Mapping[str, bool] = True


def _np_dtype(dtype_name: str) -> type:
    if dtype_name == "uint8":
        return np.uint8
    if dtype_name == "float16":
        return np.float16
    return np.float32


def _storage_array(capacity: int, shape: tuple[int, ...], dtype_name: str) -> np.ndarray:
    return np.zeros((capacity, *shape), dtype=_np_dtype(dtype_name))


def _obs_storage(config: BufferConfig, capacity: int) -> Any:
    if isinstance(config.obs_shape, Mapping):
        dtype_map = config.obs_dtype if isinstance(config.obs_dtype, Mapping) else {}
        return {
            key: _storage_array(
                capacity,
                tuple(shape),
                str(dtype_map.get(key, "uint8")),
            )
            for key, shape in config.obs_shape.items()
        }
    return _storage_array(capacity, tuple(config.obs_shape), str(config.obs_dtype))


def _normalize_flags(config: BufferConfig) -> bool | dict[str, bool]:
    if isinstance(config.obs_shape, Mapping):
        dtype_map = config.obs_dtype if isinstance(config.obs_dtype, Mapping) else {}
        norm_map = config.normalize_obs if isinstance(config.normalize_obs, Mapping) else {}
        return {
            key: bool(norm_map.get(key, True)) and str(dtype_map.get(key, "uint8")) == "uint8"
            for key in config.obs_shape
        }
    return bool(config.normalize_obs) and str(config.obs_dtype) == "uint8"


def _gather_obs(obs: Any, indices: np.ndarray, normalize: bool | Mapping[str, bool]) -> Any:
    if isinstance(obs, Mapping):
        norm_map = normalize if isinstance(normalize, Mapping) else {}
        return {
            key: _gather_obs(value, indices, bool(norm_map.get(key, False)))
            for key, value in obs.items()
        }
    obs_jnp = jnp.array(obs[indices], dtype=jnp.float32)
    if normalize:
        obs_jnp = obs_jnp / 255.0
    return obs_jnp


def _gather_sequence_batch(
    starts: np.ndarray,
    seq_len: int,
    obs: Any,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    terminals: np.ndarray,
    normalize: bool | Mapping[str, bool],
) -> dict[str, jnp.ndarray]:
    """Gather length-``seq_len`` windows at ``starts`` and pack a batch dict.

    Shared by :class:`ReplayBuffer` and :class:`ValReplayDataset`, which differ
    only in how they choose ``starts`` (ring-buffer wrap arithmetic vs. random
    per-episode offsets). ``is_first`` is set at t=0 and wherever ``done`` flipped
    on the previous step; ``obs`` is divided by 255 when ``normalize`` is set.
    """
    indices = starts[:, None] + np.arange(seq_len)[None, :]  # (B, T)
    actions_b = actions[indices]
    rewards_b = rewards[indices]
    dones_b = dones[indices]
    terminals_b = terminals[indices]

    is_first = np.zeros_like(dones_b)
    is_first[:, 0] = True
    is_first[:, 1:] = dones_b[:, :-1]

    return {
        "obs": _gather_obs(obs, indices, normalize),
        "actions": jnp.array(actions_b, dtype=jnp.int32),
        "rewards": jnp.array(rewards_b, dtype=jnp.float32),
        "dones": jnp.array(dones_b, dtype=jnp.float32),
        "terminals": jnp.array(terminals_b, dtype=jnp.float32),
        "is_first": jnp.array(is_first, dtype=jnp.float32),
    }


class ReplayBuffer:
    """Ring buffer that stores transitions and samples fixed-length sequences.

    Replaces the old separate ReplayBuffer (uint8) and VGGTReplayBuffer (float32).
    """

    def __init__(self, config: BufferConfig | object) -> None:
        # Backward compat: accept a DreamerConfig or R2DreamerConfig
        if not isinstance(config, BufferConfig):
            config = BufferConfig(
                capacity=config.buffer_capacity,
                obs_shape=config.obs_shape,
            )
        cap = config.capacity
        self.obs = _obs_storage(config, cap)
        self.actions = np.zeros(cap, dtype=np.int32)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.dones = np.zeros(cap, dtype=np.bool_)
        self.terminals = np.zeros(cap, dtype=np.bool_)
        self._normalize = _normalize_flags(config)
        self.capacity = cap
        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,
            terminal: bool = False) -> None:
        if isinstance(self.obs, Mapping):
            if not isinstance(obs, Mapping):
                raise TypeError("ReplayBuffer configured for mapping observations")
            for key, value in obs.items():
                self.obs[key][self.idx] = value
        else:
            self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.terminals[self.idx] = terminal
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int) -> dict[str, jnp.ndarray]:
        starts = self._sample_starts(batch_size, seq_len)
        return _gather_sequence_batch(
            starts, seq_len, self.obs, self.actions, self.rewards,
            self.dones, self.terminals, self._normalize,
        )

    def _sample_starts(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Pick ``batch_size`` valid sequence start indices into the ring buffer.

        Before the buffer wraps, ``[0, size)`` is contiguous. After it wraps,
        sequences must not cross the write head, so starts are drawn from the two
        safe regions ``[0, idx-seq_len]`` (new) and ``[idx, cap-seq_len]`` (old)
        via a single ``randint`` over their combined length and remapped onto the
        old region — preserving the original RNG draw order.
        """
        if self.size < self.capacity:
            # Buffer hasn't wrapped — data in [0, size) is contiguous
            n_valid = self.size - seq_len + 1
            assert n_valid > 0, "Not enough data in buffer"
            return np.random.randint(0, n_valid, size=batch_size)

        # Buffer has wrapped — avoid sequences crossing the write head.
        n_new = max(0, self.idx - seq_len + 1)
        n_old = max(0, self.capacity - seq_len - self.idx + 1)
        n_valid = n_new + n_old
        assert n_valid > 0, "Not enough contiguous data in buffer"
        raw = np.random.randint(0, n_valid, size=batch_size)
        return np.where(raw < n_new, raw, raw - n_new + self.idx)


class ValReplayDataset:
    """Static replay dataset loaded from a pre-collected .npz file.

    Provides the same sample() interface as ReplayBuffer for computing
    validation loss without a live environment.
    """

    def __init__(self, path: str, normalize: bool = True):
        self._normalize = normalize
        data = np.load(path)
        self.obs = data["obs"]          # (N, ...) uint8 or float32
        self.actions = data["actions"]  # (N,) int32
        self.rewards = data["rewards"]  # (N,) float32
        self.dones = data["dones"]      # (N,) bool
        self.terminals = data["terminals"]  # (N,) bool

        # Reconstruct episode boundaries from dones
        # Episode starts at index 0 and after every done
        done_indices = np.where(self.dones)[0]
        self._ep_starts = np.concatenate([[0], done_indices + 1])
        # Remove any start that would be past the end of data
        self._ep_starts = self._ep_starts[self._ep_starts < len(self.obs)]
        # Episode lengths
        ends = np.concatenate([done_indices + 1, [len(self.obs)]])
        self._ep_lengths = ends[:len(self._ep_starts)] - self._ep_starts

        print(f"ValReplayDataset: {len(self.obs)} steps, "
              f"{len(self._ep_starts)} episodes from {path}")

    def sample(self, batch_size: int, seq_len: int) -> dict:
        """Sample random subsequences, same format as ReplayBuffer.sample()."""
        # Find episodes long enough
        valid = np.where(self._ep_lengths >= seq_len)[0]
        assert len(valid) > 0, (
            f"No episodes with length >= {seq_len} "
            f"(max length: {self._ep_lengths.max()})"
        )

        # Sample random episodes and random start offsets within them
        ep_idx = np.random.choice(valid, size=batch_size)
        ep_starts = self._ep_starts[ep_idx]
        ep_lens = self._ep_lengths[ep_idx]
        offsets = np.array([
            np.random.randint(0, l - seq_len + 1) for l in ep_lens
        ])
        starts = ep_starts + offsets
        return _gather_sequence_batch(
            starts, seq_len, self.obs, self.actions, self.rewards,
            self.dones, self.terminals, self._normalize,
        )


def VGGTReplayBuffer(capacity: int, feature_dim: int = 4116) -> ReplayBuffer:
    """Backward-compat factory. Use ReplayBuffer(BufferConfig(...)) directly."""
    return ReplayBuffer(BufferConfig(
        capacity=capacity,
        obs_shape=(feature_dim,),
        obs_dtype="float32",
        normalize_obs=False,
    ))
