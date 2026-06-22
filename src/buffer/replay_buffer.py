"""Unified numpy ring buffer with sequence sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import jax.numpy as jnp


ObsShape = tuple[int, ...] | Mapping[str, tuple[int, ...]]
ObsDType = str | Mapping[str, str]
ObsNormalize = bool | Mapping[str, bool]
ReplayBatch = dict[str, jnp.ndarray | dict[str, jnp.ndarray]]


@dataclass(frozen=True)
class BufferConfig:
    """Configuration for ReplayBuffer.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of one observation, or a mapping of named fields to
            shapes for multi-modal observations such as hybrid image+WP/CP.
        obs_dtype: Storage dtype, either one dtype for single-observation
            buffers or a mapping keyed like ``obs_shape``.
        normalize_obs: If True and obs_dtype is "uint8", divide by 255.0 on
            sample. Mapping configs can choose this per field.
    """

    capacity: int
    obs_shape: ObsShape
    obs_dtype: ObsDType = "uint8"
    normalize_obs: ObsNormalize = True


@dataclass(frozen=True)
class _FieldSpec:
    shape: tuple[int, ...]
    dtype: str
    normalize: bool
    keep_uint8_on_sample: bool


def _gather_sequence_batch(
    starts: np.ndarray,
    seq_len: int,
    obs: np.ndarray | Mapping[str, np.ndarray],
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    terminals: np.ndarray,
    normalize: bool | Mapping[str, bool],
    keep_uint8_on_sample: bool | Mapping[str, bool],
) -> ReplayBatch:
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
        "obs": _gather_obs(obs, indices, normalize, keep_uint8_on_sample),
        "actions": jnp.array(actions_b, dtype=jnp.int32),
        "rewards": jnp.array(rewards_b, dtype=jnp.float32),
        "dones": jnp.array(dones_b, dtype=jnp.float32),
        "terminals": jnp.array(terminals_b, dtype=jnp.float32),
        "is_first": jnp.array(is_first, dtype=jnp.float32),
    }


def _np_dtype(name: str) -> type:
    if name == "uint8":
        return np.uint8
    if name == "float16":
        return np.float16
    return np.float32


def _single_field_spec(config: BufferConfig) -> _FieldSpec:
    if not isinstance(config.obs_shape, tuple):
        raise TypeError("single-field BufferConfig requires tuple obs_shape")
    if not isinstance(config.obs_dtype, str):
        raise TypeError("single-field BufferConfig requires string obs_dtype")
    if not isinstance(config.normalize_obs, bool):
        raise TypeError("single-field BufferConfig requires bool normalize_obs")
    return _FieldSpec(
        shape=config.obs_shape,
        dtype=config.obs_dtype,
        normalize=config.normalize_obs and config.obs_dtype == "uint8",
        keep_uint8_on_sample=False,
    )


def _mapping_value(mapping_or_scalar, key: str, default):
    if isinstance(mapping_or_scalar, Mapping):
        return mapping_or_scalar.get(key, default)
    return mapping_or_scalar


def _field_specs(config: BufferConfig) -> dict[str, _FieldSpec] | None:
    if not isinstance(config.obs_shape, Mapping):
        return None
    specs: dict[str, _FieldSpec] = {}
    for key, shape in config.obs_shape.items():
        dtype = _mapping_value(config.obs_dtype, key, "float32")
        normalize = bool(_mapping_value(config.normalize_obs, key, False))
        specs[key] = _FieldSpec(
            shape=tuple(shape),
            dtype=str(dtype),
            normalize=normalize and dtype == "uint8",
            keep_uint8_on_sample=(dtype == "uint8" and not normalize),
        )
    return specs


def _to_jax_obs(
    obs_b: np.ndarray,
    normalize: bool,
    keep_uint8_on_sample: bool,
) -> jnp.ndarray:
    """Convert sampled replay storage to the dtype expected downstream.

    Feature observations always leave replay as float32, even when stored as
    float16. The only deferred-cast case is a modal uint8 image field: hybrid
    training normalizes that image later in obs_batch, after the modalities are
    still inspectable as separate fields.
    """
    if normalize:
        return jnp.array(obs_b, dtype=jnp.float32) / 255.0
    if keep_uint8_on_sample:
        return jnp.asarray(obs_b)
    return jnp.array(obs_b, dtype=jnp.float32)


def _gather_obs(
    obs: np.ndarray | Mapping[str, np.ndarray],
    indices: np.ndarray,
    normalize: bool | Mapping[str, bool],
    keep_uint8_on_sample: bool | Mapping[str, bool],
) -> jnp.ndarray | dict[str, jnp.ndarray]:
    """Gather observation windows and apply each field's sample conversion."""
    if isinstance(obs, Mapping):
        if not isinstance(normalize, Mapping):
            raise TypeError("mapping observations require mapping normalize flags")
        if not isinstance(keep_uint8_on_sample, Mapping):
            raise TypeError("mapping observations require mapping uint8 sample flags")
        return {
            key: _to_jax_obs(
                value[indices],
                bool(normalize.get(key, False)),
                bool(keep_uint8_on_sample.get(key, False)),
            )
            for key, value in obs.items()
        }
    if not isinstance(normalize, bool):
        raise TypeError("single observations require a bool normalize flag")
    if not isinstance(keep_uint8_on_sample, bool):
        raise TypeError("single observations require a bool uint8 sample flag")
    return _to_jax_obs(obs[indices], normalize, keep_uint8_on_sample)


class ReplayBuffer:
    """Ring buffer that stores transitions and samples fixed-length sequences.

    Stores uint8 images, float features, or structured multi-modal observations.
    """

    def __init__(self, config: BufferConfig) -> None:
        cap = config.capacity
        field_specs = _field_specs(config)
        if field_specs is None:
            spec = _single_field_spec(config)
            self.obs = np.zeros((cap, *spec.shape), dtype=_np_dtype(spec.dtype))
            self._normalize: bool | dict[str, bool] = spec.normalize
            self._keep_uint8_on_sample: bool | dict[str, bool] = (
                spec.keep_uint8_on_sample
            )
        else:
            self.obs = {
                key: np.zeros((cap, *spec.shape), dtype=_np_dtype(spec.dtype))
                for key, spec in field_specs.items()
            }
            self._normalize = {key: spec.normalize for key, spec in field_specs.items()}
            self._keep_uint8_on_sample = {
                key: spec.keep_uint8_on_sample for key, spec in field_specs.items()
            }
        self.actions = np.zeros(cap, dtype=np.int32)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.dones = np.zeros(cap, dtype=np.bool_)
        self.terminals = np.zeros(cap, dtype=np.bool_)
        self.capacity = cap
        self.idx = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray | Mapping[str, np.ndarray],
        action: int,
        reward: float,
        done: bool,
        terminal: bool = False,
    ) -> None:
        if isinstance(self.obs, Mapping):
            if not isinstance(obs, Mapping):
                raise TypeError("mapping replay buffer requires mapping obs")
            missing = set(self.obs) - set(obs)
            if missing:
                raise KeyError(f"obs missing replay fields: {sorted(missing)}")
            for key, storage in self.obs.items():
                storage[self.idx] = obs[key]
        else:
            if isinstance(obs, Mapping):
                raise TypeError("single replay buffer requires array obs")
            self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.terminals[self.idx] = terminal
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch:
        starts = self._sample_starts(batch_size, seq_len)
        return _gather_sequence_batch(
            starts,
            seq_len,
            self.obs,
            self.actions,
            self.rewards,
            self.dones,
            self.terminals,
            self._normalize,
            self._keep_uint8_on_sample,
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
        self.obs = data["obs"]  # (N, ...) uint8 or float32
        self.actions = data["actions"]  # (N,) int32
        self.rewards = data["rewards"]  # (N,) float32
        self.dones = data["dones"]  # (N,) bool
        self.terminals = data["terminals"]  # (N,) bool

        # Reconstruct episode boundaries from dones
        # Episode starts at index 0 and after every done
        done_indices = np.where(self.dones)[0]
        self._ep_starts = np.concatenate([[0], done_indices + 1])
        # Remove any start that would be past the end of data
        self._ep_starts = self._ep_starts[self._ep_starts < len(self.obs)]
        # Episode lengths
        ends = np.concatenate([done_indices + 1, [len(self.obs)]])
        self._ep_lengths = ends[: len(self._ep_starts)] - self._ep_starts

        print(
            f"ValReplayDataset: {len(self.obs)} steps, "
            f"{len(self._ep_starts)} episodes from {path}"
        )

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
        offsets = np.array(
            [np.random.randint(0, ep_len - seq_len + 1) for ep_len in ep_lens]
        )
        starts = ep_starts + offsets
        return _gather_sequence_batch(
            starts,
            seq_len,
            self.obs,
            self.actions,
            self.rewards,
            self.dones,
            self.terminals,
            self._normalize,
            False,
        )
