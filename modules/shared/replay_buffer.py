"""Unified numpy ring buffer with sequence sampling."""

from __future__ import annotations

from dataclasses import dataclass

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
    obs_shape: tuple[int, ...]
    obs_dtype: str = "uint8"
    normalize_obs: bool = True


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
        np_dtype = np.uint8 if config.obs_dtype == "uint8" else np.float32
        self.obs = np.zeros((cap, *config.obs_shape), dtype=np_dtype)
        self.actions = np.zeros(cap, dtype=np.int32)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.dones = np.zeros(cap, dtype=np.bool_)
        self.terminals = np.zeros(cap, dtype=np.bool_)
        self._normalize = config.normalize_obs and config.obs_dtype == "uint8"
        self.capacity = cap
        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,
            terminal: bool = False) -> None:
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.terminals[self.idx] = terminal
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int) -> dict[str, jnp.ndarray]:
        if self.size < self.capacity:
            # Buffer hasn't wrapped — data in [0, size) is contiguous
            n_valid = self.size - seq_len + 1
            assert n_valid > 0, "Not enough data in buffer"
            starts = np.random.randint(0, n_valid, size=batch_size)
        else:
            # Buffer has wrapped — avoid sequences crossing the write head.
            # Safe regions: [0, idx-seq_len] (new) and [idx, cap-seq_len] (old)
            n_new = max(0, self.idx - seq_len + 1)
            n_old = max(0, self.capacity - seq_len - self.idx + 1)
            n_valid = n_new + n_old
            assert n_valid > 0, "Not enough contiguous data in buffer"
            raw = np.random.randint(0, n_valid, size=batch_size)
            starts = np.where(raw < n_new, raw, raw - n_new + self.idx)
        indices = starts[:, None] + np.arange(seq_len)[None, :]  # (B, T)

        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        terminals = self.terminals[indices]

        is_first = np.zeros_like(dones)
        is_first[:, 0] = True
        is_first[:, 1:] = dones[:, :-1]

        obs_jnp = jnp.array(obs, dtype=jnp.float32)
        if self._normalize:
            obs_jnp = obs_jnp / 255.0

        return {
            "obs": obs_jnp,
            "actions": jnp.array(actions, dtype=jnp.int32),
            "rewards": jnp.array(rewards, dtype=jnp.float32),
            "dones": jnp.array(dones, dtype=jnp.float32),
            "terminals": jnp.array(terminals, dtype=jnp.float32),
            "is_first": jnp.array(is_first, dtype=jnp.float32),
        }


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
        indices = starts[:, None] + np.arange(seq_len)[None, :]

        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        terminals = self.terminals[indices]

        is_first = np.zeros_like(dones)
        is_first[:, 0] = True
        is_first[:, 1:] = dones[:, :-1]

        obs_jnp = jnp.array(obs, dtype=jnp.float32)
        if self._normalize:
            obs_jnp = obs_jnp / 255.0

        return {
            "obs": obs_jnp,
            "actions": jnp.array(actions, dtype=jnp.int32),
            "rewards": jnp.array(rewards, dtype=jnp.float32),
            "dones": jnp.array(dones, dtype=jnp.float32),
            "terminals": jnp.array(terminals, dtype=jnp.float32),
            "is_first": jnp.array(is_first, dtype=jnp.float32),
        }


def VGGTReplayBuffer(capacity: int, feature_dim: int = 4116) -> ReplayBuffer:
    """Backward-compat factory. Use ReplayBuffer(BufferConfig(...)) directly."""
    return ReplayBuffer(BufferConfig(
        capacity=capacity,
        obs_shape=(feature_dim,),
        obs_dtype="float32",
        normalize_obs=False,
    ))
