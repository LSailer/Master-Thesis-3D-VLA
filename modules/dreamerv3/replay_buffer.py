"""Simple numpy ring buffer with sequence sampling."""

import numpy as np
import jax.numpy as jnp

from .configs import DreamerConfig


class ReplayBuffer:
    def __init__(self, config: DreamerConfig):
        cap = config.buffer_capacity
        H, W = config.obs_shape[1], config.obs_shape[2]
        self.obs = np.zeros((cap, 3, H, W), dtype=np.uint8)
        self.actions = np.zeros(cap, dtype=np.int32)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.dones = np.zeros(cap, dtype=np.bool_)
        self.terminals = np.zeros(cap, dtype=np.bool_)
        self.capacity = cap
        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,
            terminal: bool = False):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.terminals[self.idx] = terminal
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int) -> dict:
        max_start = self.size - seq_len
        assert max_start > 0, "Not enough data in buffer"
        starts = np.random.randint(0, max_start, size=batch_size)
        indices = starts[:, None] + np.arange(seq_len)[None, :]  # (B, T)

        obs = self.obs[indices]  # (B, T, C, H, W)
        actions = self.actions[indices]  # (B, T)
        rewards = self.rewards[indices]  # (B, T)
        dones = self.dones[indices]  # (B, T)
        terminals = self.terminals[indices]  # (B, T)

        # is_first: True at t=0 of each sequence and after any done
        is_first = np.zeros_like(dones)
        is_first[:, 0] = True
        is_first[:, 1:] = dones[:, :-1]

        return {
            "obs": jnp.array(obs, dtype=jnp.float32) / 255.0,
            "actions": jnp.array(actions, dtype=jnp.int32),
            "rewards": jnp.array(rewards, dtype=jnp.float32),
            "dones": jnp.array(dones, dtype=jnp.float32),
            "terminals": jnp.array(terminals, dtype=jnp.float32),
            "is_first": jnp.array(is_first, dtype=jnp.float32),
        }


class VGGTReplayBuffer:
    """Replay buffer for VGGT features (flat float32 vectors)."""

    def __init__(self, capacity: int, feature_dim: int = 4116):
        self.obs = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.terminals = np.zeros(capacity, dtype=np.bool_)
        self.capacity = capacity
        self.idx = 0
        self.size = 0

    def add(self, features: np.ndarray, action: int, reward: float,
            done: bool, terminal: bool = False):
        self.obs[self.idx] = features
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.terminals[self.idx] = terminal
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int) -> dict:
        if self.idx < self.size:  # buffer has wrapped
            max_start = self.idx - seq_len
        else:
            max_start = self.size - seq_len
        assert max_start > 0, "Not enough contiguous data in buffer"
        starts = np.random.randint(0, max_start, size=batch_size)
        indices = starts[:, None] + np.arange(seq_len)[None, :]

        obs = self.obs[indices]  # (B, T, feature_dim)
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        terminals = self.terminals[indices]

        is_first = np.zeros_like(dones)
        is_first[:, 0] = True
        is_first[:, 1:] = dones[:, :-1]

        return {
            "obs": jnp.array(obs, dtype=jnp.float32),  # no /255 normalization
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

    def __init__(self, path: str):
        data = np.load(path)
        self.obs = data["obs"]          # (N, C, H, W) uint8
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

        return {
            "obs": jnp.array(obs, dtype=jnp.float32) / 255.0,
            "actions": jnp.array(actions, dtype=jnp.int32),
            "rewards": jnp.array(rewards, dtype=jnp.float32),
            "dones": jnp.array(dones, dtype=jnp.float32),
            "terminals": jnp.array(terminals, dtype=jnp.float32),
            "is_first": jnp.array(is_first, dtype=jnp.float32),
        }
