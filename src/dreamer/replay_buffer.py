"""Simple numpy replay buffer with sequence sampling for DreamerV3."""

import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    """Circular replay buffer storing transitions as flat numpy arrays.

    Samples contiguous sequences for RSSM training.
    """

    def __init__(self, obs_shape: tuple, action_size: int, capacity: int):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_size = action_size

        self.observations = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)

        self.index = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.index

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        self.observations[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = float(done)

        self.index = (self.index + 1) % self.capacity
        self.full = self.full or self.index == 0

    def sample(self, batch_size: int, seq_len: int) -> dict[str, jnp.ndarray]:
        """Sample batch of contiguous sequences.

        Returns dict with arrays of shape (batch_size, seq_len, ...).
        """
        max_start = len(self) - seq_len
        assert max_start >= batch_size, "Not enough data in buffer"

        starts = np.random.randint(0, max_start, size=batch_size)
        # Build index array: (batch_size, seq_len)
        offsets = np.arange(seq_len)[None, :]
        indices = (starts[:, None] + offsets) % self.capacity

        return {
            "observations": jnp.array(self.observations[indices]),
            "actions": jnp.array(self.actions[indices]),
            "rewards": jnp.array(self.rewards[indices]),
            "dones": jnp.array(self.dones[indices]),
        }
