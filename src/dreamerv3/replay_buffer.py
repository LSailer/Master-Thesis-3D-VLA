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
        self.capacity = cap
        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
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

        # is_first: True at t=0 of each sequence and after any done
        is_first = np.zeros_like(dones)
        is_first[:, 0] = True
        is_first[:, 1:] = dones[:, :-1]

        return {
            "obs": jnp.array(obs, dtype=jnp.float32) / 255.0,
            "actions": jnp.array(actions, dtype=jnp.int32),
            "rewards": jnp.array(rewards, dtype=jnp.float32),
            "dones": jnp.array(dones, dtype=jnp.float32),
            "is_first": jnp.array(is_first, dtype=jnp.float32),
        }
