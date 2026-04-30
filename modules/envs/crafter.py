"""Crafter environment wrapper compatible with our DreamerV3 training loop."""

import numpy as np


class CrafterEnv:
    def __init__(self, size=(64, 64), seed=None):
        import crafter
        self._env = crafter.Env(size=size, reward=True, seed=seed)
        self.num_actions = self._env.action_space.n  # 17

    def reset(self):
        obs = self._env.reset()  # (H, W, C) uint8
        return {
            "image": np.transpose(obs, (2, 0, 1)),  # CHW
            "is_first": True,
            "reward": 0.0,
            "done": False,
        }

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return {
            "image": np.transpose(obs, (2, 0, 1)),  # CHW
            "reward": float(reward),
            "done": done,
            "is_first": False,
        }

    def close(self):
        pass
