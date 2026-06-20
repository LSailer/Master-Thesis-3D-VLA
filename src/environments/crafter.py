"""Crafter environment wrapper compatible with our DreamerV3 training loop."""

import numpy as np

from src.environments.observation import ObservationFrame


class CrafterEnv:
    def __init__(self, size=(64, 64), seed=None):
        import crafter

        self._env = crafter.Env(size=size, reward=True, seed=seed)
        self.num_actions = self._env.action_space.n  # 17

    def reset(self) -> ObservationFrame:
        obs = self._env.reset()  # (H, W, C) uint8
        return ObservationFrame(
            image=np.transpose(obs, (2, 0, 1)),  # CHW
            is_first=True,
        )

    def step(self, action) -> ObservationFrame:
        obs, reward, done, _info = self._env.step(action)
        return ObservationFrame(
            image=np.transpose(obs, (2, 0, 1)),  # CHW
            reward=float(reward),
            done=bool(done),
            is_first=False,
            is_last=bool(done),
        )

    def close(self):
        pass
