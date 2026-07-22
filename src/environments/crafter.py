"""Crafter environment wrapper compatible with our DreamerV3 training loop."""

from typing import Any, cast

import jax.numpy as jnp

from src.environments.observation import ObservationFrame


class CrafterEnv:
    def __init__(self, size=(64, 64), seed=None):
        import crafter

        crafter_module = cast(Any, crafter)
        self._env = crafter_module.Env(size=size, reward=True, seed=seed)
        self.num_actions = self._env.action_space.n  # 17

    def reset(self) -> ObservationFrame:
        obs = self._env.reset()  # (H, W, C) uint8
        return ObservationFrame(
            image=obs,  # HWC
            is_first=True,
        )

    def step(self, action) -> ObservationFrame:
        obs, reward, done, _info = self._env.step(action)
        return ObservationFrame(
            image=obs,  # HWC
            is_first=False,
            previous_action=int(action),
            reward=float(reward),
            done=bool(done),
        )

    def close(self):
        pass
