"""Habitat ObjectNav env adapter for PyTorch r2dreamer's interface."""
import numpy as np


class HabitatR2DreamerEnv:
    def __init__(self, obs_size=64, split="train", max_episode_steps=500,
                 max_geodesic=None, reward_type="geodesic_delta"):
        from modules.dreamerv3.configs import DreamerConfig
        from modules.envs.habitat import HabitatObjectNavEnv
        config = DreamerConfig(
            obs_shape=(3, obs_size, obs_size),
            max_episode_steps=max_episode_steps,
            split=split, reward_type=reward_type)
        self._env = HabitatObjectNavEnv(config, max_geodesic=max_geodesic)
        self.num_actions = 4

    def reset(self):
        obs = self._env.reset()
        image = np.transpose(obs["image"], (1, 2, 0))  # CHW->HWC
        return {"image": image, "reward": np.float32(0.0),
                "is_first": True, "is_last": False, "is_terminal": False}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(np.argmax(action))
        obs = self._env.step(action)
        image = np.transpose(obs["image"], (1, 2, 0))
        done = obs["done"]
        return {"image": image, "reward": np.float32(obs["reward"]),
                "is_first": False, "is_last": done, "is_terminal": done}

    def close(self):
        self._env.close()
