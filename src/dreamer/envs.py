"""Habitat ObjectNav environment wrapper for DreamerV3.

Provides a simple step/reset interface with combined RGB+depth observations.
"""

import numpy as np


class HabitatObjectNavEnv:
    """Wraps Habitat ObjectNav into a simple RL interface.

    Observation: (4, 256, 256) float32 — RGB (3ch, [0,1]) + depth (1ch).
    Actions: 0=FORWARD, 1=TURN_LEFT, 2=TURN_RIGHT, 3=STOP (discrete).
    """

    ACTION_NAMES = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]

    def __init__(self, scene_dataset: str = "data/scene_datasets/hm3d_minival", split: str = "val_mini"):
        self._split = split
        self._scene_dataset = scene_dataset
        self._env = None
        self._setup()

    def _setup(self):
        try:
            import habitat
            from habitat.config.default_structured_configs import (
                HabitatConfigPlugin,
                TaskConfig,
            )
            from omegaconf import OmegaConf

            config = habitat.get_config(
                "benchmark/nav/objectnav/objectnav_v2_hm3d_stretch.yaml"
            )
            with habitat.config.read_write(config):
                config.habitat.dataset.split = self._split
                config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 256
                config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 256
                config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = 256
                config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = 256
            self._env = habitat.Env(config=config)
        except ImportError:
            self._env = None

    @property
    def observation_shape(self) -> tuple:
        return (4, 256, 256)

    @property
    def action_size(self) -> int:
        return 4

    def _process_obs(self, obs: dict) -> np.ndarray:
        """Combine RGB + depth into (4, 256, 256) float32."""
        rgb = obs["rgb"].astype(np.float32) / 255.0  # (H, W, 3) → [0, 1]
        depth = obs["depth"].astype(np.float32)  # (H, W, 1)
        combined = np.concatenate([rgb, depth], axis=-1)  # (H, W, 4)
        return combined.transpose(2, 0, 1)  # (4, H, W)

    def reset(self) -> np.ndarray:
        if self._env is None:
            return np.zeros(self.observation_shape, dtype=np.float32)
        obs = self._env.reset()
        return self._process_obs(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self._env is None:
            return (
                np.zeros(self.observation_shape, dtype=np.float32),
                0.0,
                False,
                {},
            )
        obs = self._env.step(action)
        done = self._env.episode_over
        info = self._env.get_metrics()
        reward = info.get("success", 0.0)  # sparse reward: 1 on success
        return self._process_obs(obs), float(reward), done, info

    def close(self):
        if self._env is not None:
            self._env.close()


class DummyEnv:
    """Dummy environment for testing without Habitat installed."""

    def __init__(self, obs_shape: tuple = (4, 256, 256), action_size: int = 4):
        self._obs_shape = obs_shape
        self._action_size = action_size
        self._step_count = 0

    @property
    def observation_shape(self) -> tuple:
        return self._obs_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return np.random.rand(*self._obs_shape).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self._step_count += 1
        obs = np.random.rand(*self._obs_shape).astype(np.float32)
        done = self._step_count >= 500
        reward = float(np.random.rand() * 0.1)
        return obs, reward, done, {"step": self._step_count}

    def close(self):
        pass
