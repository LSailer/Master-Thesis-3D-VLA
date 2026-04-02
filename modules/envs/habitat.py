"""Thin Habitat ObjectNav wrapper for DreamerV3.

STOP is treated as a no-op (no movement, no termination) following DreamerNav.
Episodes terminate on: (1) agent within goal_radius of target, or
(2) max_episode_steps exceeded.
"""

from pathlib import Path

import numpy as np

from modules.dreamerv3.configs import DreamerConfig

# Discrete actions: STOP is a no-op (no movement), kept for action-space parity
ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}

SCENE_DIR = Path("data/scene_datasets/hm3d")
DATA_DIR = Path("data/datasets/objectnav/hm3d/objectnav_hm3d_v2")

# Goal reaching radius (meters) — episode terminates when agent is within this
GOAL_RADIUS = 0.1


class HabitatObjectNavEnv:
    def __init__(self, config: DreamerConfig, max_geodesic: float | None = None):
        import habitat

        self._cfg = config
        H, W = config.obs_shape[1], config.obs_shape[2]
        split = config.split

        hab_cfg = habitat.get_config(
            "benchmark/nav/objectnav/objectnav_hm3d.yaml"
        )
        with habitat.config.read_write(hab_cfg):
            hab_cfg.habitat.dataset.split = split
            hab_cfg.habitat.dataset.data_path = str(
                DATA_DIR / "{split}" / "{split}.json.gz"
            )
            hab_cfg.habitat.dataset.scenes_dir = "data/scene_datasets"
            scene_cfg = next(SCENE_DIR.rglob("*scene_dataset_config.json"), None)
            if scene_cfg:
                hab_cfg.habitat.simulator.scene_dataset = str(scene_cfg)
            agent_cfg = hab_cfg.habitat.simulator.agents.main_agent
            agent_cfg.sim_sensors.rgb_sensor.height = H
            agent_cfg.sim_sensors.rgb_sensor.width = W
            hab_cfg.habitat.environment.max_episode_steps = config.max_episode_steps

        self._env = habitat.Env(config=hab_cfg)

        if max_geodesic is not None:
            before = len(self._env._dataset.episodes)
            self._env._dataset.episodes = [
                ep for ep in self._env._dataset.episodes
                if ep.info is not None
                and ep.info.get("geodesic_distance", float("inf")) < max_geodesic
            ]
            self._env._setup_episode_iterator()
            self._env.current_episode = next(self._env.episode_iterator)
            print(f"Filtered: {before} → {len(self._env._dataset.episodes)} "
                  f"episodes (geodesic < {max_geodesic}m)")

        self._prev_dist = 0.0
        self._step_count = 0
        self._last_obs = None

    def reset(self) -> dict:
        obs = self._env.reset()
        self._prev_dist = self._env.get_metrics().get("distance_to_goal", 0.0)
        self._step_count = 0
        image = self._obs_to_image(obs)
        self._last_obs = obs
        return {"image": image, "is_first": True, "reward": 0.0, "done": False}

    def step(self, action: int) -> dict:
        # STOP (action 0) is a no-op: no movement, no termination
        if action == 0:
            self._step_count += 1
            image = self._obs_to_image(self._last_obs)
            done = self._step_count >= self._cfg.max_episode_steps
            return {
                "image": image,
                "reward": 0.0,
                "done": done,
                "is_first": False,
                "success": 0.0,
                "spl": 0.0,
            }

        obs = self._env.step(action=action)
        self._step_count += 1
        self._last_obs = obs
        metrics = self._env.get_metrics()

        reward = self._compute_reward(metrics)
        dist = metrics.get("distance_to_goal", float("inf"))
        success = 1.0 if dist < GOAL_RADIUS else 0.0
        done = success > 0 or self._step_count >= self._cfg.max_episode_steps
        spl = metrics.get("spl", 0.0) if success > 0 else 0.0
        image = self._obs_to_image(obs)

        return {
            "image": image,
            "reward": reward,
            "done": done,
            "is_first": False,
            "success": success,
            "spl": spl,
        }

    def _obs_to_image(self, obs) -> np.ndarray:
        rgb = obs["rgb"][:, :, :3]  # (H, W, 3) uint8
        return np.transpose(rgb, (2, 0, 1))  # (3, H, W)

    def _compute_reward(self, metrics: dict) -> float:
        if self._cfg.reward_type == "sparse":
            return 10.0 * (1.0 if metrics.get("distance_to_goal", float("inf")) < GOAL_RADIUS else 0.0)

        curr_dist = metrics.get("distance_to_goal", 0.0)
        reward = self._prev_dist - curr_dist  # geodesic delta
        self._prev_dist = curr_dist
        if curr_dist < GOAL_RADIUS:
            reward += 10.0
        return reward

    def close(self):
        self._env.close()
