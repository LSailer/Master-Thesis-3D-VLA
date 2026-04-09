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

# Success radius (meters) — geodesic distance to nearest viewpoint.
# 0.2m is the tightest threshold that gives 100% SR for the optimal agent
# with 0.25m discrete steps (see goal_distance_analysis notebook).
GOAL_RADIUS = 0.2


def find_nearest_viewpoint(env):
    """Find nearest viewpoint across all goal instances of a habitat.Env.

    Returns (position, goal_index) or (None, 0) if no viewpoints exist.
    """
    agent_pos = env.sim.get_agent_state().position
    best_dist = float("inf")
    best_pos = None
    best_idx = 0
    for gi, goal in enumerate(env.current_episode.goals):
        if goal.view_points:
            for vp in goal.view_points:
                d = env.sim.geodesic_distance(
                    agent_pos, vp.agent_state.position
                )
                if d < best_dist:
                    best_dist = d
                    best_pos = vp.agent_state.position
                    best_idx = gi
    return best_pos, best_idx


def sample_navmesh(env, resolution: float = 0.05) -> dict:
    """Sample navigable area at current agent height for top-down visualization.

    Works with a raw habitat.Env. Returns dict with 'grid' (bool array),
    bounds, and resolution.
    """
    agent_y = env.sim.get_agent_state().position[1]
    pathfinder = env.sim.pathfinder
    bounds = pathfinder.get_bounds()
    x_min, z_min = bounds[0][0], bounds[0][2]
    x_max, z_max = bounds[1][0], bounds[1][2]

    xs = np.arange(x_min, x_max, resolution)
    zs = np.arange(z_min, z_max, resolution)
    grid = np.zeros((len(zs), len(xs)), dtype=bool)

    for zi, z in enumerate(zs):
        for xi, x in enumerate(xs):
            grid[zi, xi] = pathfinder.is_navigable(
                np.array([x, agent_y, z]), max_y_delta=0.5
            )

    return {
        "grid": grid,
        "x_min": float(x_min), "x_max": float(x_max),
        "z_min": float(z_min), "z_max": float(z_max),
        "resolution": resolution,
    }


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
        self._start_geodesic = 0.0
        self._path_length = 0.0
        self._prev_position = None

    def reset(self) -> dict:
        obs = self._env.reset()
        self._prev_dist = self._env.get_metrics().get("distance_to_goal", 0.0)
        self._start_geodesic = self._prev_dist
        self._prev_position = np.array(self._env.sim.get_agent_state().position)
        self._path_length = 0.0
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

        current_position = np.array(self._env.sim.get_agent_state().position)
        self._path_length += np.linalg.norm(current_position - self._prev_position)
        self._prev_position = current_position

        dist = metrics.get("distance_to_goal", float("inf"))
        reward = self._compute_reward(dist)
        success = 1.0 if dist < GOAL_RADIUS else 0.0
        done = success > 0 or self._step_count >= self._cfg.max_episode_steps

        spl = 0.0
        if done and success > 0:
            shortest = self._start_geodesic
            spl = shortest / max(shortest, self._path_length) if shortest > 0 else 0.0

        image = self._obs_to_image(obs)

        return {
            "image": image,
            "reward": reward,
            "done": done,
            "is_first": False,
            "success": success,
            "spl": spl,
        }

    def find_nearest_viewpoint(self):
        """Find nearest viewpoint. Delegates to module-level function."""
        return find_nearest_viewpoint(self._env)

    def sample_navmesh(self, resolution: float = 0.05) -> dict:
        """Sample navigable area. Delegates to module-level function."""
        return sample_navmesh(self._env, resolution)

    def _obs_to_image(self, obs) -> np.ndarray:
        rgb = obs["rgb"][:, :, :3]  # (H, W, 3) uint8
        return np.transpose(rgb, (2, 0, 1))  # (3, H, W)

    def _compute_reward(self, dist: float) -> float:
        if self._cfg.reward_type == "sparse":
            return 10.0 * (1.0 if dist < GOAL_RADIUS else 0.0)

        reward = self._prev_dist - dist  # geodesic delta
        self._prev_dist = dist
        if dist < GOAL_RADIUS:
            reward += 10.0
        return reward

    def close(self):
        self._env.close()
