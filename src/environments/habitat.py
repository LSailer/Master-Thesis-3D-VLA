"""Thin Habitat ObjectNav wrapper for DreamerV3.

STOP is treated as a no-op (no movement, no termination) following DreamerNav.
Episodes terminate on: (1) agent within goal_radius of target, or
(2) max_episode_steps exceeded.
"""

import sys
from pathlib import Path

import numpy as np

from src.shared.configs import DreamerConfig

# Discrete actions: STOP is a no-op (no movement), kept for action-space parity
ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}

SCENE_DIR = Path("data/scene_datasets/hm3d")
DATA_DIR = Path("data/datasets/objectnav/hm3d/objectnav_hm3d_v2")

# Success radius (meters) — geodesic distance to nearest viewpoint.
# 0.2m is the tightest threshold that gives 100% SR for the optimal agent
# with 0.25m discrete steps (see goal_distance_analysis notebook).
GOAL_RADIUS = 0.2


def _validate_goal_distance(dist: float) -> float:
    if dist is None:
        raise ValueError("distance_to_goal must be a finite non-negative value, got None")
    dist = float(dist)
    if not np.isfinite(dist) or dist < 0.0:
        raise ValueError(
            "distance_to_goal must be a finite non-negative value, "
            f"got {dist!r}"
        )
    return dist


def _is_success_distance(dist: float) -> bool:
    return _validate_goal_distance(dist) < GOAL_RADIUS


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
    def __init__(self, config: DreamerConfig, max_geodesic: float | None = None,
                 step_counts_path: str | None = None, semantic: bool = False,
                 curriculum_path: str | None = None,
                 curriculum_mode: str = "train", seed: int | None = None):
        import habitat
        from omegaconf import OmegaConf

        self._cfg = config
        H, W = config.obs_shape[1], config.obs_shape[2]
        split = config.split

        # Curriculum episodes always come from the train split
        curriculum = None
        if curriculum_path is not None:
            import json
            split = "train"
            with open(curriculum_path) as f:
                curriculum = json.load(f)

        hab_cfg = habitat.get_config(
            "benchmark/nav/objectnav/objectnav_hm3d.yaml"
        )
        with habitat.config.read_write(hab_cfg):
            hab_cfg.habitat.dataset.split = split
            if seed is not None:
                hab_cfg.habitat.seed = int(seed)
            hab_cfg.habitat.dataset.data_path = str(
                DATA_DIR / "{split}" / "{split}.json.gz"
            )
            hab_cfg.habitat.dataset.scenes_dir = "data/scene_datasets"
            # Pre-filter: only load scene files listed in curriculum
            if curriculum is not None:
                hab_cfg.habitat.dataset.content_scenes = curriculum["scenes"]
            scene_cfg = next(SCENE_DIR.rglob("*scene_dataset_config.json"), None)
            if scene_cfg:
                hab_cfg.habitat.simulator.scene_dataset = str(scene_cfg)
            agent_cfg = hab_cfg.habitat.simulator.agents.main_agent
            agent_cfg.sim_sensors.rgb_sensor.height = H
            agent_cfg.sim_sensors.rgb_sensor.width = W
            hab_cfg.habitat.environment.max_episode_steps = config.max_episode_steps
            # load_semantic_mesh is a habitat-sim attr not in the OmegaConf schema
            OmegaConf.set_struct(hab_cfg.habitat.simulator, False)
            hab_cfg.habitat.simulator.load_semantic_mesh = semantic
            OmegaConf.set_struct(hab_cfg.habitat.simulator, True)

        self._env = habitat.Env(config=hab_cfg)
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(int(seed))

        if curriculum is not None:

            # Keys are [episode_id, object_category, scene_name] triples
            key_set = {(k[0], k[1], k[2])
                       for k in curriculum[f"{curriculum_mode}_episode_keys"]}
            before = len(self._env._dataset.episodes)
            self._env._dataset.episodes = [
                ep for ep in self._env._dataset.episodes
                if (ep.episode_id, ep.object_category,
                    ep.scene_id.split("/")[-1].replace(".basis.glb", ""))
                in key_set
            ]
            after = len(self._env._dataset.episodes)
            assert after > 0, (
                f"Curriculum filter matched 0 episodes for mode='{curriculum_mode}'. "
                f"Check that curriculum JSON keys match the dataset split."
            )
            self._env._setup_episode_iterator()
            self._env.current_episode = next(self._env.episode_iterator)
            print(f"Curriculum [{curriculum['name']}] {curriculum_mode}: "
                  f"{before} → {after} episodes")
        else:
            if max_geodesic is not None:
                before = len(self._env._dataset.episodes)
                self._env._dataset.episodes = [
                    ep for ep in self._env._dataset.episodes
                    if ep.info is not None
                    and ep.info.get("geodesic_distance", float("inf")) < max_geodesic
                ]
                after = len(self._env._dataset.episodes)
                assert after > 0, (
                    f"max_geodesic={max_geodesic} filtered out all episodes."
                )
                self._env._setup_episode_iterator()
                self._env.current_episode = next(self._env.episode_iterator)
                print(f"Filtered: {before} → {after} "
                      f"episodes (geodesic < {max_geodesic}m)")

            if step_counts_path is not None:
                import json
                with open(step_counts_path) as f:
                    step_counts = json.load(f)
                split_counts = step_counts.get(config.split, {})
                before = len(self._env._dataset.episodes)
                self._env._dataset.episodes = [
                    ep for ep in self._env._dataset.episodes
                    if split_counts.get(ep.episode_id, 0) < 200
                ]
                after = len(self._env._dataset.episodes)
                assert after > 0, (
                    "step_counts filter removed all episodes — check split key in JSON."
                )
                self._env._setup_episode_iterator()
                self._env.current_episode = next(self._env.episode_iterator)
                print(f"Filtered (step count): {before} → "
                      f"{after} episodes (steps < 200)")

        self._prev_dist = 0.0
        self._step_count = 0
        self._last_obs = None
        self._start_geodesic = 0.0
        self._path_length = 0.0
        self._prev_position = None
        self._collisions = 0
        self._forward_steps = 0

    def reset(self) -> dict:
        for attempt in range(100):
            obs = self._env.reset()
            raw_dist = self._env.get_metrics().get("distance_to_goal", 0.0)
            try:
                dist = _validate_goal_distance(raw_dist)
                break
            except ValueError:
                self._log_invalid_goal_distance(
                    raw_dist, phase=f"reset attempt {attempt + 1}"
                )
        else:
            raise RuntimeError(
                "Habitat reset did not find an episode with finite "
                "distance_to_goal after 100 attempts"
            )

        self._prev_dist = dist
        self._start_geodesic = self._prev_dist
        self._prev_position = np.array(self._env.sim.get_agent_state().position)
        self._path_length = 0.0
        self._step_count = 0
        self._collisions = 0
        self._forward_steps = 0
        image = self._obs_to_image(obs)
        self._last_obs = obs
        return {"image": image, "is_first": True, "reward": 0.0, "done": False}

    def step(self, action: int) -> dict:
        # STOP (action 0) is a no-op: no movement, no termination
        if action == 0:
            self._step_count += 1
            image = self._obs_to_image(self._last_obs)
            done = self._step_count >= self._cfg.max_episode_steps
            # If the timeout fires on STOP, the agent ended at _prev_dist
            # with the current path_length — surface SoftSPL/DTG accordingly.
            return {
                "image": image,
                "reward": self._cfg.step_penalty,
                "done": done,
                "is_first": False,
                "success": 0.0,
                "spl": 0.0,
                **self._episode_end_metrics(self._prev_dist, done),
            }

        obs = self._env.step(action=action)
        self._step_count += 1
        self._last_obs = obs
        metrics = self._env.get_metrics()

        current_position = np.array(self._env.sim.get_agent_state().position)
        delta = float(np.linalg.norm(current_position - self._prev_position))
        self._path_length += delta
        # Habitat has no direct collision API. Workaround: a FORWARD step that
        # moves the agent < 0.01 m is treated as a collision. Nominal forward
        # step is 0.25 m, so genuine motion is ~25x above the threshold; this
        # leaves headroom for sliding along walls without false positives.
        # TURN_LEFT/TURN_RIGHT rotate without translating, so only action == 1
        # counts toward forward_steps.
        if action == 1:
            self._forward_steps += 1
            if delta < 0.01:
                self._collisions += 1
        self._prev_position = current_position

        raw_dist = metrics.get("distance_to_goal", float("inf"))
        try:
            dist = _validate_goal_distance(raw_dist)
        except ValueError:
            self._log_invalid_goal_distance(raw_dist, phase="step")
            return self._invalid_goal_distance_transition(obs, raw_dist)

        reward = self._compute_reward(dist)
        success = 1.0 if _is_success_distance(dist) else 0.0
        done = success > 0 or self._step_count >= self._cfg.max_episode_steps

        spl = self._length_ratio() if (done and success > 0) else 0.0

        image = self._obs_to_image(obs)

        return {
            "image": image,
            "reward": reward,
            "done": done,
            "is_first": False,
            "success": success,
            "spl": spl,
            **self._episode_end_metrics(dist, done),
        }

    def _invalid_goal_distance_transition(self, obs, raw_dist: object) -> dict:
        image = self._obs_to_image(obs)
        fallback_dist = (
            self._prev_dist if np.isfinite(float(self._prev_dist)) else 0.0
        )
        return {
            "image": image,
            "reward": self._cfg.step_penalty,
            "done": True,
            "is_first": False,
            "success": 0.0,
            "spl": 0.0,
            "invalid_goal_distance": 1.0,
            "invalid_goal_distance_raw": str(raw_dist),
            **self._episode_end_metrics(fallback_dist, True),
        }

    def _log_invalid_goal_distance(self, raw_dist: object, *, phase: str) -> None:
        print(
            "[HabitatObjectNavEnv] invalid distance_to_goal; "
            f"phase={phase} raw={raw_dist!r} {self._episode_context()}",
            file=sys.stderr,
            flush=True,
        )

    def _episode_context(self) -> str:
        episode = getattr(self._env, "current_episode", None)
        episode_id = getattr(episode, "episode_id", "unknown")
        category = getattr(episode, "object_category", "unknown")
        scene_id = getattr(episode, "scene_id", "unknown")
        scene_name = str(scene_id).split("/")[-1].replace(".basis.glb", "")
        start_position = getattr(episode, "start_position", "unknown")
        return (
            f"episode_id={episode_id} category={category} scene={scene_name} "
            f"scene_id={scene_id} start_position={start_position} "
            f"step={getattr(self, '_step_count', 'unknown')}"
        )

    def _length_ratio(self) -> float:
        """shortest_geodesic / max(shortest_geodesic, path_length). 0 if degenerate."""
        shortest = self._start_geodesic
        if shortest <= 0:
            return 0.0
        return shortest / max(shortest, self._path_length)

    def _episode_end_metrics(self, dist: float, done: bool) -> dict[str, float]:
        """SoftSPL / DTG / collision_rate. Zero mid-episode; only meaningful at done."""
        if not done:
            return {"softspl": 0.0, "dtg": 0.0, "collision_rate": 0.0}
        shortest = self._start_geodesic
        # Progress clipped at 0 — moving away from the goal floors SoftSPL at 0.
        progress = max(0.0, 1.0 - dist / shortest) if shortest > 0 else 0.0
        return {
            "softspl": progress * self._length_ratio(),
            "dtg": dist,
            "collision_rate": self._compute_collision_rate(),
        }

    def _compute_collision_rate(self) -> float:
        if self._forward_steps <= 0:
            return 0.0
        return self._collisions / self._forward_steps

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
        dist = _validate_goal_distance(dist)
        if self._cfg.reward_type == "sparse":
            bonus = self._cfg.success_bonus if _is_success_distance(dist) else 0.0
            return bonus + self._cfg.step_penalty

        if self._cfg.reward_type == "geodesic_delta":
            reward = self._prev_dist - dist
            self._prev_dist = dist
            if _is_success_distance(dist):
                reward += self._cfg.success_bonus
            return reward + self._cfg.step_penalty

        raise ValueError(f"Unknown reward_type: {self._cfg.reward_type!r}")

    def close(self):
        self._env.close()


def build_habitat_env(
    obs_shape: tuple[int, int, int],
    *,
    max_episode_steps: int = 500,
    split: str = "train",
    curriculum_path: str | None = None,
    curriculum_mode: str = "train",
    semantic: bool = False,
    seed: int | None = None,
    reward_type: str = "geodesic_delta",
    max_geodesic: float | None = None,
) -> "HabitatObjectNavEnv":
    """Build a ``HabitatObjectNavEnv`` with a ``DreamerConfig`` derived from ``obs_shape``.

    Consolidates the boilerplate ``DreamerConfig(...) + HabitatObjectNavEnv(...)``
    pair that several callers duplicate.
    """
    config = DreamerConfig(
        obs_shape=obs_shape,
        max_episode_steps=max_episode_steps,
        split=split,
        reward_type=reward_type,
    )
    return HabitatObjectNavEnv(
        config,
        semantic=semantic,
        curriculum_path=curriculum_path,
        curriculum_mode=curriculum_mode,
        seed=seed,
        max_geodesic=max_geodesic,
    )
