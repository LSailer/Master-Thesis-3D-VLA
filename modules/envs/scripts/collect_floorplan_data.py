"""Collect spatial data for semantic floor plan visualization.

Loads the fK2vEV32Lag scene via Habitat, samples the navmesh, renders
a top-down semantic map, and extracts goal positions + episode start
positions for all L2 curriculum episodes.
Saves to output/methods/scenes/floorplan_fK2vEV32Lag.pkl.

Usage (GPU required for habitat-sim):
    srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00 --mem=32G \
        PYTHONUNBUFFERED=1 uv run python modules/envs/scripts/collect_floorplan_data.py
"""

import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import habitat
import habitat_sim
from habitat.config import read_write
from omegaconf import OmegaConf

from modules.envs.habitat import DATA_DIR, SCENE_DIR, sample_navmesh

OUTPUT_PATH = "output/methods/scenes/floorplan_fK2vEV32Lag.pkl"
CURRICULUM_PATH = "data/curriculum/level2_1house_6goals.json"


def render_topdown_semantic(sim, bounds, resolution=0.05, height_offset=0.1):
    """Render a top-down semantic map by querying the semantic scene.

    Instead of using the camera, directly queries the semantic mesh
    by raycasting or sampling semantic IDs at grid positions.
    Returns semantic_grid (object IDs) and id_to_category mapping.
    """
    semantic_scene = sim.semantic_scene
    if semantic_scene is None or len(semantic_scene.objects) == 0:
        print("  No semantic scene loaded — skipping semantic map")
        return None, {}

    # Build mappings: string obj.id -> sequential int, int -> category
    id_to_cat = {}   # int_id -> category_name
    str_to_int = {}  # string obj.id -> int_id
    next_id = 1      # 0 = background/empty

    for obj in semantic_scene.objects:
        if obj is not None and obj.category is not None:
            int_id = next_id
            next_id += 1
            str_to_int[obj.id] = int_id
            id_to_cat[int_id] = obj.category.name()

    print(f"  Semantic scene: {len(id_to_cat)} objects")
    for int_id, cat in sorted(id_to_cat.items()):
        print(f"    id={int_id}: {cat}")

    x_min, z_min = bounds[0][0], bounds[0][2]
    x_max, z_max = bounds[1][0], bounds[1][2]

    xs = np.arange(x_min, x_max, resolution)
    zs = np.arange(z_min, z_max, resolution)
    semantic_grid = np.zeros((len(zs), len(xs)), dtype=np.int32)

    # Project each object's AABB onto the 2D grid
    for obj in semantic_scene.objects:
        if obj is None or obj.aabb is None or obj.id not in str_to_int:
            continue
        aabb = obj.aabb
        obj_x_min = float(aabb.min[0])
        obj_x_max = float(aabb.max[0])
        obj_z_min = float(aabb.min[2])
        obj_z_max = float(aabb.max[2])

        xi_min = max(0, int((obj_x_min - x_min) / resolution))
        xi_max = min(len(xs), int((obj_x_max - x_min) / resolution) + 1)
        zi_min = max(0, int((obj_z_min - z_min) / resolution))
        zi_max = min(len(zs), int((obj_z_max - z_min) / resolution) + 1)

        semantic_grid[zi_min:zi_max, xi_min:xi_max] = str_to_int[obj.id]

    return semantic_grid, id_to_cat


def main():
    import json

    with open(CURRICULUM_PATH) as f:
        curriculum = json.load(f)

    config = habitat.get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    with read_write(config):
        config.habitat.dataset.split = "train"
        config.habitat.dataset.data_path = str(
            DATA_DIR / "{split}" / "{split}.json.gz"
        )
        config.habitat.dataset.scenes_dir = "data/scene_datasets"
        config.habitat.dataset.content_scenes = curriculum["scenes"]
        scene_cfg = next(SCENE_DIR.rglob("*scene_dataset_config.json"), None)
        if scene_cfg:
            config.habitat.simulator.scene_dataset = str(scene_cfg)
        agent_cfg = config.habitat.simulator.agents.main_agent
        agent_cfg.sim_sensors.rgb_sensor.height = 64
        agent_cfg.sim_sensors.rgb_sensor.width = 64
        config.habitat.environment.max_episode_steps = 10
        # Load semantic mesh for floor plan rendering
        OmegaConf.set_struct(config.habitat.simulator, False)
        config.habitat.simulator.load_semantic_mesh = True
        OmegaConf.set_struct(config.habitat.simulator, True)

    env = habitat.Env(config=config)

    # If existing pkl has episode data, only add semantic layer
    existing = None
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "rb") as f:
            existing = pickle.load(f)
        if "starts_by_cat" in existing and "semantic_grid" not in existing:
            print(f"Existing pkl found with episode data — adding semantic only")

    # One reset to initialize the simulator (loads the scene mesh)
    env.reset()

    if existing and "starts_by_cat" in existing:
        # Reuse episode data, just add semantic grid
        navmesh = existing["navmesh"]
        print(f"  Reusing navmesh: {navmesh['grid'].shape}")
    else:
        # Full collection (first run)
        # Filter to L2 train episodes
        train_keys = {
            (k[0], k[1], k[2]) for k in curriculum["train_episode_keys"]
        }
        env._dataset.episodes = [
            ep for ep in env._dataset.episodes
            if (ep.episode_id, ep.object_category,
                ep.scene_id.split("/")[-1].replace(".basis.glb", ""))
            in train_keys
        ]
        n_episodes = len(env._dataset.episodes)
        print(f"L2 train episodes: {n_episodes}")

        print("Sampling navmesh...")
        navmesh = sample_navmesh(env, resolution=0.05)
        print(f"  Grid shape: {navmesh['grid'].shape}")

    # Render top-down semantic map
    print("Rendering semantic map...")
    bounds = env.sim.pathfinder.get_bounds()
    semantic_grid, id_to_cat = render_topdown_semantic(
        env.sim, bounds, resolution=0.05
    )

    if existing and "starts_by_cat" in existing:
        # Merge semantic data into existing result
        existing["semantic_grid"] = semantic_grid
        existing["id_to_cat"] = id_to_cat
        result = existing
    else:
        # Full episode collection
        starts_by_cat = defaultdict(list)
        goals_by_cat = defaultdict(list)
        viewpoints_by_cat = defaultdict(list)
        geodesics_by_cat = defaultdict(list)
        seen_goals = set()

        for i, ep in enumerate(env._dataset.episodes):
            cat = ep.object_category
            start = np.array(ep.start_position)
            starts_by_cat[cat].append(start.tolist())

            for goal in ep.goals:
                pos = tuple(np.round(goal.position, 4))
                key = (cat, pos)
                if key not in seen_goals:
                    seen_goals.add(key)
                    goals_by_cat[cat].append(list(goal.position))
                    if hasattr(goal, "view_points") and goal.view_points:
                        for vp in goal.view_points:
                            viewpoints_by_cat[cat].append(
                                list(vp.agent_state.position)
                            )

            best_geo = float("inf")
            for goal in ep.goals:
                if hasattr(goal, "view_points") and goal.view_points:
                    for vp in goal.view_points:
                        d = env.sim.geodesic_distance(
                            start, np.array(vp.agent_state.position)
                        )
                        if d < best_geo:
                            best_geo = d
            if best_geo < float("inf"):
                geodesics_by_cat[cat].append(best_geo)

            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{n_episodes}] episodes processed")

        print("\n=== Summary ===")
        for cat in sorted(goals_by_cat):
            print(
                f"  {cat:<12} "
                f"starts={len(starts_by_cat[cat]):>4}  "
                f"goals={len(goals_by_cat[cat])}  "
                f"viewpoints={len(viewpoints_by_cat[cat])}  "
                f"mean_geo={np.mean(geodesics_by_cat[cat]):.2f}m"
            )

        result = {
            "scene": "fK2vEV32Lag",
            "navmesh": navmesh,
            "semantic_grid": semantic_grid,
            "id_to_cat": id_to_cat,
            "starts_by_cat": dict(starts_by_cat),
            "goals_by_cat": dict(goals_by_cat),
            "viewpoints_by_cat": dict(viewpoints_by_cat),
            "geodesics_by_cat": dict(geodesics_by_cat),
            "n_episodes": n_episodes,
        }

    env.close()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
