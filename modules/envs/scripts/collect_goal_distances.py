"""Collect goal distance data for shortest-path episodes.

For each episode, runs the ShortestPathFollower and records both
viewpoint-based and object-centric distances at every step.
Saves results to output/goal_distance_analysis.pkl for notebook analysis.

Usage:
    srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 --mem=32G \
        uv run python modules/envs/scripts/collect_goal_distances.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import habitat
from habitat.config import read_write
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from modules.envs.habitat import (
    DATA_DIR, SCENE_DIR, find_nearest_viewpoint, sample_navmesh,
)

OUTPUT_PATH = "output/goal_distance_analysis.pkl"


def dist_to_nearest_object(env, agent_pos, horizontal_only=False):
    """Euclidean distance to nearest goal object centroid.

    If horizontal_only=True, uses only XZ components (ignores height).
    """
    best = float("inf")
    for goal in env.current_episode.goals:
        gp = np.array(goal.position)
        if horizontal_only:
            d = np.linalg.norm(gp[[0, 2]] - agent_pos[[0, 2]])
        else:
            d = np.linalg.norm(gp - agent_pos)
        if d < best:
            best = d
    return best


def find_nearest_object_centroid(env):
    """Find nearest object centroid (projected to agent height for navigation)."""
    agent_pos = np.array(env.sim.get_agent_state().position)
    best_dist = float("inf")
    best_pos = None
    best_idx = 0
    for gi, goal in enumerate(env.current_episode.goals):
        gp = np.array(goal.position)
        # Project to agent's height so follower can navigate there
        nav_target = np.array([gp[0], agent_pos[1], gp[2]])
        d = np.linalg.norm(gp[[0, 2]] - agent_pos[[0, 2]])
        if d < best_dist:
            best_dist = d
            best_pos = nav_target
            best_idx = gi
    return best_pos, best_idx


def run_episode(env, follower, max_steps=500, target_mode="viewpoint"):
    """Run shortest path follower, recording distances at each step."""
    obs = env.reset()
    metrics = env.get_metrics()

    agent_pos = np.array(env.sim.get_agent_state().position)
    if target_mode == "object_centroid":
        goal_pos, target_goal_idx = find_nearest_object_centroid(env)
    else:
        goal_pos, target_goal_idx = find_nearest_viewpoint(env)

    # Save spatial info for top-down map visualization
    goal_positions = [
        np.array(g.position).tolist() for g in env.current_episode.goals
    ]

    # Sample navmesh at agent's floor level for top-down visualization
    navmesh = sample_navmesh(env)

    ep_info = {
        "episode_id": env.current_episode.episode_id,
        "scene_id": env.current_episode.scene_id,
        "category": env.current_episode.object_category,
        "n_goals": len(env.current_episode.goals),
        "start_geodesic": metrics.get("distance_to_goal", 0.0),
        "start_pos": agent_pos.tolist(),
        "target_viewpoint": list(goal_pos) if goal_pos is not None else None,
        "target_goal_idx": target_goal_idx,
        "goal_positions": goal_positions,
        "navmesh": navmesh,
    }

    steps = []
    path_length = 0.0
    prev_pos = agent_pos.copy()

    for step_i in range(max_steps):
        action = follower.get_next_action(goal_pos)
        if action is None:
            break
        action = int(action)
        if action == 0:  # STOP
            break

        obs = env.step(action)
        metrics = env.get_metrics()
        agent_pos = np.array(env.sim.get_agent_state().position)
        path_length += np.linalg.norm(agent_pos - prev_pos)
        prev_pos = agent_pos.copy()

        steps.append({
            "step": step_i,
            "dist_viewpoint": metrics.get("distance_to_goal", float("inf")),
            "dist_object_3d": dist_to_nearest_object(env, agent_pos),
            "dist_object_2d": dist_to_nearest_object(env, agent_pos, horizontal_only=True),
            "path_length": path_length,
            "agent_pos": agent_pos.tolist(),
        })

    # Final state
    final = steps[-1] if steps else {
        "dist_viewpoint": metrics.get("distance_to_goal", float("inf")),
        "dist_object_3d": dist_to_nearest_object(env, agent_pos),
        "dist_object_2d": dist_to_nearest_object(env, agent_pos, horizontal_only=True),
        "path_length": 0.0,
    }

    ep_info["n_steps"] = len(steps)
    ep_info["path_length"] = path_length
    ep_info["final_dist_viewpoint"] = final["dist_viewpoint"]
    ep_info["final_dist_object_3d"] = final["dist_object_3d"]
    ep_info["final_dist_object_2d"] = final["dist_object_2d"]
    ep_info["steps"] = steps

    return ep_info


def main():
    config = habitat.get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    with read_write(config):
        config.habitat.dataset.split = "val_mini"
        config.habitat.dataset.data_path = str(
            DATA_DIR / "{split}" / "{split}.json.gz"
        )
        config.habitat.dataset.scenes_dir = "data/scene_datasets"
        scene_cfg = next(SCENE_DIR.rglob("*scene_dataset_config.json"), None)
        if scene_cfg:
            config.habitat.simulator.scene_dataset = str(scene_cfg)
        agent_cfg = config.habitat.simulator.agents.main_agent
        agent_cfg.sim_sensors.rgb_sensor.height = 64
        agent_cfg.sim_sensors.rgb_sensor.width = 64
        config.habitat.environment.max_episode_steps = 500

    env = habitat.Env(config=config)
    follower = ShortestPathFollower(
        env.sim, goal_radius=0.01, return_one_hot=False, stop_on_error=True
    )

    n_episodes = len(env._dataset.episodes)

    all_results = {}
    for mode in ["viewpoint", "object_centroid"]:
        print(f"\n=== Mode: {mode} ===")
        print(f"Running {n_episodes} val_mini episodes...")

        results = []
        for i in range(n_episodes):
            ep = run_episode(env, follower, target_mode=mode)
            ep["target_mode"] = mode
            results.append(ep)
            print(
                f"  [{i+1}/{n_episodes}] ep={ep['episode_id']} "
                f"cat={ep['category']:<12} steps={ep['n_steps']:>3} "
                f"vp={ep['final_dist_viewpoint']:.3f}m "
                f"obj3d={ep['final_dist_object_3d']:.3f}m "
                f"obj2d={ep['final_dist_object_2d']:.3f}m"
            )
        all_results[mode] = results

    env.close()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results for both modes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
