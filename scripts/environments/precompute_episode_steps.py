"""Precompute shortest-path step counts for all HM3D ObjectNav episodes.

Runs ShortestPathFollower on every episode in train + val splits,
saves {split: {episode_id: steps}} to JSON for episode filtering.

Uses habitat_sim directly with no sensors (no rendering) for speed.
Episodes are grouped by scene to minimize scene reloads.
"""

import argparse
import glob
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

SCENE_DIR = "data/scene_datasets/hm3d"
DATA_DIR = "data/datasets/objectnav/hm3d/objectnav_hm3d_v2"
GOAL_RADIUS = 0.01  # tight radius for follower (matches test_spl.py)


def load_episodes_by_scene(split):
    """Load all episodes grouped by scene from per-scene content files."""
    content_dir = os.path.join(DATA_DIR, split, "content")
    scenes = {}  # scene_id -> (episodes, goals_by_category)

    for gz_path in sorted(glob.glob(os.path.join(content_dir, "*.json.gz"))):
        with gzip.open(gz_path, "rt") as f:
            data = json.load(f)
        episodes = data.get("episodes", [])
        goals_by_cat = data.get("goals_by_category", {})
        if not episodes:
            continue
        scene_id = episodes[0]["scene_id"]
        scenes[scene_id] = (episodes, goals_by_cat)

    return scenes


def find_nearest_viewpoint(sim, goals):
    """Find nearest goal viewpoint from current agent position."""
    agent_pos = sim.get_agent(0).get_state().position
    best_dist = float("inf")
    best_pos = None
    for goal in goals:
        for vp in goal.get("view_points", []):
            vp_pos = np.array(vp["agent_state"]["position"])
            d = sim.pathfinder.geodesic_distance(agent_pos, vp_pos)
            if d < best_dist:
                best_dist = d
                best_pos = vp_pos
    return best_pos


def count_steps(sim, follower, goal_pos, max_steps=500):
    """Count ShortestPathFollower steps from current agent state to goal."""
    if goal_pos is None:
        return max_steps

    steps = 0
    while steps < max_steps:
        action = follower.get_next_action(goal_pos)
        if action is None:
            break
        action = int(action)
        if action == 0:  # STOP
            steps += 1
            break
        sim.step(action)
        steps += 1

    return steps


def precompute_split(split, max_steps=500):
    """Compute step counts for all episodes in a split."""
    import habitat_sim
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    scenes = load_episodes_by_scene(split)
    n_scenes = len(scenes)
    n_total = sum(len(eps) for eps, _ in scenes.values())
    print(f"[{split}] {n_total} episodes across {n_scenes} scenes", flush=True)

    # Find scene dataset config
    scene_dataset_cfg = None
    for root, _, files in os.walk(SCENE_DIR):
        for f in files:
            if f.endswith("scene_dataset_config.json"):
                scene_dataset_cfg = os.path.join(root, f)
                break
        if scene_dataset_cfg:
            break

    results = {}
    t0 = time.time()
    ep_count = 0

    for scene_idx, (scene_id, (episodes, goals_by_cat)) in enumerate(scenes.items()):
        # Create sim with NO sensors (no rendering)
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_id
        if scene_dataset_cfg:
            backend_cfg.scene_dataset_config_file = scene_dataset_cfg

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = []  # No sensors = no rendering

        try:
            sim = habitat_sim.Simulator(
                habitat_sim.Configuration(backend_cfg, [agent_cfg])
            )
        except Exception as e:
            print(f"  WARN: failed to load {scene_id}: {e}", flush=True)
            # Mark all episodes in this scene as max_steps
            for ep in episodes:
                results[ep["episode_id"]] = max_steps
                ep_count += 1
            continue

        follower = ShortestPathFollower(sim, goal_radius=GOAL_RADIUS,
                                        return_one_hot=False)

        for ep in episodes:
            # Set agent to episode start
            agent = sim.get_agent(0)
            state = agent.get_state()
            state.position = np.array(ep["start_position"])
            rot = ep["start_rotation"]  # [x, y, z, w] in dataset
            state.rotation = np.quaternion(rot[3], rot[0], rot[1], rot[2])
            agent.set_state(state)

            # Resolve goals from goals_by_category
            scene_name = scene_id.split("/")[-1]  # e.g. "6imZUJGRUq4.basis.glb"
            goals_key = f"{scene_name}_{ep['object_category']}"
            goals = goals_by_cat.get(goals_key, [])

            goal_pos = find_nearest_viewpoint(sim, goals)
            steps = count_steps(sim, follower, goal_pos, max_steps)
            results[ep["episode_id"]] = steps
            ep_count += 1

        sim.close()

        if (scene_idx + 1) % 20 == 0 or (scene_idx + 1) == n_scenes:
            elapsed = time.time() - t0
            rate = ep_count / elapsed if elapsed > 0 else 0
            print(f"  [{split}] {ep_count}/{n_total} episodes, "
                  f"{scene_idx + 1}/{n_scenes} scenes "
                  f"({rate:.1f} ep/s, {elapsed:.0f}s)", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="data/episode_step_counts.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max steps before marking episode as max",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    data = {}
    for split in ["train", "val"]:
        data[split] = precompute_split(split, args.max_steps)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    for split, results in data.items():
        steps = list(results.values())
        n_total = len(steps)
        n_filtered = sum(1 for s in steps if s >= 200)
        print(f"[{split}] {n_total} episodes, {n_filtered} >= 200 steps "
              f"({n_filtered / n_total * 100:.1f}%)")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
