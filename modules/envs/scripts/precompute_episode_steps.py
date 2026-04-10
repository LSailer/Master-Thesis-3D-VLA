"""Precompute shortest-path step counts for all HM3D ObjectNav episodes.

Runs ShortestPathFollower on every episode in train + val splits,
saves {split: {episode_id: steps}} to JSON for episode filtering.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from modules.dreamerv3.configs import DreamerConfig
from modules.envs.habitat import HabitatObjectNavEnv, find_nearest_viewpoint


def count_steps(env, follower, max_steps=500):
    """Run ShortestPathFollower on current episode, return step count."""
    goal_pos, _ = find_nearest_viewpoint(env._env)
    if goal_pos is None:
        return max_steps

    steps = 0
    while steps < max_steps:
        action = follower.get_next_action(goal_pos)
        if action is None:
            break
        action = int(action)
        if action == 0:  # STOP — follower thinks it's done
            steps += 1
            break
        env._env.step(action=action)
        steps += 1

    return steps


def precompute_split(split, max_steps=500):
    """Compute step counts for all episodes in a split."""
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    config = DreamerConfig(
        obs_shape=(3, 64, 64),
        max_episode_steps=max_steps,
        split=split,
        reward_type="geodesic_delta",
    )
    env = HabitatObjectNavEnv(config)
    follower = ShortestPathFollower(
        env._env.sim, goal_radius=0.01, return_one_hot=False
    )

    n_episodes = len(env._env._dataset.episodes)
    print(f"[{split}] Processing {n_episodes} episodes...")

    results = {}
    t0 = time.time()
    for i in range(n_episodes):
        env._env.reset()
        ep_id = env._env.current_episode.episode_id
        steps = count_steps(env, follower, max_steps)
        results[ep_id] = steps

        if (i + 1) % 100 == 0 or (i + 1) == n_episodes:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{split}] {i + 1}/{n_episodes} "
                  f"({rate:.1f} ep/s, {elapsed:.0f}s elapsed)")

    env.close()
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

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
