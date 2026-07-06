"""Validate that HabitatObjectNavEnv computes SPL correctly.

Runs ShortestPathFollower through the wrapper and checks that
successful episodes produce non-zero, plausible SPL values.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv


@pytest.mark.gpu
@pytest.mark.habitat_sim
@pytest.mark.integration
def test_spl_with_shortest_path():
    config = HabitatEnvConfig(
        obs_shape=(3, 64, 64),
        max_episode_steps=500,
        split="val_mini",
        reward_type="geodesic_delta",
    )
    env = HabitatObjectNavEnv(config)

    # Use tight goal_radius so follower navigates close enough for
    # the wrapper's GOAL_RADIUS (0.2m) to trigger on a movement step
    follower = ShortestPathFollower(
        env._env.sim, goal_radius=0.01, return_one_hot=False
    )

    results = []
    for ep_idx in range(10):
        obs = env.reset()
        start_geodesic = env._start_geodesic
        goal_pos, _ = env.find_nearest_viewpoint()
        assert goal_pos is not None
        category = getattr(env.current_episode, "object_category", "unknown")
        n_goals = len(env.current_episode.goals)

        print(f"\nEpisode {ep_idx}: target={category} ({n_goals} instances), "
              f"geodesic={start_geodesic:.2f}m")

        done = False
        step_count = 0
        while not done and step_count < 500:
            action = follower.get_next_action(goal_pos)
            if action is None:
                print(f"  Step {step_count}: follower returned None, breaking")
                break
            action = int(action)
            if action == 0:
                # Follower says STOP — try MOVE_FORWARD to cross the 0.2m
                # viewpoint threshold so the wrapper detects success
                obs = env.step(1)
                step_count += 1
                done = obs["done"]
                break
            obs = env.step(action)
            step_count += 1
            done = obs["done"]

        result = {
            "episode": ep_idx,
            "category": category,
            "geodesic": start_geodesic,
            "steps": step_count,
            "success": obs["success"],
            "spl": obs["spl"],
            "path_length": env._path_length,
        }
        results.append(result)
        print(
            f"  Result: steps={step_count} path={env._path_length:.2f}m "
            f"success={obs['success']:.0f} spl={obs['spl']:.3f}"
        )

    env.close()

    # Assertions
    successes = [r for r in results if r["success"] > 0]
    print(f"\n{'='*60}")
    print(f"Summary: {len(successes)}/{len(results)} episodes succeeded")

    assert len(successes) > 0, "No successful episodes — can't validate SPL"

    for r in successes:
        assert r["spl"] > 0, f"Episode {r['episode']}: success but SPL=0"
        assert r["spl"] <= 1.0, f"Episode {r['episode']}: SPL > 1.0"
        print(f"  Episode {r['episode']}: spl={r['spl']:.3f} "
              f"(path={r['path_length']:.2f}m, geodesic={r['geodesic']:.2f}m)")

    for r in results:
        if r["success"] == 0:
            assert r["spl"] == 0.0, f"Episode {r['episode']}: failed but SPL!=0"

    print(f"\nPASSED: {len(successes)} episodes with valid non-zero SPL")


if __name__ == "__main__":
    test_spl_with_shortest_path()
