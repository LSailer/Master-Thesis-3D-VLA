"""Random-action baseline for Habitat ObjectNav.

Runs uniform-random actions on curriculum eval episodes and saves
per-episode CSV + aggregate JSON for comparison against trained agents.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from src.environments.habitat import ACTIONS, HabitatObjectNavEnv, build_habitat_env


@dataclass(frozen=True)
class EpisodeResult:
    """Per-episode metrics; fields match ``episodes.csv`` columns."""

    episode: int
    scene: str
    category: str
    steps: int
    reward: float
    success: float
    spl: float
    stop_pct: float
    forward_pct: float
    left_pct: float
    right_pct: float

    @classmethod
    def csv_header(cls) -> list[str]:
        return [field.name for field in fields(cls)]

    def to_csv_row(self) -> list[str | int]:
        return [
            self.episode,
            self.scene,
            self.category,
            self.steps,
            f"{self.reward:.4f}",
            f"{self.success:.0f}",
            f"{self.spl:.4f}",
            f"{self.stop_pct:.1f}",
            f"{self.forward_pct:.1f}",
            f"{self.left_pct:.1f}",
            f"{self.right_pct:.1f}",
        ]

    def action_count(self, action_idx: int) -> float:
        pcts = (self.stop_pct, self.forward_pct, self.left_pct, self.right_pct)
        return self.steps * pcts[action_idx] / 100


def _print_summary(
    summary: dict, csv_path: str, json_path: str, elapsed: float
) -> None:
    """Print the human-readable run summary."""
    print(f"\n--- Summary ({summary['episodes']} episodes, {elapsed:.0f}s) ---")
    print(f"Success Rate: {summary['sr'] * 100:.2f}%")
    print(f"SPL:          {summary['spl']:.4f}")
    print(f"Mean Reward:  {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Mean Steps:   {summary['mean_steps']:.0f} ± {summary['std_steps']:.0f}")
    print(f"Actions:      {summary['action_distribution']}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


def _run_episode(
    env: HabitatObjectNavEnv,
    rng,
    num_actions: int,
    max_episode_steps: int,
    ep_idx: int,
) -> EpisodeResult:
    """Roll one uniform-random episode and return its summary metrics."""
    obs = env.reset()
    episode = env._env.current_episode
    scene = episode.scene_id.split("/")[-1].replace(".basis.glb", "")
    category = episode.object_category

    action_counts = {a: 0 for a in range(num_actions)}
    total_reward = 0.0
    steps = 0
    for _ in range(max_episode_steps):
        action = int(rng.integers(0, num_actions))
        obs = env.step(action)
        action_counts[action] += 1
        total_reward += obs.reward
        steps += 1
        if obs.done:
            break

    action_pcts = {
        name: action_counts[idx] / steps * 100 for idx, name in ACTIONS.items()
    }
    return EpisodeResult(
        episode=ep_idx,
        scene=scene,
        category=category,
        steps=steps,
        reward=total_reward,
        success=obs.success,
        spl=obs.spl,
        stop_pct=action_pcts["STOP"],
        forward_pct=action_pcts["MOVE_FORWARD"],
        left_pct=action_pcts["TURN_LEFT"],
        right_pct=action_pcts["TURN_RIGHT"],
    )


def _aggregate_results(
    all_results: list[EpisodeResult],
    *,
    seed: int,
    max_episode_steps: int,
    curriculum_path: str,
) -> dict:
    """Reduce per-episode results to aggregate metrics + action distribution."""
    successes = [r.success for r in all_results]
    spls = [r.spl for r in all_results]
    rewards = [r.reward for r in all_results]
    steps_list = [r.steps for r in all_results]
    total_actions = sum(r.steps for r in all_results)
    agg_action_counts = {name: 0 for name in ACTIONS.values()}
    for result in all_results:
        for idx, name in ACTIONS.items():
            agg_action_counts[name] += result.action_count(idx)

    return {
        "episodes": len(all_results),
        "sr": float(np.mean(successes)),
        "spl": float(np.mean(spls)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "action_distribution": {
            name: count / total_actions * 100
            for name, count in agg_action_counts.items()
        },
        "seed": seed,
        "max_episode_steps": max_episode_steps,
        "curriculum_path": curriculum_path,
    }


def run_random_baseline(
    curriculum_path: str,
    output_dir: str,
    max_episode_steps: int = 500,
    seed: int = 42,
) -> dict:
    """Run random agent on all eval episodes from a curriculum.

    Returns aggregate metrics dict.
    """
    rng = np.random.default_rng(seed)
    num_actions = len(ACTIONS)

    env = build_habitat_env(
        (3, 64, 64),
        max_episode_steps=max_episode_steps,
        curriculum_path=curriculum_path,
        curriculum_mode="eval",
    )
    num_episodes = len(env._env._dataset.episodes)
    print(f"Running random baseline on {num_episodes} eval episodes")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "episodes.csv")

    all_results = []
    t_start = time.time()

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(EpisodeResult.csv_header())

        for ep_idx in range(num_episodes):
            result = _run_episode(env, rng, num_actions, max_episode_steps, ep_idx)
            writer.writerow(result.to_csv_row())
            all_results.append(result)

            if (ep_idx + 1) % 50 == 0 or ep_idx == num_episodes - 1:
                elapsed = time.time() - t_start
                sr_so_far = np.mean([r.success for r in all_results]) * 100
                print(
                    f"  [{ep_idx + 1}/{num_episodes}] SR={sr_so_far:.1f}%  "
                    f"elapsed={elapsed:.0f}s"
                )

    env.close()

    summary = _aggregate_results(
        all_results,
        seed=seed,
        max_episode_steps=max_episode_steps,
        curriculum_path=curriculum_path,
    )

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary, csv_path, json_path, time.time() - t_start)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Random-action baseline for Habitat ObjectNav"
    )
    parser.add_argument(
        "--curriculum_path",
        type=str,
        required=True,
        help="Path to curriculum JSON (e.g. data/curriculum/level1_1house_1goal.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save episodes.csv and summary.json",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=500,
        help="Max steps per episode (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()
    run_random_baseline(
        curriculum_path=args.curriculum_path,
        output_dir=args.output_dir,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
