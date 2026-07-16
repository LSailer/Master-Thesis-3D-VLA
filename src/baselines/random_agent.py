"""Random-action baseline for Habitat ObjectNav.

Runs uniform-random actions on curriculum eval episodes and saves
per-episode CSV + aggregate JSON for comparison against trained agents.
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from src.environments.habitat import (
    ACTIONS,
    HABITAT_CURRICULA,
    HabitatEnvConfig,
    HabitatObjectNavEnv,
)
from src.environments.observation import ObservationFrame


def _curriculum_name(curriculum_path: str) -> str:
    """Map a curriculum JSON path to a named level."""
    from pathlib import Path

    path = Path(curriculum_path)
    for name, known in HABITAT_CURRICULA.items():
        if path == known or path.resolve() == known.resolve():
            return name
    raise ValueError(
        f"Unknown curriculum path {curriculum_path!r}. "
        f"Expected one of: {list(HABITAT_CURRICULA.values())}"
    )


class RandomAgent:
    """Uniform-random discrete-action agent bound to one environment."""

    def __init__(
        self, env: HabitatObjectNavEnv, num_actions: int = 4, seed: int = 42
    ) -> None:
        self.env = env
        self.num_actions = int(num_actions)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def reset(self) -> None:
        """Reset the RNG to the construction seed."""
        self._rng = np.random.default_rng(self._seed)

    def sample_action(self) -> int:
        """Sample one action uniformly from ``[0, num_actions)``."""
        return int(self._rng.integers(0, self.num_actions))

    def act(self) -> ObservationFrame:
        """Sample and apply one action, returning the resulting observation."""
        return self.env.step(self.sample_action())


@dataclass(frozen=True)
class EpisodeMetrics:
    """Scalar per-episode outcome metrics."""

    steps: int
    reward: float
    success: float
    spl: float


@dataclass(frozen=True)
class ActionPercentages:
    """Per-action percentages for one episode."""

    stop: float
    forward: float
    left: float
    right: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return percentages in action-index order."""
        return (self.stop, self.forward, self.left, self.right)

    def count(self, steps: int, action_idx: int) -> float:
        """Estimate the absolute count for one action."""
        return steps * self.as_tuple()[action_idx] / 100


@dataclass(frozen=True)
class EpisodeResult:
    """Per-episode metrics; fields match ``episodes.csv`` columns."""

    episode: int
    scene: str
    category: str
    metrics: EpisodeMetrics
    actions: ActionPercentages

    @classmethod
    def csv_header(cls) -> list[str]:
        """Return CSV column names."""
        return [
            "episode",
            "scene",
            "category",
            "steps",
            "reward",
            "success",
            "spl",
            "stop_pct",
            "forward_pct",
            "left_pct",
            "right_pct",
        ]

    def to_csv_row(self) -> list[str | int]:
        """Return this result formatted for CSV output."""
        return [
            self.episode,
            self.scene,
            self.category,
            self.metrics.steps,
            f"{self.metrics.reward:.4f}",
            f"{self.metrics.success:.0f}",
            f"{self.metrics.spl:.4f}",
            f"{self.actions.stop:.1f}",
            f"{self.actions.forward:.1f}",
            f"{self.actions.left:.1f}",
            f"{self.actions.right:.1f}",
        ]


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
    agent: RandomAgent,
    max_episode_steps: int,
    ep_idx: int,
) -> EpisodeResult:
    """Roll one uniform-random episode and return its summary metrics."""
    obs = agent.env.reset()
    episode = agent.env.current_episode
    scene = episode.scene_id.rsplit("/", maxsplit=1)[-1].replace(".basis.glb", "")
    category = getattr(episode, "object_category", "unknown")

    action_counts = {a: 0.0 for a in range(agent.num_actions)}
    total_reward = 0.0
    steps = 0
    for _ in range(max_episode_steps):
        obs = agent.act()
        action = obs.previous_action
        if action is None:
            raise RuntimeError("random-agent step observation is missing previous_action")
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
        metrics=EpisodeMetrics(
            steps=steps,
            reward=total_reward,
            success=obs.success,
            spl=obs.spl,
        ),
        actions=ActionPercentages(
            stop=action_pcts["STOP"],
            forward=action_pcts["MOVE_FORWARD"],
            left=action_pcts["TURN_LEFT"],
            right=action_pcts["TURN_RIGHT"],
        ),
    )


def _aggregate_results(
    all_results: list[EpisodeResult],
    *,
    seed: int,
    max_episode_steps: int,
    curriculum_path: str,
) -> dict:
    """Reduce per-episode results to aggregate metrics + action distribution."""
    successes = [r.metrics.success for r in all_results]
    spls = [r.metrics.spl for r in all_results]
    rewards = [r.metrics.reward for r in all_results]
    steps_list = [r.metrics.steps for r in all_results]
    total_actions = sum(r.metrics.steps for r in all_results)
    agg_action_counts = {name: 0.0 for name in ACTIONS.values()}
    for result in all_results:
        for idx, name in ACTIONS.items():
            agg_action_counts[name] += result.actions.count(result.metrics.steps, idx)

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


def _write_episode_results(
    *,
    env: HabitatObjectNavEnv,
    agent: RandomAgent,
    max_episode_steps: int,
    csv_path: str,
    start_time: float,
) -> list[EpisodeResult]:
    """Run all eval episodes and write per-episode CSV rows."""
    all_results: list[EpisodeResult] = []
    num_episodes = env.episode_count
    print(f"Running random baseline on {num_episodes} eval episodes")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(EpisodeResult.csv_header())

        for ep_idx in range(num_episodes):
            result = _run_episode(agent, max_episode_steps, ep_idx)
            writer.writerow(result.to_csv_row())
            all_results.append(result)

            if (ep_idx + 1) % 50 == 0 or ep_idx == num_episodes - 1:
                elapsed = time.time() - start_time
                sr_so_far = np.mean([r.metrics.success for r in all_results]) * 100
                print(
                    f"  [{ep_idx + 1}/{num_episodes}] SR={sr_so_far:.1f}%  "
                    f"elapsed={elapsed:.0f}s"
                )
    return all_results


def run_random_baseline(
    curriculum_path: str,
    output_dir: str,
    max_episode_steps: int = 500,
    seed: int = 42,
) -> dict:
    """Run random agent on all eval episodes from a curriculum.

    Returns aggregate metrics dict.
    """
    env = HabitatObjectNavEnv(
        HabitatEnvConfig(
            obs_shape=(3, 64, 64),
            max_episode_steps=max_episode_steps,
            curriculum=_curriculum_name(curriculum_path),
            mode="eval",
        ),
        seed=seed,
    )
    agent = RandomAgent(env=env, num_actions=len(ACTIONS), seed=seed)
    start_time = time.time()
    csv_path = os.path.join(output_dir, "episodes.csv")
    os.makedirs(output_dir, exist_ok=True)
    try:
        all_results = _write_episode_results(
            env=env,
            agent=agent,
            max_episode_steps=max_episode_steps,
            csv_path=csv_path,
            start_time=start_time,
        )
    finally:
        env.close()

    summary = _aggregate_results(
        all_results,
        seed=seed,
        max_episode_steps=max_episode_steps,
        curriculum_path=curriculum_path,
    )

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary, csv_path, json_path, time.time() - start_time)
    return summary


def main():
    """CLI entry point for the random baseline."""
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
