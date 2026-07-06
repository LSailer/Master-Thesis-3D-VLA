"""Artifact + W&B side-effects for the evaluate() loop.

Video composition, top-down PNG writing, per-episode W&B video logging, the
eval-results JSON dump, and the W&B init/teardown lifecycle live here, kept
apart from the rollout logic in ``eval_loop.py`` and the composition/CLI logic
in ``eval_cli.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image

from src.shared.video_utils import (
    compose_frame,
    log_episode_video,
    render_topdown_frame,
)


def init_eval_wandb(args: Any):
    """Init a W&B run for eval video logging, or return None when disabled.

    Args:
      args: Parsed argparse namespace (needs ``wandb_project``, ``wandb_name``,
        ``log_video_episodes``).

    Returns:
      The imported ``wandb`` module with an active run, or ``None`` when no
      project is set or no video episodes are requested.
    """
    if args.wandb_project is None or args.log_video_episodes <= 0:
        return None
    import wandb

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    return wandb


def obs_value(obs: Any, name: str):
    """Read a field from an observation that may be a dict or an object.

    Args:
      obs: An observation dict or an object with the named attribute.
      name: The field/attribute name.

    Returns:
      The field value.
    """
    return obs[name] if isinstance(obs, dict) else getattr(obs, name)


def initial_video_frames(
    env_instance: Any,
    obs: Any,
    trajectory: list[list[float]],
    goal_positions: list[list[float]],
    record_video: bool,
) -> list[np.ndarray]:
    """Build the first composed video frame for an episode, if recording.

    Args:
      env_instance: The eval environment (for top-down rendering).
      obs: The reset observation.
      trajectory: Agent trajectory so far (start position only).
      goal_positions: Goal view-point positions.
      record_video: Whether video is being recorded for this episode.

    Returns:
      A single-element frame list when recording, else an empty list.
    """
    if not record_video:
        return []
    topdown = render_topdown_frame(env_instance, trajectory, goal_positions)
    return [compose_frame(obs_value(obs, "image"), topdown)]


def append_video_frame(
    video_frames: list[np.ndarray],
    env_instance: Any,
    next_obs: Any,
    trajectory: list[list[float]],
    goal_positions: list[list[float]],
) -> None:
    """Append one composed (RGB + top-down) frame to the running video.

    Args:
      video_frames: The frame list being accumulated (mutated in place).
      env_instance: The eval environment (for top-down rendering).
      next_obs: The observation after the step.
      trajectory: Agent trajectory including the latest position.
      goal_positions: Goal view-point positions.
    """
    topdown = render_topdown_frame(env_instance, trajectory, goal_positions)
    video_frames.append(compose_frame(obs_value(next_obs, "image"), topdown))


def write_episode_artifacts(
    *,
    args: Any,
    env_instance: Any,
    output_dir: str,
    ep_idx: int,
    trajectory: list[list[float]],
    goal_positions: list[list[float]],
    record_video: bool,
    wandb_module,
    video_frames: list[np.ndarray],
) -> None:
    """Write the top-down PNG and log the W&B episode video for one episode.

    Args:
      args: Parsed argparse namespace (needs ``render_topdown``).
      env_instance: The eval environment (for top-down rendering).
      output_dir: Directory to write the top-down PNG under.
      ep_idx: Episode index (used in filenames and W&B keys).
      trajectory: Agent trajectory for the episode.
      goal_positions: Goal view-point positions.
      record_video: Whether a video was recorded for this episode.
      wandb_module: The active W&B module, or ``None``.
      video_frames: The composed video frames for this episode.
    """
    if args.render_topdown:
        topdown_dir = os.path.join(output_dir, "topdown")
        os.makedirs(topdown_dir, exist_ok=True)
        topdown_path = os.path.join(topdown_dir, f"episode_{ep_idx:03d}.png")
        Image.fromarray(
            render_topdown_frame(env_instance, trajectory, goal_positions)
        ).save(topdown_path)
    if record_video:
        log_episode_video(
            wandb_module, f"eval/episode_video_{ep_idx}", video_frames, ep_idx
        )


def write_eval_results(output_path: str, meta: dict, results: list[dict]) -> dict:
    """Serialize eval results to JSON and return the written payload.

    Args:
      output_path: Path to write the ``eval_results.json`` file.
      meta: Run metadata dict.
      results: Per-episode result dicts.

    Returns:
      The ``{"meta": ..., "results": ...}`` payload that was written.
    """
    output = {"meta": meta, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")
    return output


def print_eval_summary(results: list[dict], episodes: int) -> None:
    """Print the aggregate success/SPL/reward/steps summary.

    Args:
      results: Per-episode result dicts.
      episodes: Number of episodes evaluated.
    """
    print(f"\n--- Summary ({episodes} episodes) ---")
    print(f"Success: {np.mean([r['success'] for r in results]) * 100:.1f}%")
    print(f"SPL: {np.mean([r['spl'] for r in results]):.3f}")
    print(f"Mean reward: {np.mean([r['reward'] for r in results]):.2f}")
    print(f"Mean steps: {np.mean([r['steps'] for r in results]):.0f}")
