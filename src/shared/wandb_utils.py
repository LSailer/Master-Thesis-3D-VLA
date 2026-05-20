"""Wandb logging utilities for 3D-VLA ObjectNav experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from typing import Any, Iterable

import numpy as np
import wandb


def init_run(
    args: argparse.Namespace,
    metadata: dict[str, Any] | None = None,
    *,
    default_tags: Iterable[str] | None = None,
) -> tuple[Any | None, Any | None]:
    """Init a W&B run from ``argparse`` args + a metadata dict.

    Returns ``(wandb_module, run)``. If ``args.wandb_project`` is ``None``,
    returns ``(None, None)`` so callers can short-circuit without a wrapper.
    Recognised attrs: ``wandb_project``, ``wandb_name``, ``wandb_tags``
    (comma-separated string or iterable), ``wandb_init_timeout``.
    """
    if getattr(args, "wandb_project", None) is None:
        return None, None

    raw_tags = getattr(args, "wandb_tags", None)
    if isinstance(raw_tags, str):
        tags: list[str] | None = [t.strip() for t in raw_tags.split(",") if t.strip()]
    elif raw_tags is None:
        tags = list(default_tags) if default_tags is not None else None
    else:
        tags = list(raw_tags)

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": getattr(args, "wandb_name", None),
        "tags": tags,
        "config": metadata,
    }
    timeout = getattr(args, "wandb_init_timeout", None)
    if timeout is not None:
        init_kwargs["settings"] = wandb.Settings(init_timeout=timeout)
    run = wandb.init(**init_kwargs)
    return wandb, run


class EpisodeTracker:
    """Track per-episode metrics with rolling averages and per-category breakdown."""

    def __init__(self, window: int = 100):
        self._window = window
        self._rewards: deque[float] = deque(maxlen=window)
        self._successes: deque[float] = deque(maxlen=window)
        self._spls: deque[float] = deque(maxlen=window)
        self._cat_successes: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._cat_rewards: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._episode_count = 0

    def record(
        self,
        reward: float,
        success: float,
        spl: float,
        category: str,
        scene_id: str,
    ) -> dict[str, Any]:
        """Record a completed episode, return dict of all metrics to log."""
        self._episode_count += 1
        self._rewards.append(reward)
        self._successes.append(success)
        self._spls.append(spl)
        self._cat_successes[category].append(success)
        self._cat_rewards[category].append(reward)

        scene = scene_id.split("/")[-1].replace(".basis.glb", "")

        metrics: dict[str, Any] = {
            "episode/reward": reward,
            "episode/success": success,
            "episode/spl": spl,
            "episode/count": self._episode_count,
            "episode/goal": category,
            "episode/scene": scene,
            "metrics/sr": float(np.mean(self._successes)),
            "metrics/spl": float(np.mean(self._spls)),
            "metrics/reward": float(np.mean(self._rewards)),
        }
        for cat, succ_deque in self._cat_successes.items():
            metrics[f"goal/{cat}/sr"] = float(np.mean(succ_deque))
        for cat, rew_deque in self._cat_rewards.items():
            metrics[f"goal/{cat}/reward"] = float(np.mean(rew_deque))
        return metrics


def log_episode(
    step: int,
    reward: float,
    success: bool,
    spl: float,
    observations: dict[str, np.ndarray] | None = None,
    vggt_features: np.ndarray | None = None,
    goal: str | None = None,
    scene: str | None = None,
) -> None:
    """Log per-episode metrics + optional obs images and VGGT stats."""
    metrics: dict[str, Any] = {
        "episode/reward": reward,
        "episode/success": int(success),
        "episode/spl": spl,
    }

    if goal is not None:
        metrics["episode/goal"] = goal
    if scene is not None:
        metrics["episode/scene"] = scene

    if vggt_features is not None:
        metrics["episode/vggt_norm"] = float(np.linalg.norm(vggt_features))
        metrics["episode/vggt_mean"] = float(np.mean(vggt_features))

    if observations is not None and step % 100 == 0:
        if "rgb" in observations:
            metrics["episode/rgb"] = wandb.Image(observations["rgb"])
        if "depth" in observations:
            depth = observations["depth"]
            depth_vis = (depth / (depth.max() + 1e-6) * 255).astype(np.uint8)
            metrics["episode/depth"] = wandb.Image(depth_vis)

    wandb.log(metrics, step=step)


def log_vggt_comparison(
    step: int,
    features_2d: np.ndarray,
    features_3d: np.ndarray,
) -> None:
    """Log comparison between 2D and 3D VGGT features."""
    # Cosine similarity
    dot = np.sum(features_2d * features_3d)
    norm_2d = np.linalg.norm(features_2d)
    norm_3d = np.linalg.norm(features_3d)
    cosine_sim = dot / (norm_2d * norm_3d + 1e-8)

    wandb.log(
        {
            "vggt/cosine_sim_2d_3d": float(cosine_sim),
            "vggt/norm_2d": float(norm_2d),
            "vggt/norm_3d": float(norm_3d),
            "vggt/norm_ratio": float(norm_3d / (norm_2d + 1e-8)),
        },
        step=step,
    )
