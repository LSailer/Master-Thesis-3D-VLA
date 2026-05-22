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

    # Per-category metric streams. Adding a new one is a one-line change here.
    _PER_CAT_KEYS: tuple[str, ...] = ("success", "spl", "reward")

    # Rename rule for output keys (`metrics/sr` / `goal/{cat}/sr` keep the
    # historical "sr" label even though the internal stream is named "success").
    _OUTPUT_RENAME: dict[str, str] = {"success": "sr"}

    def __init__(self, window: int = 100, track_collision_rate: bool = False):
        self._window = window
        # Collision rate is val-only by default: train rollouts already log
        # per-step action statistics, so a per-episode aggregate is more
        # informative in the deterministic val setting.
        self._track_collision_rate = track_collision_rate
        self._global: dict[str, deque[float]] = defaultdict(self._new_window)
        self._per_cat: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(self._new_window)
        )
        self._episode_count = 0

    def _new_window(self) -> deque[float]:
        return deque(maxlen=self._window)

    def record(
        self,
        reward: float,
        success: float,
        spl: float,
        category: str,
        scene_id: str,
        softspl: float = 0.0,
        dtg: float = 0.0,
        collision_rate: float = 0.0,
    ) -> dict[str, Any]:
        """Record a completed episode, return dict of all metrics to log."""
        self._episode_count += 1

        samples: dict[str, float] = {
            "reward": reward,
            "success": success,
            "spl": spl,
            "softspl": softspl,
            "dtg": dtg,
        }
        if self._track_collision_rate:
            samples["collision_rate"] = collision_rate

        for k, v in samples.items():
            self._global[k].append(v)
        for k in self._PER_CAT_KEYS:
            self._per_cat[category][k].append(samples[k])

        scene = scene_id.split("/")[-1].replace(".basis.glb", "")

        metrics: dict[str, Any] = {
            "episode/count": self._episode_count,
            "episode/goal": category,
            "episode/scene": scene,
        }
        for k, v in samples.items():
            metrics[f"episode/{k}"] = v
        for k, dq in self._global.items():
            metrics[f"metrics/{self._OUTPUT_RENAME.get(k, k)}"] = float(np.mean(dq))
        for cat, per in self._per_cat.items():
            for k, dq in per.items():
                metrics[f"goal/{cat}/{self._OUTPUT_RENAME.get(k, k)}"] = float(np.mean(dq))
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
