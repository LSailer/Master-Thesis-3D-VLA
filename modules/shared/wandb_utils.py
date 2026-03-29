"""Wandb logging utilities for 3D-VLA ObjectNav experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import wandb


def init_run(
    project: str = "3d-vla-objectnav",
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
) -> wandb.sdk.wandb_run.Run:
    """Init wandb run with standard project defaults."""
    return wandb.init(project=project, config=config, tags=tags, group=group)


def log_episode(
    step: int,
    reward: float,
    success: bool,
    spl: float,
    observations: dict[str, np.ndarray] | None = None,
    vggt_features: np.ndarray | None = None,
) -> None:
    """Log per-episode metrics + optional obs images and VGGT stats."""
    metrics: dict[str, Any] = {
        "episode/reward": reward,
        "episode/success": int(success),
        "episode/spl": spl,
    }

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
