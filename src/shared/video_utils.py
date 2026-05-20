"""Small helpers for Habitat episode videos."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from PIL import Image


MAX_VIDEO_FRAMES = 300
MAX_PANEL_SIZE = 256


def resize_chw_uint8(image: np.ndarray, size: int) -> np.ndarray:
    """Bilinear-resize a CHW RGB uint8 image to ``(3, size, size)``.

    Accepts CHW or HWC input via ``_as_hwc_uint8`` and returns CHW uint8.
    """
    hwc = _as_hwc_uint8(image)
    resized = Image.fromarray(hwc).resize((size, size), Image.Resampling.BILINEAR)
    return np.transpose(np.asarray(resized, dtype=np.uint8), (2, 0, 1))


def _as_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D frame, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"expected RGB-like frame, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_panel(frame: np.ndarray, target_size: int = MAX_PANEL_SIZE) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target_size / max(h, w, 1)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    if size != (w, h):
        resample = Image.NEAREST if scale > 1 else Image.BILINEAR
        frame = np.asarray(Image.fromarray(frame).resize(size, resample))
    return frame


def compose_frame(rgb: np.ndarray, topdown: np.ndarray) -> np.ndarray:
    """Return a side-by-side RGB frame with both panels fit to 256px."""
    left = _resize_panel(_as_hwc_uint8(rgb))
    right = _resize_panel(_as_hwc_uint8(topdown))
    height = max(left.shape[0], right.shape[0])

    def pad(panel: np.ndarray) -> np.ndarray:
        if panel.shape[0] == height:
            return panel
        pad_top = (height - panel.shape[0]) // 2
        pad_bottom = height - panel.shape[0] - pad_top
        return np.pad(panel, ((pad_top, pad_bottom), (0, 0), (0, 0)), constant_values=255)

    return np.concatenate([pad(left), pad(right)], axis=1).astype(np.uint8)


def render_topdown_frame(
    env: Any,
    trajectory_so_far: Sequence[Sequence[float]],
    goal_positions: Sequence[Sequence[float]],
) -> np.ndarray:
    """Render a top-down Habitat map frame as ``(H, W, 3) uint8``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if hasattr(env, "sample_navmesh"):
        nav = env.sample_navmesh(resolution=0.1)
    else:
        from src.environments.habitat import sample_navmesh
        nav = sample_navmesh(env._env, resolution=0.1)
    fig, ax = plt.subplots(1, 1, figsize=(2.56, 2.56), dpi=100)

    extent = [nav["x_min"], nav["x_max"], nav["z_max"], nav["z_min"]]
    ax.imshow(nav["grid"], extent=extent, cmap="Greys_r", alpha=0.3)

    traj = np.asarray(trajectory_so_far, dtype=np.float32)
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 2], "b-", linewidth=1.5, alpha=0.75)
        ax.plot(traj[0, 0], traj[0, 2], "go", markersize=5)
        ax.plot(traj[-1, 0], traj[-1, 2], "rs", markersize=5)

    for goal in goal_positions:
        gp = np.asarray(goal)
        ax.plot(gp[0], gp[2], "m*", markersize=8)

    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame.astype(np.uint8)


def log_episode_video(wandb_module: Any, key: str, frames: list[np.ndarray],
                      step: int, fps: int = 10) -> Any | None:
    """Log frames to W&B as an MP4 video, capped to ``MAX_VIDEO_FRAMES``."""
    if wandb_module is None or not frames:
        return None
    stride = max(1, math.ceil(len(frames) / MAX_VIDEO_FRAMES))
    frames = frames[::stride][:MAX_VIDEO_FRAMES]
    video = np.stack([_as_hwc_uint8(frame) for frame in frames], axis=0)
    video = np.transpose(video, (0, 3, 1, 2))
    wandb_video = wandb_module.Video(video, fps=fps, format="mp4")
    wandb_module.log({key: wandb_video}, step=step)
    return wandb_video
