"""Shared point-map helpers for adapters that read the VGGT point head."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def squeeze_frame_axis(points: jnp.ndarray) -> jnp.ndarray:
    """Drop the streaming extractor's leading singleton frame axis, if present.

    Args:
        points: ``(H, W, C)`` or ``(1, H, W, C)`` point map.

    Returns:
        The map without a frame axis.
    """
    points = jnp.asarray(points)
    if points.ndim == 4 and points.shape[0] == 1:
        return points[0]
    return points


def pool_point_map(world_points: jnp.ndarray, side: int) -> jnp.ndarray:
    """Downsample an HWC point map to ``(side, side, 3)``.

    An exact integer factor (the production case, 518 -> 37) is a box mean,
    which keeps the coordinates unbiased; other ratios fall back to an
    antialiased linear resize.

    Args:
        world_points: ``(H, W, 3)`` or ``(1, H, W, 3)`` scale-free world-point
            map (first-frame reference, not metric).
        side: Target side length.

    Returns:
        The pooled ``(side, side, 3)`` map.

    Raises:
        ValueError: If the input is not a square HWC point map.
    """
    points = squeeze_frame_axis(world_points)
    if points.ndim != 3 or points.shape[0] != points.shape[1] or points.shape[2] != 3:
        raise ValueError(f"expected a square (H, H, 3) point map, got {points.shape}")
    source = points.shape[0]
    if source == side:
        return points
    if source % side == 0:
        factor = source // side
        return points.reshape(side, factor, side, factor, 3).mean(axis=(1, 3))
    return jax.image.resize(
        points[None], (1, side, side, 3), method="linear", antialias=True
    )[0]
