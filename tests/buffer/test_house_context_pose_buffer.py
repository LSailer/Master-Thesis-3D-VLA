"""Behavior checks for ``HouseContextPoseBuffer.add`` voxel accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import numpy as np

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGTExtractOutput


@dataclass(frozen=True)
class _FakeVGGTOutput:
    """Minimal VGGT stand-in exposing only the fields ``add`` reads.

    A stub keeps the test focused on point/color/confidence alignment.
    """

    world_points: np.ndarray
    confidence: np.ndarray


def _inputs(
    points: list[list[float]],
    colors: list[list[int]],
    confidences: list[float],
) -> tuple[VGGTExtractOutput, ObservationFrame]:
    """Build a ``(vggt_output, observation)`` pair from per-point rows.

    Point ``i``, color ``i`` and confidence ``i`` describe the same pixel, so
    tests can reason about which color survives filtering and voxel dedup.

    Args:
        points: ``(x, y, z)`` world coordinates, one row per pixel.
        colors: ``(r, g, b)`` uint8 pixel colors aligned to ``points``.
        confidences: Per-pixel VGGT confidence scores aligned to ``points``.

    Returns:
        A fake VGGT output plus the observation frame that produced it.
    """
    points_arr = np.asarray(points, dtype=np.float32)
    colors_arr = np.asarray(colors, dtype=np.uint8)
    confidence_arr = np.asarray(confidences, dtype=np.float32)
    n = points_arr.shape[0]

    world_points = points_arr.reshape(1, n, 3)  # (H=1, W=n, C=3)
    confidence = confidence_arr.reshape(1, n)  # (H=1, W=n)
    image = np.transpose(colors_arr.reshape(1, n, 3), (2, 0, 1))  # (C=3, H=1, W=n)

    vggt_output = _FakeVGGTOutput(world_points=world_points, confidence=confidence)
    observation = ObservationFrame(image=image, is_first=True)
    return cast(VGGTExtractOutput, vggt_output), observation


def _stored_colors(buffer: HouseContextPoseBuffer) -> set[tuple[int, int, int]]:
    """Return buffered colors as an order-free set of RGB tuples."""
    colors_rgb = buffer.colors_rgb
    assert colors_rgb is not None
    return {
        (int(row[0]), int(row[1]), int(row[2]))
        for row in np.asarray(colors_rgb)
    }


def test_first_add_returns_bfloat16_points_and_populates_buffer() -> None:
    """A fresh add returns the accumulated ``(P, 3)`` buffer and stores colors."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")
    vggt_output, observation = _inputs(
        points=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        colors=[[10, 20, 30], [40, 50, 60]],
        confidences=[0.9, 0.9],
    )

    result = buffer.add(vggt_output, observation)

    assert result.shape == (2, 3)
    assert result.dtype == jnp.bfloat16
    assert result is buffer.points_xyz
    assert buffer.colors_rgb is not None
    assert buffer.colors_rgb.shape == (2, 3)
    assert buffer.colors_rgb.dtype == np.uint8
    assert _stored_colors(buffer) == {(10, 20, 30), (40, 50, 60)}


def test_add_without_admitted_points_returns_empty_and_leaves_buffer_unset() -> None:
    """When nothing passes admission the buffer stays empty and returns ``(0, 3)``."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")
    vggt_output, observation = _inputs(
        points=[[0.0, 0.0, 0.0]],
        colors=[[1, 2, 3]],
        confidences=[0.1],  # below the confidence threshold
    )

    result = buffer.add(vggt_output, observation)

    assert result.shape == (0, 3)
    assert result.dtype == jnp.bfloat16
    assert buffer.points_xyz is None
    assert buffer.colors_rgb is None


def test_low_confidence_points_are_filtered_out() -> None:
    """Only points at or above ``confidence_score`` are admitted."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")
    vggt_output, observation = _inputs(
        points=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        colors=[[1, 1, 1], [7, 8, 9]],
        confidences=[0.1, 0.9],
    )

    result = buffer.add(vggt_output, observation)

    assert result.shape == (1, 3)
    assert _stored_colors(buffer) == {(7, 8, 9)}


def test_non_finite_points_are_dropped() -> None:
    """Points with a NaN/inf coordinate never reach the buffer."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")
    vggt_output, observation = _inputs(
        points=[[np.nan, 0.0, 0.0], [0.5, 0.5, 0.5]],
        colors=[[1, 1, 1], [7, 8, 9]],
        confidences=[0.9, 0.9],
    )

    result = buffer.add(vggt_output, observation)

    assert result.shape == (1, 3)
    assert _stored_colors(buffer) == {(7, 8, 9)}


def test_points_in_same_voxel_collapse_to_first_representative() -> None:
    """Points within one 1 cm voxel keep only the first occurrence's color."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")
    vggt_output, observation = _inputs(
        # first two rows land in voxel (0, 0, 0); the third is a separate voxel
        points=[[0.001, 0.001, 0.001], [0.002, 0.002, 0.002], [0.5, 0.5, 0.5]],
        colors=[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        confidences=[0.9, 0.9, 0.9],
    )

    result = buffer.add(vggt_output, observation)

    assert result.shape == (2, 3)
    # representative is the first point in the voxel, so (2, 2, 2) is discarded
    assert _stored_colors(buffer) == {(1, 1, 1), (3, 3, 3)}


def test_voxels_seen_in_earlier_add_are_not_re_added() -> None:
    """Voxels already stored by a prior add do not grow the buffer again."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")

    buffer.add(*_inputs([[0.0, 0.0, 0.0]], [[1, 1, 1]], [0.9]))

    # same voxel as the first add -> nothing new is appended
    result = buffer.add(*_inputs([[0.005, 0.005, 0.005]], [[2, 2, 2]], [0.9]))
    assert result.shape == (1, 3)
    assert _stored_colors(buffer) == {(1, 1, 1)}

    # a previously unseen voxel does extend the buffer
    result = buffer.add(*_inputs([[0.5, 0.5, 0.5]], [[3, 3, 3]], [0.9]))
    assert result.shape == (2, 3)
    assert _stored_colors(buffer) == {(1, 1, 1), (3, 3, 3)}
