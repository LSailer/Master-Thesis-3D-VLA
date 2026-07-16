
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
    """Return logical buffered colors as an order-free set of RGB tuples."""
    colors_rgb = buffer.colors_rgb[: buffer.point_count]
    return {
        (int(row[0]), int(row[1]), int(row[2]))
        for row in np.asarray(colors_rgb)
    }


def _stored_xyz(buffer: HouseContextPoseBuffer) -> np.ndarray:
    """Return logical buffered XYZ rows as a host float32 array."""
    return np.asarray(buffer.points_xyz[: buffer.point_count], dtype=np.float32)


def test_first_add_returns_bfloat16_points_and_populates_buffer() -> None:
    """A fresh add returns the accumulated ``(P, 3)`` buffer and stores colors."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")
    vggt_output, observation = _inputs(
        points=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        colors=[[10, 20, 30], [40, 50, 60]],
        confidences=[0.9, 0.9],
    )

    result = buffer.add(vggt_output, observation)

    assert result.shape == (buffer.capacity, 3)
    assert result.dtype == jnp.bfloat16
    assert buffer.point_count == 2
    assert buffer.points_xyz.shape == (2, 3)
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

    assert result.shape == (buffer.capacity, 3)
    assert result.dtype == jnp.bfloat16
    assert buffer.point_count == 0


def test_low_confidence_points_are_filtered_out() -> None:
    """Only points at or above ``confidence_score`` are admitted."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")
    vggt_output, observation = _inputs(
        points=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        colors=[[1, 1, 1], [7, 8, 9]],
        confidences=[0.1, 0.9],
    )

    result = buffer.add(vggt_output, observation)

    assert result.shape == (buffer.capacity, 3)
    assert buffer.point_count == 1
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

    assert result.shape == (buffer.capacity, 3)
    assert buffer.point_count == 1
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

    assert result.shape == (buffer.capacity, 3)
    assert buffer.point_count == 2
    # representative is the first point in the voxel, so (2, 2, 2) is discarded
    assert _stored_colors(buffer) == {(1, 1, 1), (3, 3, 3)}


def test_hash_collisions_do_not_drop_distinct_voxels() -> None:
    """Open addressing keeps distinct voxels even when their first slot collides."""
    buffer = HouseContextPoseBuffer(
        confidence_score=0.5,
        scene_id="scene",
        capacity=2,
        hash_table_size=2,
    )
    vggt_output, observation = _inputs(
        points=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.02]],
        colors=[[1, 1, 1], [2, 2, 2]],
        confidences=[0.9, 0.9],
    )

    buffer.add(vggt_output, observation)

    assert buffer.point_count == 2
    assert buffer.failed_insert_count == 0
    assert buffer.overflow_count == 0
    assert _stored_colors(buffer) == {(1, 1, 1), (2, 2, 2)}


def test_voxels_seen_in_earlier_add_are_not_re_added() -> None:
    """Voxels already stored by a prior add do not grow the buffer again."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")

    buffer.add(*_inputs([[0.0, 0.0, 0.0]], [[1, 1, 1]], [0.9]))

    # same voxel as the first add -> nothing new is appended
    result = buffer.add(*_inputs([[0.005, 0.005, 0.005]], [[2, 2, 2]], [0.9]))
    assert result.shape == (buffer.capacity, 3)
    assert buffer.point_count == 1
    assert _stored_colors(buffer) == {(1, 1, 1)}

    # a previously unseen voxel does extend the buffer
    result = buffer.add(*_inputs([[0.5, 0.5, 0.5]], [[3, 3, 3]], [0.9]))
    assert result.shape == (buffer.capacity, 3)
    assert buffer.point_count == 2
    assert _stored_colors(buffer) == {(1, 1, 1), (3, 3, 3)}


def _row_set(array: np.ndarray) -> set[tuple[float, ...]]:
    """Return the rows of ``array`` as an order-free set of float tuples."""
    return {tuple(float(value) for value in row) for row in np.asarray(array)}


def test_resample_xyzrgb_upsamples_and_preserves_max() -> None:
    """Upsampling repeats real rows, keeps float32, and preserves the max."""
    xyzrgb = jnp.asarray(
        [
            [0.0, 1.0, 2.0, 0.1, 0.2, 0.3],
            [3.0, 4.0, 5.0, 0.4, 0.5, 0.6],
        ],
        dtype=jnp.float32,
    )

    out = HouseContextPoseBuffer.resample_xyzrgb(xyzrgb, max_points=5)

    assert out.shape == (5, 6)
    assert out.dtype == jnp.float32
    input_rows = _row_set(np.asarray(xyzrgb))
    assert _row_set(np.asarray(out)).issubset(input_rows)
    assert np.array_equal(
        np.asarray(jnp.max(out, axis=0)),
        np.asarray(jnp.max(xyzrgb, axis=0)),
    )


def test_resample_xyzrgb_empty_input_returns_zeros() -> None:
    """A ``(0, 6)`` input yields an all-zeros ``(max_points, 6)`` float32 array."""
    out = HouseContextPoseBuffer.resample_xyzrgb(
        jnp.zeros((0, 6), dtype=jnp.float32), max_points=4
    )

    assert out.shape == (4, 6)
    assert out.dtype == jnp.float32
    assert np.array_equal(np.asarray(out), np.zeros((4, 6), dtype=np.float32))


def test_resample_xyzrgb_downsamples_to_max_points() -> None:
    """Downsampling reduces ``(10, 6)`` to ``(4, 6)`` float32 via even stride."""
    xyzrgb = jnp.asarray(
        [[float(i)] * 6 for i in range(10)],
        dtype=jnp.float32,
    )

    out = HouseContextPoseBuffer.resample_xyzrgb(xyzrgb, max_points=4)

    assert out.shape == (4, 6)
    assert out.dtype == jnp.float32
    assert _row_set(np.asarray(out)).issubset(_row_set(np.asarray(xyzrgb)))


def test_resample_xyzrgb_no_int32_overflow_at_large_point_counts() -> None:
    """Stride indices stay exact when ``point_count * max_points > 2**31``.

    Regression: ``arange(max_points, int32) * point_count`` used to wrap to
    negative indices (e.g. 65536 * 209806), collapsing coverage to a fraction
    of the stored cloud.
    """
    point_count = 600_000
    max_points = 4096
    assert point_count * max_points > 2**31
    xyzrgb = jnp.zeros((point_count, 6), dtype=jnp.float32)
    xyzrgb = xyzrgb.at[:, 0].set(jnp.arange(point_count, dtype=jnp.float32))

    out = HouseContextPoseBuffer.resample_xyzrgb(xyzrgb, max_points=max_points)

    expected_indices = (
        np.arange(max_points, dtype=np.int64) * point_count
    ) // max_points
    np.testing.assert_array_equal(
        np.asarray(out[:, 0]), expected_indices.astype(np.float32)
    )


def test_house_context_snapshot_no_int32_overflow_at_large_sizes() -> None:
    """Snapshot striding covers the full store when ``size * max_points > 2**31``."""
    from src.buffer.house_context_pose_buffer import (
        _house_context_snapshot,
        _VoxelContextState,
    )

    size = 600_000
    max_points = 4096
    assert size * max_points > 2**31
    positions = jnp.arange(size, dtype=jnp.float32) * 0.01
    state = _VoxelContextState(
        key_xyz=jnp.zeros((1, 3), dtype=jnp.int32),
        occupied=jnp.zeros((1,), dtype=jnp.bool_),
        store_xyz=jnp.stack([positions] * 3, axis=1).astype(jnp.bfloat16),
        store_rgb=jnp.zeros((size, 3), dtype=jnp.uint8),
        size=jnp.asarray(size, dtype=jnp.int32),
        overflow_count=jnp.asarray(0, dtype=jnp.int32),
        failed_insert_count=jnp.asarray(0, dtype=jnp.int32),
    )

    snapshot, count = _house_context_snapshot(state, max_points)
    snapshot = np.asarray(snapshot)

    assert int(count) == max_points
    first_column = snapshot[:, 0]
    assert np.all(np.diff(first_column) >= 0.0)
    expected_last = 0.01 * size * (max_points - 1) / max_points
    assert first_column[-1] >= 0.99 * expected_last


def test_house_context_array_empty_buffer_returns_zeros() -> None:
    """A fresh buffer yields an all-zeros ``(max_points, 6)`` array, count 0."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")

    arr, count = buffer.house_context_array(max_points=8)

    assert arr.shape == (8, 6)
    assert arr.dtype == jnp.float32
    assert int(count) == 0
    assert np.array_equal(np.asarray(arr), np.zeros((8, 6), dtype=np.float32))


def test_house_context_array_after_add_matches_stored_points() -> None:
    """Valid rows carry admitted xyz/RGB; rows past ``count`` are zero padding."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")
    vggt_output, observation = _inputs(
        points=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        colors=[[10, 20, 30], [40, 50, 60]],
        confidences=[0.9, 0.9],
    )
    buffer.add(vggt_output, observation)

    arr, count = buffer.house_context_array(max_points=8)

    assert arr.shape == (8, 6)
    assert arr.dtype == jnp.float32
    valid_count = int(count)
    assert valid_count == buffer.point_count

    stored_xyz_rows = _row_set(_stored_xyz(buffer))
    valid_rows = np.asarray(arr)[:valid_count]
    for row in valid_rows:
        assert tuple(float(value) for value in row[:3]) in stored_xyz_rows

    assert float(arr[:, 3:6].max()) <= 1.0
    # Normalize through the same JAX float32 path the buffer uses so the
    # comparison is not defeated by JAX/NumPy last-bit rounding differences.
    stored_rgb01 = np.asarray(
        buffer.colors_rgb[: buffer.point_count].astype(jnp.float32) / 255.0
    )
    for row in valid_rows:
        matches = np.all(np.isclose(stored_rgb01, row[3:6], atol=1e-6), axis=1)
        assert bool(matches.any())

    padding = np.asarray(arr)[valid_count:]
    assert np.array_equal(padding, np.zeros_like(padding))


def test_seed_xyzrgb_registers_voxels_and_stores_uint8_colors() -> None:
    """Seeding stores uint8 colors and blocks re-adding the seeded voxel."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")
    buffer.seed_xyzrgb(
        jnp.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        )
    )

    assert buffer.point_count == 2
    assert _stored_colors(buffer) == {(255, 0, 0), (0, 255, 0)}

    # a point in the same voxel as the origin seed must not grow the buffer
    buffer.add(*_inputs([[0.001, 0.001, 0.001]], [[9, 9, 9]], [0.9]))
    assert buffer.point_count == 2
    assert _stored_colors(buffer) == {(255, 0, 0), (0, 255, 0)}

    # a point in a brand-new voxel does grow the buffer
    buffer.add(*_inputs([[2.0, 2.0, 2.0]], [[7, 8, 9]], [0.9]))
    assert buffer.point_count == 3
    assert _stored_colors(buffer) == {(255, 0, 0), (0, 255, 0), (7, 8, 9)}


def test_seed_xyzrgb_empty_seed_leaves_buffer_empty() -> None:
    """An empty static seed is a no-op, not a shape error."""
    buffer = HouseContextPoseBuffer(confidence_score=0.5, scene_id="scene")

    buffer.seed_xyzrgb(jnp.zeros((0, 6), dtype=jnp.float32))

    assert buffer.point_count == 0
