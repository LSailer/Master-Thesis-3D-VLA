"""Direct unit tests for the pure JAX voxel-hash kernel (CPU-only).

Exercises ``src.buffer.voxel_hash`` in isolation: the empty-state ->
add-frame -> snapshot round-trip, exact voxel dedup, and the overflow /
failed-insert counters on a capacity-constrained config.
"""
# pylint: disable=missing-function-docstring

import jax.numpy as jnp
import pytest

from src.buffer.voxel_hash import (
    VoxelContextConfig,
    add_frame_to_state,
    empty_state,
    house_context_snapshot,
    is_power_of_two,
)


def _config(*, capacity: int = 4, hash_table_size: int = 8) -> VoxelContextConfig:
    return VoxelContextConfig(
        voxel_size_m=1.0,
        confidence_score=0.0,
        hash_table_size=hash_table_size,
        capacity=capacity,
        max_probe_count=8,
    )


def test_is_power_of_two():
    assert is_power_of_two(1)
    assert is_power_of_two(8)
    assert not is_power_of_two(0)
    assert not is_power_of_two(6)


def test_empty_state_add_snapshot_round_trip():
    """A fresh state accumulates points and snapshots them zero-padded."""
    cfg = _config()
    state = empty_state(cfg.hash_table_size, cfg.capacity)
    assert int(state.size) == 0

    xyz = jnp.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    rgb = jnp.array([[10, 20, 30], [40, 50, 60]], dtype=jnp.uint8)
    confidence = jnp.ones((2,))

    state = add_frame_to_state(state, xyz, rgb, confidence, cfg)
    assert int(state.size) == 2

    snapshot, count = house_context_snapshot(state, cfg.capacity, dtype=jnp.float32)
    assert snapshot.shape == (cfg.capacity, 6)
    assert int(count) == 2
    # Rows beyond the valid count are zero padding.
    assert bool(jnp.all(snapshot[count:] == 0.0))


def test_exact_voxel_dedup():
    """Two points in the same voxel collapse to a single stored voxel."""
    cfg = _config()
    state = empty_state(cfg.hash_table_size, cfg.capacity)

    # Both points floor to voxel key (0, 0, 0) at voxel_size_m=1.0.
    xyz = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    rgb = jnp.zeros((2, 3), dtype=jnp.uint8)
    confidence = jnp.ones((2,))

    state = add_frame_to_state(state, xyz, rgb, confidence, cfg)
    assert int(state.size) == 1


def test_overflow_counter_on_capacity_one():
    """Novel voxels past capacity increment overflow, not size."""
    cfg = _config(capacity=1, hash_table_size=8)
    state = empty_state(cfg.hash_table_size, cfg.capacity)

    xyz = jnp.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]])
    rgb = jnp.zeros((3, 3), dtype=jnp.uint8)
    confidence = jnp.ones((3,))

    state = add_frame_to_state(state, xyz, rgb, confidence, cfg)
    assert int(state.size) == 1  # capped at capacity
    assert int(state.overflow_count) == 2  # two novel voxels could not be stored
    assert int(state.failed_insert_count) == 0  # hash table (size 8) had slots


def test_confidence_threshold_drops_low_points():
    """Points below the confidence threshold are not admitted."""
    cfg = VoxelContextConfig(
        voxel_size_m=1.0,
        confidence_score=1.5,
        hash_table_size=8,
        capacity=4,
        max_probe_count=8,
    )
    state = empty_state(cfg.hash_table_size, cfg.capacity)

    xyz = jnp.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    rgb = jnp.zeros((2, 3), dtype=jnp.uint8)
    confidence = jnp.array([2.0, 0.5])  # only the first clears the threshold

    state = add_frame_to_state(state, xyz, rgb, confidence, cfg)
    assert int(state.size) == 1
