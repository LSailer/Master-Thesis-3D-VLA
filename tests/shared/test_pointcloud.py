"""Tests for device-side point-cloud geometry ops."""
import jax.numpy as jnp
import pytest

from src.shared.pointcloud import voxel_down_sample


def test_rejects_non_positive_voxel_size():
    xyz = jnp.zeros((1, 3))
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        voxel_down_sample(xyz, xyz, 0.0)


def test_points_in_one_voxel_average_to_one_row():
    xyz = jnp.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]])
    rgb = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    xyz_down, rgb_down = voxel_down_sample(xyz, rgb, 1.0)
    assert xyz_down.shape == (1, 3)
    assert jnp.allclose(xyz_down[0], jnp.array([0.2, 0.2, 0.2]))
    assert jnp.allclose(rgb_down[0], jnp.array([0.5, 0.0, 0.5]))


def test_points_in_distinct_voxels_stay_separate():
    xyz = jnp.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [-0.5, 0.5, 0.5]])
    rgb = jnp.ones_like(xyz)
    xyz_down, _ = voxel_down_sample(xyz, rgb, 1.0)
    assert xyz_down.shape == (3, 3)


def test_grid_is_origin_anchored_via_floor():
    # -0.1 and +0.1 straddle the origin plane -> two voxels even though
    # they are only 0.2 m apart (Open3D would anchor at min_bound instead).
    xyz = jnp.array([[-0.1, 0.5, 0.5], [0.1, 0.5, 0.5]])
    rgb = jnp.zeros_like(xyz)
    xyz_down, _ = voxel_down_sample(xyz, rgb, 1.0)
    assert xyz_down.shape == (2, 3)


def test_bfloat16_input_is_assigned_and_returned_in_float32():
    # Offsets below bf16 resolution at 100 m must not merge/shift voxels:
    # the cast to float32 happens before the divide by voxel_size.
    xyz = jnp.array([[100.25, 0.5, 0.5], [100.75, 0.5, 0.5]], dtype=jnp.bfloat16)
    rgb = jnp.zeros_like(xyz)
    xyz_down, rgb_down = voxel_down_sample(xyz, rgb, 0.5)
    assert xyz_down.dtype == jnp.float32
    assert rgb_down.dtype == jnp.float32
    assert xyz_down.shape == (2, 3)


def test_mean_reduction_matches_manual_groupby():
    xyz = jnp.array(
        [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [2.1, 0.1, 0.1], [2.9, 0.9, 0.9]]
    )
    rgb = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )
    xyz_down, rgb_down = voxel_down_sample(xyz, rgb, 1.0)
    assert xyz_down.shape == (2, 3)
    # jnp.unique sorts keys, so voxel (0,0,0) precedes voxel (2,0,0).
    assert jnp.allclose(xyz_down[0], jnp.array([0.5, 0.5, 0.5]))
    assert jnp.allclose(xyz_down[1], jnp.array([2.5, 0.5, 0.5]))
    assert jnp.allclose(rgb_down[0], jnp.array([0.5, 0.5, 0.0]))
    assert jnp.allclose(rgb_down[1], jnp.array([0.5, 0.5, 1.0]))
