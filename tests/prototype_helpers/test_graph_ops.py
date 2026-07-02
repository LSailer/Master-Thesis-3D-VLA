"""Behavior checks for ``src.prototype_helpers.graph_ops``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.prototype_helpers.graph_ops import (
    block_gft,
    graph_high_pass,
    group_indices_by_block,
    gumbel_topk_sample,
    local_variation_scores,
    truncate_coeffs,
    voxel_block_keys,
)
from src.prototype_helpers.knn_graph import build_knn_graph


def _stepped_plane() -> tuple[jnp.ndarray, np.ndarray, np.ndarray]:
    """Grid plane with a height step; returns (xyz, contour mask, interior mask).

    The interior of each flat level has fully symmetric k=8 grid
    neighborhoods, so the Laplacian response there is ~0; points along the
    step line and the outer boundary have asymmetric neighborhoods (the
    geometric "contours" of Ch 7 s7.4.2.1).
    """
    side = 40
    spacing = 0.02
    half = side // 2
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    heights = np.where(ys >= half, 0.1, 0.0)
    xyz = np.stack(
        [xs.reshape(-1) * spacing, ys.reshape(-1) * spacing, heights.reshape(-1)],
        axis=-1,
    ).astype(np.float32)

    on_outer_boundary = (
        (xs == 0) | (xs == side - 1) | (ys == 0) | (ys == side - 1)
    ).reshape(-1)
    on_step = ((ys == half - 1) | (ys == half)).reshape(-1)
    contour_mask = on_outer_boundary | on_step
    interior_mask = ~contour_mask
    return jnp.asarray(xyz), contour_mask, interior_mask


def test_high_pass_annihilates_constant_signal() -> None:
    xyz, _, _ = _stepped_plane()
    graph = build_knn_graph(xyz, k=8)
    constant = jnp.ones((graph.num_nodes, 3), dtype=jnp.float32) * 3.7

    response = graph_high_pass(constant, graph)

    np.testing.assert_allclose(np.asarray(response), 0.0, atol=1e-4)


def test_local_variation_highlights_contours() -> None:
    xyz, contour_mask, interior_mask = _stepped_plane()
    graph = build_knn_graph(xyz, k=8)

    scores = np.asarray(local_variation_scores(xyz, graph))

    assert np.median(scores[contour_mask]) > 5.0 * np.median(scores[interior_mask])


def test_gumbel_topk_returns_unique_indices_and_prefers_high_scores() -> None:
    scores = jnp.array([1000.0] + [1.0] * 99, dtype=jnp.float32)

    for seed in range(5):
        indices = gumbel_topk_sample(jax.random.PRNGKey(seed), scores, m=5)
        indices_np = np.asarray(indices)
        assert indices_np.shape == (5,)
        assert len(set(indices_np.tolist())) == 5
        assert np.all(indices_np >= 0) and np.all(indices_np < 100)
        assert 0 in indices_np


def test_contour_sampling_keeps_more_contour_points_than_stride() -> None:
    xyz, contour_mask, _ = _stepped_plane()
    # Shuffle point order: the raw grid order aliases with even-stride
    # sampling (every 10th point lands exactly on boundary/step columns),
    # which real insertion orders do not exhibit.
    permutation = np.random.default_rng(0).permutation(xyz.shape[0])
    xyz = xyz[permutation]
    contour_mask = contour_mask[permutation]
    graph = build_knn_graph(xyz, k=8)
    scores = local_variation_scores(xyz, graph)
    budget = 160

    contour_indices = np.asarray(
        gumbel_topk_sample(jax.random.PRNGKey(0), scores, m=budget)
    )
    contour_hit_count = int(np.sum(contour_mask[contour_indices]))

    # Even-stride keeps contour points only in proportion to their share of
    # the cloud; the point index is recoverable from stride semantics.
    xyzrgb = jnp.concatenate(
        [xyz, jnp.zeros((xyz.shape[0], 3), dtype=jnp.float32)], axis=-1
    )
    point_count = xyz.shape[0]
    stride_indices = np.minimum(
        (np.arange(budget) * point_count) // budget, point_count - 1
    )
    stride_rows = np.asarray(HouseContextPoseBuffer.resample_xyzrgb(xyzrgb, budget))
    np.testing.assert_allclose(stride_rows[:, :3], np.asarray(xyz)[stride_indices])
    stride_hit_count = int(np.sum(contour_mask[stride_indices]))

    assert contour_hit_count >= 2 * max(stride_hit_count, 1)


def test_voxel_block_keys_and_grouping_partition_all_points() -> None:
    xyz = jnp.array(
        [[0.1, 0.1, 0.1], [0.2, 0.3, 0.4], [1.1, 0.1, 0.1], [1.4, 0.2, 0.3]],
        dtype=jnp.float32,
    )
    keys = voxel_block_keys(xyz, block_size_m=1.0)
    blocks = group_indices_by_block(keys)

    assert len(blocks) == 2
    all_indices = np.sort(np.concatenate(blocks))
    np.testing.assert_array_equal(all_indices, np.arange(4))


def test_block_gft_roundtrip_at_full_rate() -> None:
    rng = np.random.default_rng(0)
    xyz = jnp.asarray(rng.normal(size=(50, 3)) * 0.1, dtype=jnp.float32)
    rgb = jnp.asarray(rng.uniform(size=(50, 3)), dtype=jnp.float32)

    _, basis, coeffs = block_gft(xyz, rgb, k=6)
    coeffs_full, kept = truncate_coeffs(coeffs, keep_fraction=1.0)
    reconstruction = basis @ coeffs_full

    assert kept == 50
    np.testing.assert_allclose(np.asarray(reconstruction), np.asarray(rgb), atol=1e-4)


def test_block_gft_constant_rgb_energy_in_dc() -> None:
    rng = np.random.default_rng(1)
    xyz = jnp.asarray(rng.normal(size=(40, 3)) * 0.1, dtype=jnp.float32)
    rgb = jnp.full((40, 3), 0.5, dtype=jnp.float32)

    eigenvalues, _, coeffs = block_gft(xyz, rgb, k=6)

    assert float(eigenvalues[0]) == pytest.approx(0.0, abs=1e-4)
    coeffs_np = np.asarray(coeffs)
    dc_energy = float(np.sum(coeffs_np[0] ** 2))
    rest_energy = float(np.sum(coeffs_np[1:] ** 2))
    assert dc_energy > 1000.0 * max(rest_energy, 1e-12)


def test_truncate_coeffs_energy_mode_keeps_largest_rows() -> None:
    coeffs = jnp.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.1, 0.1, 0.1], [3.0, 3.0, 3.0]],
        dtype=jnp.float32,
    )
    truncated, kept = truncate_coeffs(coeffs, keep_fraction=0.5, mode="energy")

    assert kept == 2
    truncated_np = np.asarray(truncated)
    np.testing.assert_allclose(truncated_np[1], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(truncated_np[3], [3.0, 3.0, 3.0])
    np.testing.assert_allclose(truncated_np[0], 0.0)
    np.testing.assert_allclose(truncated_np[2], 0.0)
