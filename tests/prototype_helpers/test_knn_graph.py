"""Structural checks for ``src.prototype_helpers.knn_graph``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.prototype_helpers.knn_graph import KnnGraph, build_knn_graph, node_degrees


def _two_clusters(points_per_cluster: int = 20, spacing: float = 0.05) -> jnp.ndarray:
    """Two well-separated point grids, 100 m apart along x."""
    side = int(np.ceil(np.sqrt(points_per_cluster)))
    grid = np.stack(
        np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), axis=-1
    ).reshape(-1, 2)[:points_per_cluster] * spacing
    cluster = np.concatenate([grid, np.zeros((points_per_cluster, 1))], axis=1)
    far_cluster = cluster + np.array([100.0, 0.0, 0.0])
    return jnp.asarray(np.concatenate([cluster, far_cluster]), dtype=jnp.float32)


def test_shapes_and_dtypes() -> None:
    xyz = _two_clusters()
    graph = build_knn_graph(xyz, k=4)

    num_nodes = xyz.shape[0]
    expected_edges = 2 * num_nodes * 4
    assert graph.num_nodes == num_nodes
    assert graph.senders.shape == (expected_edges,)
    assert graph.receivers.shape == (expected_edges,)
    assert graph.weights.shape == (expected_edges,)
    assert graph.senders.dtype == jnp.int32
    assert graph.weights.dtype == jnp.float32


def test_no_cross_cluster_edges() -> None:
    xyz = _two_clusters()
    graph = build_knn_graph(xyz, k=4)

    half = xyz.shape[0] // 2
    sender_cluster = np.asarray(graph.senders) < half
    receiver_cluster = np.asarray(graph.receivers) < half
    np.testing.assert_array_equal(sender_cluster, receiver_cluster)


def test_symmetry_and_no_self_loops() -> None:
    xyz = _two_clusters()
    graph = build_knn_graph(xyz, k=4)

    senders = np.asarray(graph.senders)
    receivers = np.asarray(graph.receivers)
    assert not np.any(senders == receivers)

    edge_set = set(zip(senders.tolist(), receivers.tolist()))
    reversed_set = {(receiver, sender) for sender, receiver in edge_set}
    assert edge_set == reversed_set


def test_weights_in_unit_interval_and_distance_monotone() -> None:
    xyz = _two_clusters()
    graph = build_knn_graph(xyz, k=4)

    weights = np.asarray(graph.weights)
    assert np.all(weights > 0.0)
    assert np.all(weights <= 1.0)

    distances = np.linalg.norm(
        np.asarray(xyz)[np.asarray(graph.senders)]
        - np.asarray(xyz)[np.asarray(graph.receivers)],
        axis=-1,
    )
    expected = np.exp(-(distances**2) / graph.sigma**2)
    np.testing.assert_allclose(weights, expected, rtol=1e-4, atol=1e-6)


def test_node_degrees_positive_everywhere() -> None:
    xyz = _two_clusters()
    graph = build_knn_graph(xyz, k=4)

    degrees = np.asarray(node_degrees(graph))
    assert degrees.shape == (graph.num_nodes,)
    assert np.all(degrees > 0.0)


def test_no_self_loops_with_exact_duplicate_points() -> None:
    """Duplicate coordinates must not create self-loops.

    Regression: bfloat16-stored clouds collapse distinct voxels to identical
    coordinates, and jaxkd returns 0-distance ties in arbitrary order, so the
    self-match is not always in column 0 of the k+1 query.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=(10, 3)).astype(np.float32)
    xyz = jnp.asarray(np.repeat(base, 4, axis=0))  # each point duplicated 4x

    graph = build_knn_graph(xyz, k=4)

    senders = np.asarray(graph.senders)
    receivers = np.asarray(graph.receivers)
    assert not np.any(senders == receivers)
    assert graph.senders.shape == (2 * xyz.shape[0] * 4,)


def test_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="need more than"):
        build_knn_graph(jnp.zeros((3, 3), dtype=jnp.float32), k=4)


def test_explicit_sigma_is_kept() -> None:
    xyz = _two_clusters()
    graph = build_knn_graph(xyz, k=4, sigma=0.25)
    assert isinstance(graph, KnnGraph)
    assert graph.sigma == 0.25
