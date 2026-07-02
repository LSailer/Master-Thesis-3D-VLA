"""Symmetrized k-NN graph over a point cloud (nodes = xyz, attribute = rgb).

Graph construction follows the model-based recipe of "Graph Spectral Image
Processing" chapter 5, section 5.3.1: k-nearest-neighbor topology with
Gaussian edge weights ``w_ij = exp(-dist(i, j)^2 / sigma^2)``.

The graph is a COO edge list (senders/receivers/weights) rather than an
adjacency matrix: at house scale (~210k nodes) a dense adjacency would need
O(N^2) memory, while the edge list is O(N * k).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxkd as jk


class KnnGraph(NamedTuple):
    """Symmetrized k-NN graph in COO edge-list form with static shapes.

    Attributes:
        senders: ``(E,)`` int32 source node index of each directed edge.
        receivers: ``(E,)`` int32 destination node index of each directed edge.
        weights: ``(E,)`` float32 Gaussian edge weights in ``(0, 1]``.
        num_nodes: Node count ``N`` (Python int, static under ``jax.jit``).
        k: Neighbors per node used at construction time.
        sigma: Gaussian kernel bandwidth in meters.

    ``E = 2 * N * k``: the directed k-NN edges plus their reversed copies.
    Mutual k-NN pairs therefore appear twice; that is harmless for
    degree-normalized aggregation and keeps ``E`` static for ``jax.jit``.
    There are no self-loops.
    """

    senders: jax.Array
    receivers: jax.Array
    weights: jax.Array
    num_nodes: int
    k: int
    sigma: float


def build_knn_graph(
    xyz: jax.Array,
    k: int = 16,
    sigma: float | None = None,
    cuda: bool = False,
) -> KnnGraph:
    """Build a symmetrized k-NN graph from ``(N, 3)`` float32 points.

    Queries ``k + 1`` neighbors per point and drops the first column
    (self-match). ``sigma=None`` uses the mean neighbor distance, so weights
    adapt to the cloud's sampling density. ``cuda=True`` uses the jaxkd CUDA
    extension (GPU nodes only).
    """
    points = jnp.asarray(xyz, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected xyz shape (N, 3), got {points.shape}")
    num_nodes = int(points.shape[0])
    if num_nodes <= k:
        raise ValueError(f"need more than k={k} points, got {num_nodes}")

    tree = jk.build_tree(points, cuda=cuda)
    neighbor_ids, neighbor_distances = jk.query_neighbors(
        tree, points, k=k + 1, cuda=cuda
    )
    # The query point is among its own k+1 nearest, but with exact duplicate
    # coordinates (bfloat16-stored clouds collapse distinct voxels beyond
    # ~2.56 m) ties are returned in arbitrary order, so the self-match is not
    # guaranteed to sit in column 0. Swap it into column 0, then drop that
    # column — keeps the graph self-loop-free with static shapes (consumers
    # re-inject self information explicitly, e.g. the GCN concat([x, agg])).
    neighbor_ids = neighbor_ids.astype(jnp.int32)
    neighbor_distances = neighbor_distances.astype(jnp.float32)
    row_ids = jnp.arange(num_nodes, dtype=jnp.int32)
    self_column = jnp.argmax(neighbor_ids == row_ids[:, None], axis=1)
    neighbor_ids = neighbor_ids.at[row_ids, self_column].set(neighbor_ids[:, 0])
    neighbor_distances = neighbor_distances.at[row_ids, self_column].set(
        neighbor_distances[:, 0]
    )
    neighbor_ids = neighbor_ids[:, 1:]
    neighbor_distances = neighbor_distances[:, 1:]

    if sigma is None:
        sigma = float(jnp.mean(neighbor_distances))
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    directed_senders = jnp.repeat(
        jnp.arange(num_nodes, dtype=jnp.int32), k
    )
    directed_receivers = neighbor_ids.reshape(-1)
    directed_weights = jnp.exp(
        -(neighbor_distances.reshape(-1) ** 2) / (sigma**2)
    )

    return KnnGraph(
        senders=jnp.concatenate([directed_senders, directed_receivers]),
        receivers=jnp.concatenate([directed_receivers, directed_senders]),
        weights=jnp.concatenate([directed_weights, directed_weights]),
        num_nodes=num_nodes,
        k=k,
        sigma=sigma,
    )


def node_degrees(graph: KnnGraph) -> jax.Array:
    """Return ``(N,)`` float32 weighted node degrees ``d_i = sum_j w_ij``."""
    return jax.ops.segment_sum(
        graph.weights, graph.senders, num_segments=graph.num_nodes
    )
