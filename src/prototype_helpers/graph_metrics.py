"""Metrics for graph point-cloud experiments: chamfer, PSNR, storage sizes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxkd as jk


def chamfer_distances(
    a: jax.Array, b: jax.Array, cuda: bool = False
) -> tuple[float, float]:
    """Return one-sided chamfer distances ``(mean_a min-dist to b, mean_b to a)``.

    Each direction builds a kd-tree over the target set and queries the
    nearest neighbor of every source point.
    """
    points_a = jnp.asarray(a, dtype=jnp.float32)
    points_b = jnp.asarray(b, dtype=jnp.float32)
    return (
        _one_sided_chamfer(points_a, points_b, cuda),
        _one_sided_chamfer(points_b, points_a, cuda),
    )


def _one_sided_chamfer(source: jax.Array, target: jax.Array, cuda: bool) -> float:
    tree = jk.build_tree(target, cuda=cuda)
    _, distances = jk.query_neighbors(tree, source, k=1, cuda=cuda)
    return float(jnp.mean(distances[:, 0]))


def rgb_psnr(rgb_true: jax.Array, rgb_pred: jax.Array) -> float:
    """PSNR in dB over the uint8 range (peak 255). Inf-safe for exact matches."""
    true = jnp.asarray(rgb_true, dtype=jnp.float32)
    pred = jnp.asarray(rgb_pred, dtype=jnp.float32)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    mse = float(jnp.mean((true - pred) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * jnp.log10(255.0**2 / mse))


def table_bytes(num_points: int, k: int) -> dict[str, int]:
    """Storage-size breakdown of the attributed graph, in bytes.

    Node table matches the live buffer layout (xyz bfloat16 + rgb uint8 =
    9 B/point). Edge tables assume the symmetrized ``E = 2 * N * k`` COO list;
    the ``implicit_senders`` variant drops the sender column because directed
    k-NN senders are just ``arange(N)`` repeated ``k`` times.
    """
    num_edges = 2 * num_points * k
    node_bytes = num_points * (3 * 2 + 3 * 1)
    edge_bytes_coo = num_edges * (4 + 4 + 2)  # int32 sender + int32 receiver + bf16 w
    edge_bytes_implicit = num_points * k * (4 + 2)  # int32 receiver + bf16 w
    return {
        "num_points": num_points,
        "num_edges": num_edges,
        "node_table_bytes": node_bytes,
        "edge_table_coo_bytes": edge_bytes_coo,
        "edge_table_implicit_senders_bytes": edge_bytes_implicit,
        "total_coo_bytes": node_bytes + edge_bytes_coo,
        "total_implicit_bytes": node_bytes + edge_bytes_implicit,
    }
