"""Graph signal-processing operators over ``KnnGraph`` edge lists.

Implements the storage-relevant tools of "Graph Spectral Image Processing":
- sparse high-pass filtering ``f(X) = L X`` and contour-aware downsampling
  scores ``pi_i ~ ||(L X)_i||`` (chapter 7, section 7.4.2.1, Chen et al. 2018);
- block-wise Graph Fourier Transform coding of RGB attributes (chapter 5,
  section 5.2), where the Laplacian eigendecomposition is kept tractable by
  partitioning the cloud into voxel blocks.

Laplacian/eigh numerics run in float32 (precision-sensitive, same exemption
from the bfloat16 default as voxel-key math).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.prototype_helpers.knn_graph import KnnGraph, node_degrees


def graph_high_pass(signal: jax.Array, graph: KnnGraph) -> jax.Array:
    """Apply the sparse graph Laplacian: ``(L X)_i = d_i x_i - sum_j w_ij x_j``.

    ``signal`` is ``(N, C)`` float32; returns ``(N, C)`` float32. Constant
    signals are annihilated (DC lies in the Laplacian nullspace), so the
    output magnitude measures local variation.
    """
    values = jnp.asarray(signal, dtype=jnp.float32)
    if values.ndim != 2 or values.shape[0] != graph.num_nodes:
        raise ValueError(
            f"expected signal shape ({graph.num_nodes}, C), got {values.shape}"
        )
    neighbor_sum = jax.ops.segment_sum(
        graph.weights[:, None] * values[graph.receivers],
        graph.senders,
        num_segments=graph.num_nodes,
    )
    return node_degrees(graph)[:, None] * values - neighbor_sum


def local_variation_scores(
    xyz: jax.Array,
    graph: KnnGraph,
    rgb: jax.Array | None = None,
    rgb_weight: float = 0.0,
) -> jax.Array:
    """Contour-aware sampling scores ``pi_i ~ ||(L X)_i||_2`` (Ch 7 s7.4.2.1).

    High-pass response over xyz measures geometric local variation (edges,
    corners); an optional rgb term (rgb scaled to ``[0, 1]``) adds color-edge
    sensitivity. Returns ``(N,)`` float32 scores clipped to a tiny positive
    floor so ``log(scores)`` stays finite for Gumbel sampling.
    """
    scores = jnp.linalg.norm(graph_high_pass(xyz, graph), axis=-1)
    if rgb is not None and rgb_weight > 0.0:
        rgb01 = jnp.asarray(rgb, dtype=jnp.float32) / 255.0
        scores = scores + rgb_weight * jnp.linalg.norm(
            graph_high_pass(rgb01, graph), axis=-1
        )
    return jnp.maximum(scores, 1e-12)


def gumbel_topk_sample(key: jax.Array, scores: jax.Array, m: int) -> jax.Array:
    """Sample ``m`` distinct indices with ``P(i) ~ scores_i`` (Gumbel top-k).

    Adding Gumbel(0, 1) noise to ``log(scores)`` and taking the top ``m``
    entries draws without replacement from the normalized score distribution
    while staying inside JAX with a static output shape ``(m,)`` int32.
    """
    if m <= 0 or m > scores.shape[0]:
        raise ValueError(f"m must be in [1, {scores.shape[0]}], got {m}")
    gumbel_noise = jax.random.gumbel(key, scores.shape, dtype=jnp.float32)
    _, indices = jax.lax.top_k(jnp.log(scores) + gumbel_noise, m)
    return indices.astype(jnp.int32)


def voxel_block_keys(xyz: jax.Array, block_size_m: float) -> jax.Array:
    """Quantize ``(N, 3)`` points to ``(N, 3)`` int32 voxel-block keys.

    Same ``floor(xyz / size)`` idiom as
    ``src.buffer.house_context_pose_buffer._quantize_points``.
    """
    if block_size_m <= 0.0:
        raise ValueError(f"block_size_m must be positive, got {block_size_m}")
    scaled = jnp.asarray(xyz, dtype=jnp.float32) / jnp.float32(block_size_m)
    return jnp.floor(scaled).astype(jnp.int32)


def group_indices_by_block(keys: jax.Array) -> list[np.ndarray]:
    """Group point indices by identical voxel-block key (host-side).

    Returns one int array of point indices per occupied block. Blocks vary in
    size, so downstream consumers loop over them in Python — acceptable for an
    offline experiment and avoids padding/vmap complexity.
    """
    keys_np = np.asarray(jax.device_get(keys))
    _, inverse = np.unique(keys_np, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.flatnonzero(np.diff(sorted_inverse)) + 1
    return np.split(order, boundaries)


def block_gft(
    xyz_block: jax.Array,
    rgb_block: jax.Array,
    k: int = 8,
    sigma: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Graph Fourier Transform of RGB over one voxel block (Ch 5 s5.2).

    Builds a dense symmetric k-nearest weight matrix within the block
    (block sizes are small enough for ``(B, B)`` float32), forms the
    combinatorial Laplacian ``L = D - W``, and projects the block's RGB onto
    the eigenbasis. ``jnp.linalg.eigh`` returns eigenvalues in ascending
    order, so leading coefficients are the low graph frequencies.

    Args:
        xyz_block: ``(B, 3)`` float32 coordinates, ``B >= 2``.
        rgb_block: ``(B, 3)`` float32 colors in ``[0, 1]``.
        k: Neighbors kept per row of the dense distance matrix.
        sigma: Gaussian bandwidth; default = mean kept-neighbor distance.

    Returns:
        ``(eigenvalues (B,), basis U (B, B), coefficients (B, 3))`` with
        ``rgb_block == U @ coefficients`` up to float32 error.
    """
    points = jnp.asarray(xyz_block, dtype=jnp.float32)
    colors = jnp.asarray(rgb_block, dtype=jnp.float32)
    block_size = points.shape[0]
    if block_size < 2:
        raise ValueError(f"block needs >= 2 points, got {block_size}")

    deltas = points[:, None, :] - points[None, :, :]
    distances = jnp.linalg.norm(deltas, axis=-1)
    distances = distances.at[jnp.diag_indices(block_size)].set(jnp.inf)

    neighbor_count = min(k, block_size - 1)
    kth_distance = jnp.sort(distances, axis=-1)[:, neighbor_count - 1]
    keep = distances <= kth_distance[:, None]

    if sigma is None:
        kept_distance_sum = float(jnp.sum(jnp.where(keep, distances, 0.0)))
        kept_count = float(jnp.sum(keep))
        sigma = max(kept_distance_sum / max(kept_count, 1.0), 1e-9)
    weights = jnp.where(keep, jnp.exp(-(distances**2) / (sigma**2)), 0.0)
    weights = jnp.maximum(weights, weights.T)

    laplacian = jnp.diag(weights.sum(axis=-1)) - weights
    eigenvalues, basis = jnp.linalg.eigh(laplacian)
    coefficients = basis.T @ colors
    return eigenvalues, basis, coefficients


def truncate_coeffs(
    coeffs: jax.Array, keep_fraction: float, mode: str = "lowfreq"
) -> tuple[jax.Array, int]:
    """Zero all but ``ceil(keep_fraction * B)`` coefficient rows.

    ``mode="lowfreq"`` keeps the leading (low graph frequency) rows — the
    honest codec, since kept indices are implicit. ``mode="energy"`` keeps the
    rows with the largest L2 norm — an oracle upper bound that would need the
    kept indices as side information.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    total_rows = coeffs.shape[0]
    kept = max(1, int(np.ceil(keep_fraction * total_rows)))
    if mode == "lowfreq":
        mask = jnp.arange(total_rows) < kept
    elif mode == "energy":
        energies = jnp.linalg.norm(coeffs, axis=-1)
        # top_k of the row order breaks energy ties deterministically, so the
        # kept-row count always matches the bytes charged for it (a >=
        # threshold mask would keep every tied row).
        _, top_rows = jax.lax.top_k(energies, kept)
        mask = jnp.zeros((total_rows,), dtype=bool).at[top_rows].set(True)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return jnp.where(mask[:, None], coeffs, 0.0), kept
