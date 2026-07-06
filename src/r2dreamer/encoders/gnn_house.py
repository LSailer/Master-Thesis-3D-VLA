"""Sparse-GNN house-context encoders for the live house-points-pose pipeline.

Replaces the house branch of ``HousePointsCameraEncoder`` (per-point MLP +
one global masked pool) with a graph branch: even-stride subsample of the
valid snapshot rows to a static node set, brute-force k-NN graph with
Gaussian edge weights in normalized coordinates, GraphSAGE-style
weighted-mean GCN layers via ``segment_sum``, then mean/max pooling.

Validated end-to-end in the live L1 pipeline: 50k-step run (job 5736907,
completed, no NaN, representation parity with the MLP branch at +10% step
cost) and canonical smokes (jobs 5744355/5744356). Design history and
variant comparison: graduated from ``src/prototyp/gnn_house_encoder``
(see its RESULTS.md / DECISION_LOG.md while the folder exists).

Design notes:
    - The graph is built inside the jitted encoder call, so k-NN is a dense
      ``(M, M)`` distance matrix + ``top_k`` rather than jaxkd: at M=4096 the
      matrix is 64 MB, every op is jit/grad-safe, and the known jaxkd CUDA
      k=1 segfault is avoided.
    - The house cloud is a singleton snapshot broadcast across the batch
      (same contract as the parent), so exactly one graph is built per call.
    - Graph/GCN math stays float32 (training-stability exemption from the
      bfloat16 default, same as the parent encoder).
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.r2dreamer.encoders.house_points_pose import VGGTHousePointsPoseEncoder
from src.r2dreamer.encoders.mlp import HousePointsCameraEncoder, RMSNorm


class GnnHousePointsCameraEncoder(HousePointsCameraEncoder):
    """House branch = k-NN graph + GCN over a strided subset of the snapshot.

    Inherits the camera branch, obs plumbing, and singleton-broadcast
    behavior from ``HousePointsCameraEncoder``; only ``_house_embedding``
    changes.
    """

    num_graph_nodes: int = 4096
    knn_k: int = 8
    gcn_hidden: int = 128
    gcn_layers: int = 2
    # "sage": Gaussian-weighted mean of raw neighbor features (50k-validated
    # baseline, jobs 5736062/5736907). "edgeconv": per-edge Dense over
    # [x_j, x_j - x_i, p_j - p_i] before the weighted mean — relative-position
    # messages per PosPool/EdgeConv (arXiv:2007.01294, arXiv:1801.07829).
    message_mode: str = "sage"
    # Residual adds around layers whose input/output widths match (i.e. all
    # but the first); mitigates oversmoothing (arXiv:2501.00762).
    residual: bool = False

    def _house_embedding(
        self,
        house_points: jnp.ndarray,
        batch_size: int,
        house_size: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if house_points.ndim == 2:
            house_points = house_points[None]
        if house_points.ndim != 3 or house_points.shape[-1] != self.house_point_dim:
            raise ValueError(
                "house_points must have shape (N, 6) or (S, N, 6), "
                f"got {house_points.shape}"
            )
        if house_points.shape[0] != 1:
            raise ValueError(
                "GNN house branch expects a singleton house cloud (S=1), "
                f"got S={house_points.shape[0]}"
            )
        points = house_points[0].astype(jnp.float32)
        n_points = points.shape[0]
        if house_size is None:
            house_size = n_points
        size = jnp.asarray(house_size, dtype=jnp.int32).reshape(-1)[0]

        m = min(self.num_graph_nodes, n_points)
        k = min(self.knn_k, m - 1)
        # Even stride over the valid prefix; rows repeat when size < m, which
        # only yields duplicate nodes (zero-distance neighbors, weight 1).
        clamped = jnp.maximum(size, 1)
        node_idx = (jnp.arange(m, dtype=jnp.int32) * clamped) // m
        nodes = points[node_idx]
        xyz = nodes[:, :3]
        rgb = nodes[:, 3:]

        center = xyz.mean(axis=0)
        scale = jnp.maximum(xyz.std(), 1e-6)
        xyz_n = (xyz - center) / scale
        feats = jnp.concatenate([xyz_n, rgb], axis=-1)

        # Brute-force k-NN in normalized coordinates; +inf diagonal drops the
        # self-match (consumers re-inject self via the GraphSAGE concat).
        sq = jnp.sum(xyz_n**2, axis=-1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (xyz_n @ xyz_n.T)
        d2 = jnp.where(jnp.eye(m, dtype=bool), jnp.inf, jnp.maximum(d2, 0.0))
        neg_d2, neighbor_idx = jax.lax.top_k(-d2, k)
        neighbor_d2 = -neg_d2
        sigma2 = jnp.maximum(neighbor_d2.mean(), 1e-12)
        knn_weights = jnp.exp(-neighbor_d2 / sigma2)

        directed_senders = jnp.repeat(jnp.arange(m, dtype=jnp.int32), k)
        directed_receivers = neighbor_idx.reshape(-1).astype(jnp.int32)
        directed_weights = knn_weights.reshape(-1)
        senders = jnp.concatenate([directed_senders, directed_receivers])
        receivers = jnp.concatenate([directed_receivers, directed_senders])
        weights = jnp.concatenate([directed_weights, directed_weights])

        if self.message_mode not in ("sage", "edgeconv"):
            raise ValueError(
                f"message_mode must be 'sage' or 'edgeconv', got {self.message_mode!r}"
            )
        # Edge geometry for "edgeconv": receiving node is senders[e] (the
        # segment_sum target), neighbor is receivers[e].
        rel_pos = xyz_n[receivers] - xyz_n[senders]

        x = feats
        for i in range(self.gcn_layers):
            if self.message_mode == "edgeconv":
                edge_in = jnp.concatenate(
                    [x[receivers], x[receivers] - x[senders], rel_pos], axis=-1
                )
                edge_feats = nn.Dense(self.gcn_hidden, name=f"gnn_edge{i}")(edge_in)
                messages = weights[:, None] * edge_feats
            else:
                messages = weights[:, None] * x[receivers]
            aggregate = jax.ops.segment_sum(messages, senders, num_segments=m)
            degree = jax.ops.segment_sum(weights, senders, num_segments=m)
            neighborhood = aggregate / (degree[:, None] + 1e-8)
            h = nn.Dense(self.gcn_hidden, name=f"gnn_hidden{i}")(
                jnp.concatenate([x, neighborhood], axis=-1)
            )
            h = RMSNorm(name=f"gnn_norm{i}")(h)
            h = nn.silu(h)
            if self.residual and x.shape[-1] == h.shape[-1]:
                h = h + x
            x = h

        pooled = jnp.concatenate([x.mean(axis=0), x.max(axis=0)], axis=-1)[None]
        house_embed = nn.Dense(self.embed_dim, name="gnn_house_proj")(pooled)
        house_embed = jnp.where(size > 0, house_embed, jnp.zeros_like(house_embed))
        if batch_size != 1:
            house_embed = jnp.broadcast_to(house_embed, (batch_size, self.embed_dim))
        return house_embed


class GnnEdgeHousePointsCameraEncoder(GnnHousePointsCameraEncoder):
    """EdgeConv-variant house branch: relative-position messages + residuals.

    Identical graph construction to the baseline; only the message function
    (per-edge Dense over ``[x_j, x_j - x_i, p_j - p_i]``) and residual adds
    differ.
    """

    message_mode: str = "edgeconv"
    residual: bool = True


class GnnHousePointsPoseEncoder(VGGTHousePointsPoseEncoder):
    """Launcher-side selection for the GNN house encoder.

    Reuses the whole VGGT house-points-pose pipeline (adapter, live buffer,
    camera pose replay); only the agent-side Flax module changes.
    """

    @property
    def encoder_type(self) -> str:
        return "gnn_house_points_pose"

    @property
    def module_cls(self) -> type[nn.Module]:
        return GnnHousePointsCameraEncoder

    @property
    def design_notes(self) -> str:
        return (
            "Sparse k-NN GCN house branch over an even-stride node subset of "
            "the live house snapshot (src/r2dreamer/encoders/gnn_house.py)."
        )


class GnnEdgeHousePointsPoseEncoder(GnnHousePointsPoseEncoder):
    """Launcher-side selection for the EdgeConv-variant GNN house encoder."""

    @property
    def encoder_type(self) -> str:
        return "gnn_edge_house_points_pose"

    @property
    def module_cls(self) -> type[nn.Module]:
        return GnnEdgeHousePointsCameraEncoder

    @property
    def design_notes(self) -> str:
        return (
            "EdgeConv-variant GNN house branch — relative-position edge "
            "messages + residual layers over the same k-NN graph as the "
            "baseline GNN (src/r2dreamer/encoders/gnn_house.py)."
        )
