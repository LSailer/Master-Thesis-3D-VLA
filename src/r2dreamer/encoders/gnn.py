"""Graph-neural-network cloud branch over a k-NN graph of the point cloud.

The branch the routed composite encoder instantiates for ``Encoder.GNN``
fields. Unlike the PointNet branch it lets each node see its spatial
neighborhood, which is what the accumulated house map needs: a global max pool
over independent points discards the layout the graph edges encode.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.r2dreamer.encoders.mlp import RMSNorm


class GnnCloudEncoder(nn.Module):
    """k-NN graph + weighted-mean GCN over one unbatched ``(N, point_dim)`` cloud.

    Same architecture as ``GnnHousePointsCameraEncoder._house_embedding``
    (even-stride node subsample, brute-force k-NN with Gaussian edge weights in
    normalized coordinates, GraphSAGE-style weighted-mean layers via
    ``segment_sum``, mean+max pool) without the snapshot/camera plumbing.
    50k-step-validated as the house branch of the live L1 pipeline.

    The graph is built inside the jitted call: k-NN is a dense ``(M, M)``
    distance matrix plus ``top_k`` rather than jaxkd, which keeps every op
    jit/grad-safe and avoids the known jaxkd CUDA k=1 segfault. Graph and GCN
    math stay float32 - a deliberate exemption from the bfloat16 default, for
    training stability.

    Attributes:
        message_mode: ``"sage"`` pools raw neighbor features (the validated
            baseline). ``"edgeconv"`` runs a per-edge Dense over
            ``[x_j, x_j - x_i, p_j - p_i]`` first (relative-position messages
            per PosPool/EdgeConv, arXiv:2007.01294, arXiv:1801.07829).
        residual: Residual adds around layers whose widths match, which
            mitigates oversmoothing (arXiv:2501.00762).
    """

    num_graph_nodes: int = 4096
    knn_k: int = 8
    point_dim: int = 6
    embed_dim: int = 1024
    gcn_hidden: int = 128
    gcn_layers: int = 2
    message_mode: str = "sage"
    residual: bool = False

    @nn.compact
    def __call__(self, points: jnp.ndarray) -> jnp.ndarray:
        """Encode ``(N, point_dim)`` points into one ``(embed_dim,)`` vector."""
        if points.ndim != 2 or points.shape[-1] != self.point_dim:
            raise ValueError(
                f"expected (N, {self.point_dim}) cloud, got {points.shape}"
            )
        if self.message_mode not in ("sage", "edgeconv"):
            raise ValueError(
                f"message_mode must be 'sage' or 'edgeconv', got {self.message_mode!r}"
            )
        n_points = points.shape[0]
        m = min(self.num_graph_nodes, n_points)
        k = min(self.knn_k, m - 1)
        node_idx = (jnp.arange(m, dtype=jnp.int32) * n_points) // m
        nodes = jnp.asarray(points[node_idx], jnp.float32)
        xyz, rgb = nodes[:, :3], nodes[:, 3:]

        center = xyz.mean(axis=0)
        scale = jnp.maximum(xyz.std(), 1e-6)
        xyz_n = (xyz - center) / scale
        feats = jnp.concatenate([xyz_n, rgb], axis=-1)

        # Brute-force k-NN in normalized coordinates; +inf diagonal drops the
        # self-match (re-injected below via the GraphSAGE concat).
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
        # Edge geometry for "edgeconv": the receiving node is senders[e] (the
        # segment_sum target), its neighbor is receivers[e].
        rel_pos = xyz_n[receivers] - xyz_n[senders]

        x = feats
        for i in range(self.gcn_layers):
            if self.message_mode == "edgeconv":
                edge_in = jnp.concatenate(
                    [x[receivers], x[receivers] - x[senders], rel_pos], axis=-1
                )
                messages = weights[:, None] * nn.Dense(
                    self.gcn_hidden, name=f"gnn_edge{i}"
                )(edge_in)
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

        pooled = jnp.concatenate([x.mean(axis=0), x.max(axis=0)], axis=-1)
        return nn.Dense(self.embed_dim, name="gnn_house_proj")(pooled)
