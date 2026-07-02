"""Sparse graph-convolution modules over ``KnnGraph`` edge lists.

Adapts the GCN of the UvA JAX tutorial 7 to house-scale point clouds: the
tutorial's dense ``adj @ feats`` aggregation is O(N^2) memory (~40 GB at
100k+ nodes), so aggregation here is a weighted mean via
``jax.ops.segment_sum`` over sender/receiver edge arrays — O(N * k).

Self-loops are not present in the graph; each layer instead concatenates the
node's own features with the neighborhood aggregate before the dense
projection (GraphSAGE-style), which preserves self-information and lets the
layer weight the two sources independently.

Float32 throughout (training-stability exemption from the bfloat16 default).
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class GCNLayer(nn.Module):
    """One sparse graph-convolution layer with weighted-mean aggregation."""

    features: int

    @nn.compact
    def __call__(
        self,
        node_feats: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        weights: jax.Array,
        num_nodes: int,
    ) -> jax.Array:
        """Return ``(N, features)`` from ``(N, F)`` node features.

        ``num_nodes`` must be a Python int so segment shapes stay static
        under ``jax.jit``.
        """
        messages = weights[:, None] * node_feats[receivers]
        aggregate = jax.ops.segment_sum(messages, senders, num_segments=num_nodes)
        degree = jax.ops.segment_sum(weights, senders, num_segments=num_nodes)
        neighborhood = aggregate / (degree[:, None] + 1e-8)
        hidden = nn.Dense(self.features)(
            jnp.concatenate([node_feats, neighborhood], axis=-1)
        )
        return nn.silu(hidden)


class RgbGcnAutoencoder(nn.Module):
    """Reconstruct per-node RGB from corrupted inputs via stacked GCN layers.

    Input node features are ``concat(normalized xyz, corrupted rgb01, mask
    flag)``; output is ``(N, 3)`` rgb01 predictions through a sigmoid.
    """

    hidden: int = 64
    num_layers: int = 3

    @nn.compact
    def __call__(
        self,
        node_feats: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        weights: jax.Array,
        num_nodes: int,
    ) -> jax.Array:
        hidden = node_feats
        for _ in range(self.num_layers):
            hidden = GCNLayer(self.hidden)(
                hidden, senders, receivers, weights, num_nodes
            )
        return nn.sigmoid(nn.Dense(3)(hidden))
