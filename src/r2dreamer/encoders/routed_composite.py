"""Routing-driven composite encoder built from ``AdapterField`` metadata.

Instead of resolving one encoder class from ``cfg.encoder_type``, this module
composes one branch per observation key from the adapter's routed fields: the
``Encoder`` enum on each field selects the branch architecture, and live
(``buffer=False``) fields are encoded once per batch and broadcast over the
``(B, T)`` leading dims of the per-step fields.

Each branch validates the shape it is handed in its own ``__call__``, so a
misrouted field fails at init with the branch's own message rather than deep
inside a jitted apply. Leading dims are whatever the caller supplies - ``(B, T)``
in training, ``(1,)`` at init - and every branch restores them, so
``embed.shape[:-1]`` always matches the replay batch.

This module only composes: every branch is a sibling module in this package,
so a reader chasing one architecture opens one file. The branches reuse the
existing modules where their contracts fit (``ConvEncoder`` and ``MLPEncoder``
handle arbitrary leading dims natively); the cloud branches are standalone
re-compositions of the proven PointNet and k-NN-GCN math, because
``PointNetHousePointsCameraEncoder`` and ``GnnHousePointsCameraEncoder`` are
inseparable from their camera-pose plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jax.typing import DTypeLike

from src.adapters.contract import AdapterOutput, Encoder
from src.r2dreamer.encoders.cnn import ConvEncoder
from src.r2dreamer.encoders.gnn import GnnCloudEncoder
from src.r2dreamer.encoders.mlp import MLPEncoder
from src.r2dreamer.encoders.pointnet import PointNetCloudEncoder
from src.r2dreamer.encoders.transformer import TokenSequenceEncoder


@dataclass(frozen=True)
class Route:
    """One branch assignment, derived from an :class:`AdapterField`.

    Attributes:
        key: Observation key this branch reads.
        encoder: Branch architecture.
        shape: Event shape of the field (no leading batch/time dims).
        live: Whether the field arrives as one global event that is encoded
            once and broadcast over the per-step leading dims (the adapter's
            ``buffer=False`` field).
    """

    key: str
    encoder: Encoder
    shape: tuple[int, ...]
    live: bool


def routes_from_fields(fields: AdapterOutput) -> tuple[Route, ...]:
    """Derive the sorted branch routing from one sample adapter output.

    Sorting by key makes concatenation order deterministic across runs, which
    keeps the params pytree (and hence checkpoints) stable. Each branch
    validates the shape it was handed itself, at init time.

    Args:
        fields: Adapter output for one representative frame. Field values carry
            no leading dims there, so their shape *is* the event shape.

    Returns:
        One :class:`Route` per field, sorted by key.
    """
    return tuple(
        Route(
            key=field.key,
            encoder=field.encoder,
            shape=tuple(jnp.shape(field.value)),
            live=not field.buffer,
        )
        for field in sorted(fields, key=lambda f: f.key)
    )


class RoutedCompositeEncoder(nn.Module):
    """One branch per obs key, selected by adapter routing, concatenated.

    Attributes:
        routes: Sorted branch assignments (see :func:`routes_from_fields`).
            Concatenation follows this order, so it must be deterministic
            across runs.
        conv_depth / conv_kernel / conv_mults: ``ConvEncoder`` branch config.
        mlp_hidden / mlp_layers: ``MLPEncoder`` branch config.
        branch_embed_dim: Output width of every non-conv branch.
        pointnet_num_points: Subsample budget of the PointNet cloud branch.
        gnn_num_nodes / gnn_message_mode / gnn_residual: GNN cloud branch config.
        transformer_layers / transformer_heads: Token branch config.
        transformer_compute_dtype: Compute dtype of the token branch only. Kept
            separate from ``compute_dtype`` because the token branch is the one
            path that never waited for the ``full_bf16`` gate: self-attention
            over the 1374 aggregator tokens is quadratic in sequence length and
            runs once per replay step, so it dominates the training arena and
            has been bfloat16 since it was introduced.
        fusion_dim: Width a final Dense fuses the concatenated branch embeddings
            down to. Applied exactly when the routing has more than one branch,
            so ``embed_size`` stays fixed no matter how many modalities a variant
            composes; a single-branch variant is its branch, unfused.
        compute_dtype: Compute dtype forwarded to the branches.
    """

    routes: tuple[Route, ...]
    conv_depth: int = 16
    conv_kernel: int = 5
    conv_mults: tuple[int, ...] = (2, 3, 4, 4)
    mlp_hidden: int = 1024
    mlp_layers: int = 1
    branch_embed_dim: int = 1024
    pointnet_num_points: int = 16384
    gnn_num_nodes: int = 4096
    gnn_message_mode: str = "sage"
    gnn_residual: bool = False
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_compute_dtype: DTypeLike = jnp.bfloat16
    fusion_dim: int = 1024
    compute_dtype: DTypeLike = jnp.float32

    @property
    def global_keys(self) -> tuple[str, ...]:
        """Keys encoded once per batch and broadcast over the leading dims."""
        return tuple(route.key for route in self.routes if route.live)

    def _branch(self, route: Route) -> nn.Module:
        key, encoder = route.key, route.encoder
        if encoder is Encoder.CONV:
            return ConvEncoder(
                depth=self.conv_depth,
                kernel_size=self.conv_kernel,
                mults=self.conv_mults,
                compute_dtype=self.compute_dtype,
                name=f"conv_{key}",
            )
        if encoder is Encoder.CONV_POINTS:
            return ConvEncoder(
                depth=self.conv_depth,
                kernel_size=self.conv_kernel,
                mults=self.conv_mults,
                input_kind="world_points",
                # Projected to a fixed width: a full-resolution point map
                # flattens to far more channels than an image does.
                embed_dim=self.branch_embed_dim,
                compute_dtype=self.compute_dtype,
                name=f"conv_points_{key}",
            )
        if encoder is Encoder.MLP:
            return MLPEncoder(
                embed_dim=self.branch_embed_dim,
                hidden=self.mlp_hidden,
                num_layers=self.mlp_layers,
                name=f"mlp_{key}",
            )
        if encoder is Encoder.POINTNET:
            return PointNetCloudEncoder(
                num_points=self.pointnet_num_points,
                point_dim=route.shape[-1],
                embed_dim=self.branch_embed_dim,
                name=f"pointnet_{key}",
            )
        if encoder is Encoder.GNN:
            return GnnCloudEncoder(
                num_graph_nodes=self.gnn_num_nodes,
                point_dim=route.shape[-1],
                embed_dim=self.branch_embed_dim,
                message_mode=self.gnn_message_mode,
                residual=self.gnn_residual,
                name=f"gnn_{key}",
            )
        if encoder is Encoder.TRANSFORMER:
            return TokenSequenceEncoder(
                num_tokens=route.shape[0],
                token_dim=route.shape[1],
                embed_dim=self.branch_embed_dim,
                layers=self.transformer_layers,
                heads=self.transformer_heads,
                compute_dtype=self.transformer_compute_dtype,
                name=f"transformer_{key}",
            )
        raise NotImplementedError(
            f"no routed branch for {encoder} (field {key!r}) yet"
        )

    @nn.compact
    def __call__(self, obs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        """Encode the obs dict into ``(*leading, embed)`` concatenated features.

        ``leading`` comes from the per-step keys (e.g. ``(B, T)`` in training,
        ``(1,)`` at init/act time); live keys contribute one embedding
        broadcast to the same leading shape.

        Raises:
            ValueError: If the routing has no per-step key to take the leading
                dims from.
        """
        step_embeds: list[jnp.ndarray] = []
        live_embeds: list[jnp.ndarray] = []
        for route in self.routes:
            embed = self._branch(route)(obs[route.key])
            (live_embeds if route.live else step_embeds).append(embed)
        if not step_embeds:
            raise ValueError("routing needs at least one per-step key")

        leading = step_embeds[0].shape[:-1]
        out = jnp.concatenate(step_embeds, axis=-1)
        if live_embeds:
            live_out = jnp.concatenate(live_embeds, axis=-1)
            live_out = jnp.broadcast_to(
                live_out.astype(out.dtype), (*leading, live_out.shape[-1])
            )
            out = jnp.concatenate([out, live_out], axis=-1)
        # More than one modality: fuse to a fixed width, so the RSSM's input
        # size does not depend on how many branches a variant happens to use.
        if len(self.routes) > 1:
            out = nn.Dense(
                self.fusion_dim, dtype=self.compute_dtype, name="fusion"
            )(out)
        return out


def routed_encoder_from_fields(
    fields: AdapterOutput, **overrides: Any
) -> RoutedCompositeEncoder:
    """Build the composite encoder from one sample adapter output.

    Branch hyperparameters default to the ``RoutedCompositeEncoder``
    attribute defaults; pass keyword overrides (``conv_depth=...``) to
    change them. The composition root (the agent) is responsible for
    translating its config into overrides so runs stay reproducible from
    the config alone.

    Args:
        fields: Adapter output for one representative frame; supplies the
            key -> encoder routing, the event shapes, and which keys are live
            (``buffer=False``).
        **overrides: ``RoutedCompositeEncoder`` attribute overrides.

    Returns:
        The routing-driven encoder module (parameters not yet initialized).
    """
    return RoutedCompositeEncoder(routes=routes_from_fields(fields), **overrides)
