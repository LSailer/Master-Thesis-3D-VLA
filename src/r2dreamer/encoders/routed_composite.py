"""Routing-driven composite encoder built from ``AdapterField`` metadata.

Instead of resolving one encoder class from ``cfg.encoder_type``, this module
composes one branch per observation key from the adapter's routed fields: the
``Encoder`` enum on each field selects the branch architecture, and global
(``buffer=False``) fields are encoded once per batch and broadcast over the
``(B, T)`` leading dims of the per-step fields.

The branches reuse the existing modules where their contracts fit
(``ConvEncoder`` handles arbitrary leading dims natively); the PointNet cloud
branch is a standalone re-composition of the proven PointNet math (``TNet`` +
shared MLPs + max pool) because ``PointNetHousePointsCameraEncoder`` is
inseparable from its camera-pose plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jax.typing import DTypeLike

from src.adapters.contract import AdapterOutput, Encoder
from src.r2dreamer.encoders.cnn import ConvEncoder
from src.r2dreamer.encoders.mlp import RMSNorm
from src.r2dreamer.encoders.pointnet import TNet


class PointNetCloudEncoder(nn.Module):
    """Classic PointNet over one unbatched ``(N, point_dim)`` cloud.

    Same architecture as ``PointNetHousePointsCameraEncoder._house_embedding``
    (input/feature T-Nets, shared MLPs, max pool, optional projection) without
    the snapshot/camera plumbing: the input is the raw live cloud, subsampled
    with an even stride to at most ``num_points`` rows.

    TODO(house-padding): once the cloud is padded to a static ``max_points``
    with a valid-count, thread the count in and mask like production's
    ``HOUSE_CONTEXT_SIZE_KEY`` path — until then every new N recompiles.
    """

    num_points: int = 16384
    point_dim: int = 6
    embed_dim: int = 1024
    tnet_mlp: tuple[int, ...] = (64, 128, 1024)
    tnet_fc: tuple[int, ...] = (512, 256)
    mlp1: tuple[int, ...] = (64, 64)
    mlp2: tuple[int, ...] = (64, 128, 1024)
    compute_dtype: DTypeLike = jnp.bfloat16

    @nn.compact
    def __call__(self, points: jnp.ndarray) -> jnp.ndarray:
        """Encode ``(N, point_dim)`` points into one ``(embed_dim,)`` vector."""
        if points.ndim != 2 or points.shape[-1] != self.point_dim:
            raise ValueError(
                f"expected (N, {self.point_dim}) cloud, got {points.shape}"
            )
        n_points = points.shape[0]
        m = min(self.num_points, n_points)
        sample_idx = (jnp.arange(m, dtype=jnp.int32) * n_points) // m
        xyz = points[sample_idx, :3]

        # Centering/scale in float32: metric world offsets exceed bfloat16
        # precision. The normalized cloud is O(1), so casting down is safe.
        center = xyz.mean(axis=0)
        scale = jnp.maximum(xyz.std(), 1e-6)
        xyz_n = ((xyz - center) / scale).astype(self.compute_dtype)

        input_transform = TNet(
            k=3,
            mlp_widths=self.tnet_mlp,
            fc_widths=self.tnet_fc,
            compute_dtype=self.compute_dtype,
            name="input_tnet",
        )(xyz_n)
        x = xyz_n @ input_transform
        for i, width in enumerate(self.mlp1):
            x = nn.Dense(width, dtype=self.compute_dtype, name=f"mlp1_{i}")(x)
            x = RMSNorm(name=f"mlp1_norm{i}")(x)
            x = nn.silu(x)
        feature_transform = TNet(
            k=self.mlp1[-1],
            mlp_widths=self.tnet_mlp,
            fc_widths=self.tnet_fc,
            compute_dtype=self.compute_dtype,
            name="feature_tnet",
        )(x)
        x = x.astype(self.compute_dtype) @ feature_transform
        for i, width in enumerate(self.mlp2):
            x = nn.Dense(width, dtype=self.compute_dtype, name=f"mlp2_{i}")(x)
            x = RMSNorm(name=f"mlp2_norm{i}")(x)
            x = nn.silu(x)

        embed = x.max(axis=0)
        if self.mlp2[-1] != self.embed_dim:
            embed = nn.Dense(
                self.embed_dim, dtype=self.compute_dtype, name="proj"
            )(embed)
        return embed


class RoutedCompositeEncoder(nn.Module):
    """One branch per obs key, selected by adapter routing, concatenated.

    Attributes:
        routing: Sorted ``(key, encoder id)`` pairs (``Encoder.value`` ints;
            tuples keep the module hashable for Flax). Concatenation follows
            this order, so it must be deterministic across runs.
        global_keys: Keys whose obs leaf is one unbatched event (no ``(B, T)``
            prefix, e.g. the live house cloud). Encoded once, then broadcast
            over the per-step leading dims.
        conv_depth / conv_kernel / conv_mults: ``ConvEncoder`` branch config.
        pointnet_num_points: Subsample budget of the PointNet cloud branch.
        fusion_dim: When set, a final Dense fuses the concatenated branch
            embeddings down to this width, so ``embed_size`` stays fixed
            regardless of how many branches the routing composes.
        compute_dtype: Compute dtype forwarded to the branches.
    """

    routing: tuple[tuple[str, int], ...]
    global_keys: tuple[str, ...] = ()
    conv_depth: int = 16
    conv_kernel: int = 5
    conv_mults: tuple[int, ...] = (2, 3, 4, 4)
    pointnet_num_points: int = 16384
    fusion_dim: int | None = None
    compute_dtype: DTypeLike = jnp.float32

    def _branch(self, key: str, encoder: Encoder) -> nn.Module:
        if encoder is Encoder.CONV:
            return ConvEncoder(
                depth=self.conv_depth,
                kernel_size=self.conv_kernel,
                mults=self.conv_mults,
                compute_dtype=self.compute_dtype,
                name=f"conv_{key}",
            )
        if encoder is Encoder.POINTNET:
            return PointNetCloudEncoder(
                num_points=self.pointnet_num_points,
                name=f"pointnet_{key}",
            )
        raise NotImplementedError(
            f"no routed branch for {encoder} (field {key!r}) yet"
        )

    @nn.compact
    def __call__(self, obs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        """Encode the obs dict into ``(*leading, embed)`` concatenated features.

        ``leading`` comes from the per-step keys (e.g. ``(B, T)`` in training,
        ``(1,)`` at init/act time); global keys contribute one embedding
        broadcast to the same leading shape.
        """
        step_keys = [key for key, _ in self.routing if key not in self.global_keys]
        if not step_keys:
            raise ValueError("routing needs at least one per-step key")

        step_embeds: list[jnp.ndarray] = []
        global_embeds: list[jnp.ndarray] = []
        for key, encoder_id in self.routing:
            branch = self._branch(key, Encoder(encoder_id))
            if key in self.global_keys:
                global_embeds.append(branch(obs[key]))
            else:
                step_embeds.append(branch(obs[key]))

        leading = step_embeds[0].shape[:-1]
        step_out = jnp.concatenate(step_embeds, axis=-1)
        if global_embeds:
            global_out = jnp.concatenate(global_embeds, axis=-1)
            global_out = jnp.broadcast_to(
                global_out.astype(step_out.dtype), (*leading, global_out.shape[-1])
            )
            step_out = jnp.concatenate([step_out, global_out], axis=-1)
        if self.fusion_dim is not None:
            step_out = nn.Dense(
                self.fusion_dim, dtype=self.compute_dtype, name="fusion"
            )(step_out)
        return step_out


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
            key -> encoder routing and which keys are global (``buffer=False``).
        **overrides: ``RoutedCompositeEncoder`` attribute overrides.

    Returns:
        The routing-driven encoder module (parameters not yet initialized).
    """
    routing = tuple(sorted((f.key, f.encoder.value) for f in fields))
    global_keys = tuple(sorted(f.key for f in fields if not f.buffer))
    return RoutedCompositeEncoder(
        routing=routing, global_keys=global_keys, **overrides
    )


def dummy_obs_from_fields(fields: AdapterOutput) -> dict[str, jnp.ndarray]:
    """Zero observations shaped like ``fields`` for init-time shape discovery.

    Per-step keys get a leading singleton batch dim (matching the agent's
    dummy-forward convention); global keys keep their raw event shape.
    """
    return {
        f.key: jnp.zeros(
            (1, *jnp.shape(f.value)) if f.buffer else jnp.shape(f.value),
            dtype=f.value.dtype,
        )
        for f in fields
    }



