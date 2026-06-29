"""MLP-based encoder modules for flat and hybrid observations."""

import flax.linen as nn
import jax.numpy as jnp

from src.r2dreamer.encoders.cnn import ConvEncoder, make_rgb_conv_encoder
from src.r2dreamer.encoders.constants import (
    AGG_RAW_DIM,
    HYBRID_RGB_DIM,
    HYBRID_VGGT_DIM,
)
from src.r2dreamer.obs_batch import CAMERA_POSE_KEY, WORLD_POINTS_KEY
from src.r2dreamer.world_model.heads import R2MLP
from src.r2dreamer.world_model.rssm import RMSNorm


class MLPEncoder(nn.Module):
    """Generic flat-feature MLP encoder.

    Parameters:
        embed_dim: Output embedding width consumed by the RSSM posterior.
        hidden: Width of each hidden block.
        num_layers: Number of hidden ``Dense -> RMSNorm -> SiLU`` blocks.
            ``num_layers=0`` is a linear encoder: only ``Dense(embed_dim)``.

    Returns:
        A float array with shape ``(..., embed_dim)``. Leading dimensions are
        preserved by Flax dense broadcasting.
    """

    embed_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode flat observations into RSSM embeddings."""
        x = obs
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden, name=f"hidden{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        return nn.Dense(self.embed_dim, name="proj")(x)


class WP64CNNCPMLPEncoder(nn.Module):
    """CNN over 64x64 world points plus MLP over camera pose."""

    embed_dim: int = 1024
    conv_depth: int = 16
    conv_kernel: int = 5
    conv_mults: tuple[int, ...] = (2, 3, 4, 4)
    cp_hidden: int = 128
    cp_layers: int = 1

    @nn.compact
    def __call__(self, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Encode structured ``world_points`` and ``camera_pose`` fields."""
        if not isinstance(obs, dict):
            raise TypeError("WP64CNNCPMLPEncoder expects structured obs")
        world_points = jnp.asarray(obs[WORLD_POINTS_KEY], dtype=jnp.float32)
        camera_pose = jnp.asarray(obs[CAMERA_POSE_KEY], dtype=jnp.float32)
        world_points_embed = ConvEncoder(
            depth=self.conv_depth,
            kernel_size=self.conv_kernel,
            mults=self.conv_mults,
            input_kind="world_points",
            name="wp_conv",
        )(world_points)
        camera_pose_embed = camera_pose
        for i in range(self.cp_layers):
            camera_pose_embed = nn.Dense(self.cp_hidden, name=f"cp_hidden{i}")(
                camera_pose_embed
            )
            camera_pose_embed = RMSNorm(name=f"cp_norm{i}")(camera_pose_embed)
            camera_pose_embed = nn.silu(camera_pose_embed)
        camera_pose_embed = nn.Dense(self.cp_hidden, name="cp_proj")(camera_pose_embed)
        fused = jnp.concatenate([world_points_embed, camera_pose_embed], axis=-1)
        return nn.Dense(self.embed_dim, name="proj")(fused)


class VGGTAggregatorMLPEncoder(nn.Module):
    """MLP encoder for pre-pooled aggregator features.

    Input layout is ``[cam | mean_patches | max_patches]`` with shape
    ``(B, 3 * pool_dim)``. Each slice is normalized separately before the hidden
    MLP blocks so scale differences between readouts do not bleed into fusion.
    """

    embed_dim: int = 1024
    pool_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode pooled token features into an RSSM embedding."""
        if obs.ndim != 2 or obs.shape[-1] != 3 * self.pool_dim:
            raise ValueError(
                f"expected (B, {3 * self.pool_dim}) pooled features, got {obs.shape}"
            )
        camera, mean_patches, max_patches = jnp.split(obs, 3, axis=-1)
        camera = RMSNorm(name="norm_cam")(camera)
        mean_patches = RMSNorm(name="norm_mean")(mean_patches)
        max_patches = RMSNorm(name="norm_max")(max_patches)
        x = jnp.concatenate([camera, mean_patches, max_patches], axis=-1)
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden, name=f"hidden{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        return nn.Dense(self.embed_dim, name="proj")(x)


class VGGTAggRawMLPEncoder(nn.Module):
    """MLP over flattened raw aggregator tokens.

    Input ``(B, AGG_RAW_DIM)`` is camera token plus patch tokens with register
    tokens dropped. Replay stores float16; the cast to float32 here lets dense
    matmuls run in the agent's working precision.
    """

    embed_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 3

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode flattened raw token features into an RSSM embedding."""
        if obs.ndim != 2 or obs.shape[-1] != AGG_RAW_DIM:
            raise ValueError(
                f"VGGTAggRawMLPEncoder expects (B, {AGG_RAW_DIM}), got {obs.shape}"
            )
        x = obs.astype(jnp.float32)
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden, name=f"hidden{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        return nn.Dense(self.embed_dim, name="proj")(x)


class HybridEncoder(nn.Module):
    """Hybrid CNN + gated MLP encoder feeding both modalities to the latent.

    Input is the packed vector ``[rgb64_flat | flat_features]``. The RGB slice is
    reshaped to ``(B, 3, 64, 64)`` and run through ``ConvEncoder``; the feature
    slice is run through an MLP branch multiplied by a learnable scalar gate
    initialized at zero.
    """

    cnn_depth: int = 16
    cnn_kernel: int = 5
    cnn_mults: tuple[int, ...] = (2, 3, 4, 4)
    vggt_embed_dim: int = 1024
    mlp_hidden: int = 1024
    mlp_layers: int = 2
    rgb_dim: int = HYBRID_RGB_DIM
    vggt_dim: int = HYBRID_VGGT_DIM

    def _branches(
        self, obs: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(cnn_embed, gated_feature_embed, gate)`` from packed obs."""
        if obs.ndim != 2 or obs.shape[-1] != self.rgb_dim + self.vggt_dim:
            raise ValueError(
                f"expected (B, {self.rgb_dim + self.vggt_dim}) hybrid features, "
                f"got {obs.shape}"
            )
        rgb = obs[..., : self.rgb_dim].reshape(obs.shape[0], 3, 64, 64)
        features = obs[..., self.rgb_dim :]
        cnn_embed = make_rgb_conv_encoder(
            depth=self.cnn_depth,
            kernel_size=self.cnn_kernel,
            mults=self.cnn_mults,
            name="cnn",
        )(rgb)
        feature_mlp = R2MLP(
            hidden=self.mlp_hidden,
            layers=self.mlp_layers,
            out_dim=self.vggt_embed_dim,
            name="vggt_mlp",
        )
        gate = self.param("gate", nn.initializers.zeros, ())
        feature_embed = gate * feature_mlp(features)
        return cnn_embed, feature_embed, gate

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode packed hybrid observations into one concatenated embedding."""
        cnn_embed, feature_embed, _ = self._branches(obs)
        return jnp.concatenate([cnn_embed, feature_embed], axis=-1)

    @nn.compact
    def branches(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Diagnostic split: ``(cnn_embed, gated_feature_embed, gate_scalar)``."""
        return self._branches(obs)
