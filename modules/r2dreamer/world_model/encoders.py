"""Observation encoders: convolutional (ConvEncoder) and VGGT-based variants.

Each encoder produces a flat embedding vector consumed by the RSSM posterior
head. The choice between them is set by `R2DreamerConfig.encoder_type`.
"""

import jax.numpy as jnp
import flax.linen as nn

from .rssm import RMSNorm


class ConvEncoder(nn.Module):
    """Convolutional encoder ported from R2-Dreamer.

    Expects obs in CHW format (JAX codebase convention: B, C, H, W).
    Applies Conv+MaxPool+RMSNorm+SiLU for each channel multiplier, then flattens.
    """
    depth: int = 16
    kernel_size: int = 5
    mults: tuple = (2, 3, 4, 4)

    @nn.compact
    def __call__(self, obs):
        # obs: (B, C, H, W) float [0,1]
        x = obs - 0.5
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        for i, mult in enumerate(self.mults):
            ch = self.depth * mult
            x = nn.Conv(ch, (self.kernel_size, self.kernel_size),
                        padding="SAME", name=f"conv{i}")(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        # Transpose back to NCHW before flatten to match PyTorch's flatten order
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        return x.reshape(x.shape[0], -1)


class VGGTEncoder(nn.Module):
    """Linear projection encoder for VGGT features.

    Takes flattened world_points (37*37*3=4107) + camera_pose (9) = 4116 dim
    and projects to embed_dim via a single Dense layer.
    """
    embed_dim: int = 1024

    @nn.compact
    def __call__(self, obs):
        # obs: (B, 4116) float32 — already flat
        out = nn.Dense(self.embed_dim, name="proj")(obs)
        # === BP #4 (inside JIT — debug walkthrough — uncomment to re-enable) ===
        # import jax as _jax_dbg
        # _jax_dbg.debug.print(
        #     "[BP#4] encoder out: shape={s} mean={m} std={st} min={mn} max={mx}",
        #     s=out.shape, m=out.mean(), st=out.std(), mn=out.min(), mx=out.max(),
        # )
        return out


class VGGTAggregatorMLPEncoder(nn.Module):
    """Variant 1 encoder for VGGT aggregator features.

    The fast training path stores mean-pooled pre-head aggregator features in
    replay as ``(B, D)`` and projects them to ``embed_dim``.  The module still
    accepts the legacy ``(B, N, D)`` all-token shape for tests/debugging by
    applying the same Dense layer tokenwise and then mean-pooling tokens.
    """
    embed_dim: int = 1024
    channels: int = 64
    hidden: int = 1024

    @nn.compact
    def __call__(self, obs):
        # obs: (B, D) mean-pooled features in the training path, or legacy
        # (B, N, D) all-token features for isolated token-path checks.
        if obs.ndim == 2:
            return nn.Dense(self.embed_dim, name="proj")(obs)
        if obs.ndim == 3:
            tokens = nn.Dense(self.embed_dim, name="proj")(obs)
            return tokens.mean(axis=1)
        raise ValueError(f"expected (B, D) or (B, N, D) VGGT features, got {obs.shape}")
