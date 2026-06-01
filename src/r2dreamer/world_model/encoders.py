"""Observation encoders: convolutional (ConvEncoder) and VGGT-based variants.

Each encoder produces a flat embedding vector consumed by the RSSM posterior
head. The choice between them is set by `R2DreamerConfig.encoder_type`.
"""

import jax.numpy as jnp
import flax.linen as nn

from .rssm import RMSNorm


def _symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log compression, ``sign(x) * log1p(|x|)``.

    Dreamer's standard transform for unbounded inputs. Used by ``WPConvEncoder``
    to tame the metric XYZ range of full-resolution world points (the RGB
    encoder's ``obs - 0.5`` centering assumes [0, 1] and is meaningless here).
    """
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


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


def _mlp_body(x, num_layers: int, hidden: int):
    """Stack ``num_layers`` Dreamer-style ``Dense -> RMSNorm -> SiLU`` blocks.

    Faithful port of ``external/r2dreamer/networks.py`` ``MLP`` (each block is
    ``Linear -> RMSNorm -> act``). ``num_layers=0`` is a no-op (returns ``x``
    unchanged), so a caller that then applies a single linear ``proj`` collapses
    to the original bare-linear projection. Must be called inside an
    ``nn.compact`` method so the submodule names are scoped to the caller.
    """
    for i in range(num_layers):
        x = nn.Dense(hidden, name=f"hidden{i}")(x)
        x = RMSNorm(name=f"norm{i}")(x)
        x = nn.silu(x)
    return x


class VGGTEncoder(nn.Module):
    """MLP encoder for flat VGGT WP/CP features (3D-52).

    Takes flattened world_points (37*37*3=4107) + camera_pose (9) = 4116 dim and
    maps to ``embed_dim`` through ``num_layers`` hidden ``Dense->RMSNorm->SiLU``
    blocks followed by a linear projection. ``num_layers`` defaults to 1 (one
    hidden block + projection); the experiment run sets it to 3 to match
    R2Dreamer's native ``encoder.mlp.layers``. ``num_layers=0`` reproduces the
    historical single-``Dense`` linear projection exactly.
    """
    embed_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 1

    @nn.compact
    def __call__(self, obs):
        # obs: (B, 4116) float32 — already flat
        x = _mlp_body(obs, self.num_layers, self.hidden)
        return nn.Dense(self.embed_dim, name="proj")(x)


class VGGTAggregatorMLPEncoder(nn.Module):
    """MLP encoder for the adapter's pre-pooled VGGT aggregator features.

    Input layout is ``[cam | mean_patches | max_patches]`` flat, i.e. shape
    ``(B, 3 * pool_dim)``. The camera-token slice carries the same pose-aware
    embedding ``camera_head`` reads (see ``aggregator.py``), the mean is a
    smooth global summary, and the max picks out salient patches; each is
    normalised separately so the mean/max scale mismatch does not bleed into
    the projection. The normalised slices are concatenated and passed through
    ``num_layers`` hidden ``Dense->RMSNorm->SiLU`` blocks + a linear readout
    (default depth 1; the experiment run uses 3 — see 3D-52).
    """
    embed_dim: int = 1024
    pool_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 1

    @nn.compact
    def __call__(self, obs):
        if obs.ndim != 2 or obs.shape[-1] != 3 * self.pool_dim:
            raise ValueError(
                f"expected (B, {3 * self.pool_dim}) VGGT pooled features, got {obs.shape}"
            )
        cam, mean_p, max_p = jnp.split(obs, 3, axis=-1)
        cam = RMSNorm(name="norm_cam")(cam)
        mean_p = RMSNorm(name="norm_mean")(mean_p)
        max_p = RMSNorm(name="norm_max")(max_p)
        x = jnp.concatenate([cam, mean_p, max_p], axis=-1)
        x = _mlp_body(x, self.num_layers, self.hidden)
        return nn.Dense(self.embed_dim, name="proj")(x)


class WPConvEncoder(nn.Module):
    """Conv encoder over full-resolution VGGT world-point maps (3D-53).

    A dense world-point map has shape ``(B, 3, H, W)`` — the same (C, H, W)
    layout as an RGB frame, but the three channels are *metric XYZ* coordinates
    rather than [0, 1] colour. We therefore reuse the RGB ``ConvEncoder``'s
    Conv+MaxPool+RMSNorm+SiLU stack but replace its ``obs - 0.5`` centering with
    ``symlog`` (Dreamer's transform for unbounded inputs), then flatten and
    project to ``embed_dim`` so the embedding width is comparable to the
    WP/CP and aggregator variants.

    At the default 518x518 input the four 2x2 max-pools take 518 -> 259 -> 129
    -> 64 -> 32, so a 32x32x(depth*mults[-1]) feature map is flattened before
    the linear readout — the spatial structure that the 37x37 grid is too small
    to preserve survives here.
    """
    embed_dim: int = 1024
    depth: int = 16
    kernel_size: int = 5
    mults: tuple = (2, 3, 4, 4)

    @nn.compact
    def __call__(self, obs):
        # obs: (B, 3, H, W) metric XYZ world points
        x = _symlog(obs)
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
        x = x.reshape(x.shape[0], -1)
        return nn.Dense(self.embed_dim, name="proj")(x)
