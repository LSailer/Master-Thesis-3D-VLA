"""Observation encoders: convolutional (ConvEncoder) and VGGT-based variants.

Each encoder produces a flat embedding vector consumed by the RSSM posterior
head. The choice between them is set by `R2DreamerConfig.encoder_type`.
"""

import jax.numpy as jnp
import flax.linen as nn

from .heads import R2MLP
from .rssm import RMSNorm

# Hybrid encoder input layout. Replay stores the modalities under explicit
# fields (see HybridObsAdapter); src.r2dreamer.obs_batch packs them into this
# flat tensor right before the Flax encoder boundary.
HYBRID_RGB_DIM = 3 * 64 * 64  # 12288 — the CNN branch's 64x64 RGB, flattened
HYBRID_VGGT_DIM = 4116        # WP/CP width: world_points 37*37*3 + camera_pose 9

# Aggregator readouts (3D-50 follow-up). Defined here (small ints) rather than
# imported from adapters.vggt_adapter so importing the world-model encoders stays
# free of the heavy VGGT extractor dependency; tests/test_agg_raw_dims.py asserts
# these agree with the adapter's constants and the live extractor shape.
AGG_RAW_TOKENS = 1370                  # cam(1) + patches(1369); 4 register tokens dropped
AGG_RAW_DIM = AGG_RAW_TOKENS * 1024    # 1,402,880 — raw flattened aggregator
AGG_POOLED_DIM = 3 * 1024              # 3,072 — pooled [cam | mean | max]


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


class VGGTAggRawMLPEncoder(nn.Module):
    """3-layer MLP over the RAW flattened aggregator tokens (V1 standalone).

    Input ``(B, AGG_RAW_DIM)`` = camera token + 1369 patch tokens x 1024-d, the 4
    register tokens dropped (see ``flatten_raw_aggregator``). Layer-1 is the
    expensive ~1.44B-param projection (1,402,880 -> ``hidden``); layers 2-3 are
    cheap 1024->1024 blocks. Replay stores float16; the cast to float32 here lets
    the dense matmuls run in the agent's working precision. No per-slice RMSNorm
    (unlike the pooled readout): raw tokens have no cam/mean/max slice structure.
    """
    embed_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 3

    @nn.compact
    def __call__(self, obs):
        if obs.ndim != 2 or obs.shape[-1] != AGG_RAW_DIM:
            raise ValueError(
                f"VGGTAggRawMLPEncoder expects (B, {AGG_RAW_DIM}), got {obs.shape}"
            )
        x = obs.astype(jnp.float32)
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


class HybridEncoder(nn.Module):
    """Hybrid encoder feeding the latent BOTH modalities at once (3D-50/51/52).

    Input is the packed WP/CP hybrid encoder vector ``[ rgb (HYBRID_RGB_DIM) |
    wp_cp (HYBRID_VGGT_DIM) ]``. Replay stores the fields separately as
    ``{"image": uint8 RGB64, "wp_cp": float32}``, and obs_batch packs them
    before this Flax module runs. The RGB slice is reshaped to ``(B, 3, 64, 64)``
    and run through the standard ``ConvEncoder``; the WP/CP slice is run through
    an MLP (the Linear->MLP upgrade of 3D-52). The two branch embeddings are
    concatenated to form the ``embed`` that conditions the RSSM posterior.

    The VGGT branch is multiplied by a single learnable scalar ``gate``
    initialised to **zero** (Flamingo/ResNet zero-gamma style): at init the
    branch contributes nothing, so the model behaves exactly like a CNN-Dreamer
    and the VGGT pathway "opens" only as training finds it useful. Crucially the
    MLP itself is *normally* initialised — zeroing both the gate and the MLP
    output would strand the gate at zero (``dloss/dgate`` would also be 0). The
    gate value is logged as the headline "how much WP/CP the latent uses" signal.

    ``setup()`` (not ``@nn.compact``) defines the submodules and the gate once so
    that ``__call__`` and the diagnostic ``branches`` method share identical
    parameters.
    """
    cnn_depth: int = 16
    cnn_kernel: int = 5
    cnn_mults: tuple = (2, 3, 4, 4)
    vggt_embed_dim: int = 1024
    mlp_hidden: int = 1024
    mlp_layers: int = 2
    rgb_dim: int = HYBRID_RGB_DIM
    vggt_dim: int = HYBRID_VGGT_DIM

    def setup(self):
        self.cnn = ConvEncoder(
            depth=self.cnn_depth, kernel_size=self.cnn_kernel, mults=self.cnn_mults
        )
        # Standard-init MLP (outscale=1.0); the zero-init lives in `gate`, not here.
        self.vggt_mlp = R2MLP(
            hidden=self.mlp_hidden, layers=self.mlp_layers, out_dim=self.vggt_embed_dim
        )
        self.gate = self.param("gate", nn.initializers.zeros, ())

    def _branches(self, obs):
        if obs.ndim != 2 or obs.shape[-1] != self.rgb_dim + self.vggt_dim:
            raise ValueError(
                f"expected (B, {self.rgb_dim + self.vggt_dim}) hybrid features, "
                f"got {obs.shape}"
            )
        rgb = obs[..., : self.rgb_dim].reshape(obs.shape[0], 3, 64, 64)
        wp_cp = obs[..., self.rgb_dim :]
        cnn_e = self.cnn(rgb)
        vggt_e = self.gate * self.vggt_mlp(wp_cp)
        return cnn_e, vggt_e

    def __call__(self, obs):
        cnn_e, vggt_e = self._branches(obs)
        return jnp.concatenate([cnn_e, vggt_e], axis=-1)

    def branches(self, obs):
        """Diagnostic split: (cnn_embed, gated_vggt_embed, gate_scalar).

        Shares params with ``__call__`` (both go through ``_branches``). A
        standalone accessor used by tests; the training loss derives the same
        per-branch contribution metrics more cheaply by slicing the
        already-computed fused ``embed`` rather than calling this.
        """
        cnn_e, vggt_e = self._branches(obs)
        return cnn_e, vggt_e, self.gate


class HybridAggPooledModule(nn.Module):
    """CNN(RGB 64) + zero-init-gated aggregator-pooled MLP, fused (V2).

    Same structure as ``HybridEncoder`` but the gated VGGT branch is the pooled
    aggregator readout (``VGGTAggregatorMLPEncoder``: per-slice RMSNorm on
    [cam | mean | max] then an MLP) over the 3072-d vector, instead of WP/CP. The
    zero-init scalar ``gate`` means training starts as plain CNN-Dreamer and the
    aggregator branch opens only as it proves useful; the per-branch hybrid/*
    metrics work unchanged (the VGGT branch still projects to ``vggt_embed_dim``
    and the module exposes a ``gate`` param via ``setup``).
    """
    cnn_depth: int = 16
    cnn_kernel: int = 5
    cnn_mults: tuple = (2, 3, 4, 4)
    vggt_embed_dim: int = 1024
    mlp_hidden: int = 1024
    mlp_layers: int = 3
    rgb_dim: int = HYBRID_RGB_DIM
    vggt_dim: int = AGG_POOLED_DIM

    def setup(self):
        self.cnn = ConvEncoder(
            depth=self.cnn_depth, kernel_size=self.cnn_kernel, mults=self.cnn_mults
        )
        # Reuse the standalone pooled-aggregator readout so the hybrid branch is a
        # faithful gated copy (per-slice RMSNorm preserved), not a degraded MLP.
        self.vggt_mlp = VGGTAggregatorMLPEncoder(
            embed_dim=self.vggt_embed_dim,
            pool_dim=self.vggt_dim // 3,
            hidden=self.mlp_hidden,
            num_layers=self.mlp_layers,
        )
        self.gate = self.param("gate", nn.initializers.zeros, ())

    def _branches(self, obs):
        if obs.ndim != 2 or obs.shape[-1] != self.rgb_dim + self.vggt_dim:
            raise ValueError(
                f"expected (B, {self.rgb_dim + self.vggt_dim}) hybrid-agg-pooled "
                f"features, got {obs.shape}"
            )
        rgb = obs[..., : self.rgb_dim].astype(jnp.float32).reshape(obs.shape[0], 3, 64, 64)
        agg = obs[..., self.rgb_dim :]
        cnn_e = self.cnn(rgb)
        vggt_e = self.gate * self.vggt_mlp(agg)
        return cnn_e, vggt_e

    def __call__(self, obs):
        cnn_e, vggt_e = self._branches(obs)
        return jnp.concatenate([cnn_e, vggt_e], axis=-1)

    def branches(self, obs):
        cnn_e, vggt_e = self._branches(obs)
        return cnn_e, vggt_e, self.gate


class HybridAggRawModule(nn.Module):
    """CNN(RGB 64) + zero-init-gated raw-aggregator 3-layer MLP, fused (V3).

    Same gate pattern as ``HybridEncoder``; the gated VGGT branch is
    ``VGGTAggRawMLPEncoder`` over the raw 1,402,880-d flattened aggregator (4
    register tokens dropped). The RGB slice is stored float16 in replay, so it is
    upcast to float32 before the CNN; the raw slice is upcast inside the MLP. The
    expensive ~1.44B-param layer-1 lives in this branch.
    """
    cnn_depth: int = 16
    cnn_kernel: int = 5
    cnn_mults: tuple = (2, 3, 4, 4)
    vggt_embed_dim: int = 1024
    mlp_hidden: int = 1024
    mlp_layers: int = 3
    rgb_dim: int = HYBRID_RGB_DIM
    vggt_dim: int = AGG_RAW_DIM

    def setup(self):
        self.cnn = ConvEncoder(
            depth=self.cnn_depth, kernel_size=self.cnn_kernel, mults=self.cnn_mults
        )
        self.vggt_mlp = VGGTAggRawMLPEncoder(
            embed_dim=self.vggt_embed_dim, hidden=self.mlp_hidden, num_layers=self.mlp_layers
        )
        self.gate = self.param("gate", nn.initializers.zeros, ())

    def _branches(self, obs):
        if obs.ndim != 2 or obs.shape[-1] != self.rgb_dim + self.vggt_dim:
            raise ValueError(
                f"expected (B, {self.rgb_dim + self.vggt_dim}) hybrid-agg-raw "
                f"features, got {obs.shape}"
            )
        rgb = obs[..., : self.rgb_dim].astype(jnp.float32).reshape(obs.shape[0], 3, 64, 64)
        raw = obs[..., self.rgb_dim :]
        cnn_e = self.cnn(rgb)
        vggt_e = self.gate * self.vggt_mlp(raw)
        return cnn_e, vggt_e

    def __call__(self, obs):
        cnn_e, vggt_e = self._branches(obs)
        return jnp.concatenate([cnn_e, vggt_e], axis=-1)

    def branches(self, obs):
        cnn_e, vggt_e = self._branches(obs)
        return cnn_e, vggt_e, self.gate


class ConvDecoder(nn.Module):
    """Transpose-conv image decoder for visual verification (3D-51).

    Maps the RSSM feature ``feat`` (B, F) back to an RGB image ``(B, 3, 64, 64)``
    in [0, 1]. Mirrors ``ConvEncoder``: a Dense lifts ``feat`` to a 4x4 grid,
    then four stride-2 ``ConvTranspose`` stages (4->8->16->32->64) with
    RMSNorm+SiLU, and a final 1-conv to 3 channels + sigmoid. R2-Dreamer is
    decoder-free by default; this is only built when ``cfg.decoder`` is set, so
    existing CNN/VGGT runs are unaffected.
    """
    depth: int = 16
    kernel_size: int = 5
    mults: tuple = (2, 3, 4, 4)
    out_channels: int = 3
    base_res: int = 4  # 4x4 -> (x2)^4 -> 64x64

    @nn.compact
    def __call__(self, feat):
        b = feat.shape[0]
        ch0 = self.depth * self.mults[-1]
        x = nn.Dense(self.base_res * self.base_res * ch0, name="in")(feat)
        x = x.reshape(b, self.base_res, self.base_res, ch0)  # NHWC
        for i, mult in enumerate(reversed(self.mults)):
            ch = self.depth * mult
            x = nn.ConvTranspose(
                ch, (self.kernel_size, self.kernel_size), strides=(2, 2),
                padding="SAME", name=f"deconv{i}",
            )(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        x = nn.Conv(self.out_channels, (self.kernel_size, self.kernel_size),
                    padding="SAME", name="out")(x)
        x = nn.sigmoid(x)
        return jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW, matches RGB layout
