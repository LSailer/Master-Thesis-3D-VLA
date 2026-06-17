"""Observation encoders: convolutional (ConvEncoder) and VGGT-based variants.

Each encoder produces a flat embedding vector consumed by the RSSM posterior
head. The choice between them is set by `R2DreamerConfig.encoder_type`.
"""

from typing import Literal

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
AGG_TOKEN_TOKENS = 1374                # cam(1) + registers(4) + patches(1369)
AGG_TOKEN_DIM = AGG_TOKEN_TOKENS * 1024  # 1,406,976 — full flattened aggregator
FULL_TOKEN_DIM = AGG_TOKEN_TOKENS * 2048  # 2,813,952 — frame + global streams
HOUSE_CONTEXT_DIM = 1024
AGG_POOLED_DIM = 3 * 1024              # 3,072 — pooled [cam | mean | max]
AGG_REGISTER_TOKENS = 4


def _symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log compression, ``sign(x) * log1p(|x|)``.

    Dreamer's standard transform for unbounded inputs. Used by
    ``ConvEncoder(input_kind="world_points")`` to tame the metric XYZ range of
    full-resolution world points (the RGB encoder's ``obs - 0.5`` centering
    assumes [0, 1] and is meaningless here).
    """
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


class ConvEncoder(nn.Module):
    """Shared spatial convolutional encoder for RGB images or world-point maps.

    ``input_kind`` is explicit because RGB and metric XYZ world points can share
    the same ``(B, 3, H, W)`` shape but need different preprocessing: RGB uses
    Dreamer's ``obs - 0.5`` centering, world points use symlog. ``embed_dim`` is
    optional for historical RGB compatibility; set it for world-point variants
    to project the flattened conv map to a fixed embedding width.
    """
    depth: int = 16
    kernel_size: int = 5
    mults: tuple = (2, 3, 4, 4)
    input_kind: Literal["rgb", "world_points"] = "rgb"
    embed_dim: int | None = None

    @nn.compact
    def __call__(self, obs):
        if self.input_kind == "rgb":
            x = obs - 0.5
        elif self.input_kind == "world_points":
            x = _symlog(obs)
        else:
            raise ValueError(
                f"input_kind must be 'rgb' or 'world_points', got {self.input_kind!r}"
            )
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
        if self.embed_dim is not None:
            x = nn.Dense(self.embed_dim, name="proj")(x)
        return x


# Backward-compatible import target for old contract snapshots/checkpoints only.
# New code should use ConvEncoder(input_kind="world_points", embed_dim=...).
WPConvEncoder = ConvEncoder


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


class _TokenTransformerBlock(nn.Module):
    """Small pre-norm Transformer block for frozen VGGT token sequences."""

    hidden: int
    heads: int
    mlp_ratio: int = 2

    @nn.compact
    def __call__(self, x):
        attn_in = RMSNorm(name="attn_norm")(x)
        attn = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.hidden,
            out_features=self.hidden,
            use_bias=False,
            name="attn",
        )(attn_in)
        x = x + attn

        mlp_in = RMSNorm(name="mlp_norm")(x)
        y = nn.Dense(self.hidden * self.mlp_ratio, name="mlp_in")(mlp_in)
        y = nn.silu(y)
        y = nn.Dense(self.hidden, name="mlp_out")(y)
        return x + y


class VGGTAggTokenTransformerEncoder(nn.Module):
    """Trainable Transformer over frozen VGGT aggregator tokens (3D-75).

    Replay stores flattened float16 full-token features. This encoder upcasts to
    float32, restores ``(tokens, token_dim)``, optionally drops register tokens
    for future ablations, projects each token to a smaller attention width, and
    returns one ``embed_dim`` vector for the existing ``R2RSSM.observe()`` path.
    """

    embed_dim: int = 1024
    token_dim: int = 1024
    num_tokens: int = AGG_TOKEN_TOKENS
    projection_dim: int = 256
    layers: int = 2
    heads: int = 8
    mlp_ratio: int = 2
    keep_register_tokens: bool = True

    def _kept_tokens(self) -> int:
        if self.keep_register_tokens:
            return self.num_tokens
        return self.num_tokens - AGG_REGISTER_TOKENS

    @nn.compact
    def __call__(self, obs):
        expected_dim = self.num_tokens * self.token_dim
        if obs.ndim != 2 or obs.shape[-1] != expected_dim:
            raise ValueError(
                "VGGTAggTokenTransformerEncoder expects "
                f"(B, {expected_dim}) flattened VGGT aggregator tokens, got {obs.shape}"
            )
        if self.projection_dim % self.heads != 0:
            raise ValueError(
                f"projection_dim={self.projection_dim} must be divisible by heads={self.heads}"
            )

        tokens = obs.astype(jnp.float32).reshape(obs.shape[0], self.num_tokens, self.token_dim)
        if self.keep_register_tokens:
            x = tokens
            patch_start = 1 + AGG_REGISTER_TOKENS
        else:
            x = jnp.concatenate([tokens[:, :1], tokens[:, 1 + AGG_REGISTER_TOKENS:]], axis=1)
            patch_start = 1

        x = nn.Dense(self.projection_dim, name="token_proj")(x)
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self._kept_tokens(), self.projection_dim),
        )
        x = x + pos

        for i in range(self.layers):
            x = _TokenTransformerBlock(
                hidden=self.projection_dim,
                heads=self.heads,
                mlp_ratio=self.mlp_ratio,
                name=f"block{i}",
            )(x)

        cam = x[:, 0]
        patches = x[:, patch_start:].mean(axis=1)
        if self.keep_register_tokens:
            regs = x[:, 1:patch_start].mean(axis=1)
            readout = jnp.concatenate([cam, regs, patches], axis=-1)
        else:
            readout = jnp.concatenate([cam, patches], axis=-1)
        readout = RMSNorm(name="readout_norm")(readout)
        return nn.Dense(self.embed_dim, name="proj")(readout)


class _FullTokenTransformerBlock(nn.Module):
    """Pre-norm Transformer block that keeps the 2048-d VGGT token width."""

    token_dim: int
    heads: int
    mlp_ratio: int = 2
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, train: bool = False):
        attn_in = nn.LayerNorm(name="attn_norm")(x)
        attn = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.token_dim,
            out_features=self.token_dim,
            dropout_rate=self.dropout,
            deterministic=not train,
            use_bias=False,
            name="attn",
        )(attn_in)
        x = x + attn

        mlp_in = nn.LayerNorm(name="mlp_norm")(x)
        y = nn.Dense(self.token_dim * self.mlp_ratio, name="mlp_in")(mlp_in)
        y = nn.gelu(y)
        y = nn.Dropout(self.dropout, deterministic=not train, name="dropout")(y)
        y = nn.Dense(self.token_dim, name="mlp_out")(y)
        y = nn.Dropout(self.dropout, deterministic=not train, name="out_dropout")(y)
        return x + y


class VGGTFullTokenContextTransformer(nn.Module):
    """3D-77 full-token context encoder.

    Consumes frozen VGGT tokens directly at ``(1374, 2048)`` by default, keeps
    attention at ``d_model == token_dim``, and owns the final ``2048 -> 1024``
    context projection used by the RGB+VGGT hybrid gate. There is intentionally
    no pre-attention token projection.
    """

    context_dim: int = HOUSE_CONTEXT_DIM
    token_dim: int = 2048
    num_tokens: int = AGG_TOKEN_TOKENS
    layers: int = 2
    heads: int = 8
    mlp_ratio: int = 2
    dropout: float = 0.0

    @nn.compact
    def __call__(self, tokens, *, train: bool = False):
        squeeze = tokens.ndim == 2
        if squeeze:
            tokens = tokens[None]
        if tokens.ndim != 3 or tokens.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError(
                "VGGTFullTokenContextTransformer expects "
                f"(B, {self.num_tokens}, {self.token_dim}) or "
                f"({self.num_tokens}, {self.token_dim}) full VGGT tokens, got {tokens.shape}"
            )
        if self.token_dim % self.heads != 0:
            raise ValueError(
                f"token_dim={self.token_dim} must be divisible by heads={self.heads}"
            )

        x = tokens.astype(jnp.float32)
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_tokens, self.token_dim),
        )
        x = x + pos

        for i in range(self.layers):
            x = _FullTokenTransformerBlock(
                token_dim=self.token_dim,
                heads=self.heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                name=f"block{i}",
            )(x, train=train)

        context_2048 = nn.LayerNorm(name="context_norm")(x.mean(axis=1))
        context = nn.Dense(self.context_dim, name="context_proj")(context_2048)
        if squeeze:
            context = context[0]
        return context


class RGBFullTokenTransformerEncoder(nn.Module):
    """RGB + live full-token VGGT encoder without a learned fusion gate.

    Replay stores only RGB64. The adapter injects the latest live VGGT full
    aggregator tokens as a separate observation field at act/train time. This
    module keeps the full-token Transformer inside the agent params, so one-batch
    overfit verifies gradients through the learned token encoder itself.
    """

    cnn_depth: int = 16
    cnn_kernel: int = 5
    cnn_mults: tuple = (2, 3, 4, 4)
    context_dim: int = HOUSE_CONTEXT_DIM
    token_dim: int = 2048
    num_tokens: int = AGG_TOKEN_TOKENS
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_mlp_ratio: int = 2
    transformer_dropout: float = 0.0

    def setup(self):
        self.cnn = ConvEncoder(
            depth=self.cnn_depth, kernel_size=self.cnn_kernel, mults=self.cnn_mults
        )
        self.token_transformer = VGGTFullTokenContextTransformer(
            context_dim=self.context_dim,
            token_dim=self.token_dim,
            num_tokens=self.num_tokens,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            mlp_ratio=self.transformer_mlp_ratio,
            dropout=self.transformer_dropout,
        )

    def _branches(self, obs):
        image = obs["image"]
        tokens = obs["full_tokens"]
        cnn_e = self.cnn(image)
        token_e = self.token_transformer(tokens, train=False)
        return cnn_e, token_e

    def __call__(self, obs):
        cnn_e, token_e = self._branches(obs)
        return jnp.concatenate([cnn_e, token_e], axis=-1)

    def branches(self, obs):
        """Diagnostic split: (cnn_embed, token_transformer_embed)."""
        return self._branches(obs)





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
    decoder-free by default; when enabled, the loss detaches ``feat`` so this
    stays a visualisation probe rather than an agent objective.
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
