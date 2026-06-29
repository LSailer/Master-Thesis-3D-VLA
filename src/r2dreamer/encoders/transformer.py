"""Token Transformer encoder modules."""

from collections.abc import Mapping
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from src.r2dreamer.encoders.cnn import make_rgb_conv_encoder
from src.r2dreamer.encoders.constants import AGG_REGISTER_TOKENS, AGG_TOKEN_TOKENS
from src.r2dreamer.world_model.rssm import RMSNorm

EncoderObs = jax.Array | Mapping[str, jax.Array]
NormKind = Literal["rms", "layer"]
ReadoutKind = Literal["mean", "camera_patch", "camera_register_patch"]
ActivationKind = Literal["gelu", "silu"]


def _make_norm(kind: NormKind, compute_dtype: DTypeLike, name: str) -> nn.Module:
    """Create the requested normalization module for a Transformer block."""
    if kind == "rms":
        return RMSNorm(name=name)
    if kind == "layer":
        return nn.LayerNorm(dtype=compute_dtype, name=name)
    raise ValueError(f"unknown norm kind {kind!r}")


def _activate(x: jax.Array, activation: ActivationKind) -> jax.Array:
    """Apply a named Transformer MLP activation."""
    if activation == "gelu":
        return nn.gelu(x)
    if activation == "silu":
        return nn.silu(x)
    raise ValueError(f"unknown activation {activation!r}")


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block for a token sequence.

    Parameters:
        model_dim: Token width inside attention and the residual stream.
        heads: Number of self-attention heads. ``model_dim`` must be divisible
            by this value.
        mlp_ratio: Expansion ratio for the feed-forward hidden layer.
        dropout: Dropout rate used in attention and feed-forward blocks.
        norm_kind: ``"rms"`` or ``"layer"``.
        activation: Feed-forward nonlinearity.
        compute_dtype: JAX/Flax compute dtype for attention and dense layers.

    Returns:
        A token array with the same shape as the input.
    """

    model_dim: int
    heads: int
    mlp_ratio: int = 2
    dropout: float = 0.0
    norm_kind: NormKind = "layer"
    activation: ActivationKind = "gelu"
    compute_dtype: DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool = False) -> jax.Array:
        """Run one residual attention block followed by one residual MLP block."""
        attn_in = _make_norm(self.norm_kind, self.compute_dtype, "attn_norm")(x)
        attn = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=self.dropout,
            use_bias=False,
            dtype=self.compute_dtype,
            name="attn",
        )(attn_in, deterministic=not train)
        x = x + attn

        mlp_in = _make_norm(self.norm_kind, self.compute_dtype, "mlp_norm")(x)
        y = nn.Dense(
            self.model_dim * self.mlp_ratio,
            dtype=self.compute_dtype,
            name="mlp_in",
        )(mlp_in)
        y = _activate(y, self.activation)
        y = nn.Dropout(self.dropout, name="dropout")(y, deterministic=not train)
        y = nn.Dense(self.model_dim, dtype=self.compute_dtype, name="mlp_out")(y)
        y = nn.Dropout(self.dropout, name="out_dropout")(y, deterministic=not train)
        return x + y


class TokenTransformerEncoder(nn.Module):
    """Generic Transformer encoder for flat or structured token observations.

    Parameters:
        embed_dim: Width of the token readout returned by this module. If an RGB
            branch is enabled, the final ``__call__`` output is
            ``concat(cnn_embed, token_embed)``.
        token_dim: Width of each input token.
        num_tokens: Number of input tokens before optional register-token drop.
        model_dim: Internal Transformer width. ``None`` keeps ``token_dim`` and
            skips the input projection; an integer inserts ``Dense(model_dim)``
            before positional embeddings.
        layers: Number of Transformer blocks.
        heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion ratio inside each block.
        dropout: Dropout rate. Keep ``0.0`` for deterministic current behavior.
        readout: Token pooling strategy: mean over all tokens, camera+patch mean,
            or camera+register mean+patch mean.
        norm_kind: ``"rms"`` or ``"layer"`` normalization inside blocks and on
            the final readout.
        activation: Feed-forward activation, ``"silu"`` or ``"gelu"``.
        keep_register_tokens: Whether register tokens remain in the sequence.
        register_tokens: Number of register tokens after the camera token.
        token_key: Dict key for token observations. If ``None``, ``obs`` itself
            is treated as the token tensor.
        image_key: Optional dict key for a CHW RGB image branch. If set,
            ``__call__`` concatenates ``ConvEncoder(image)`` with the token
            embedding and ``branches()`` returns both parts.
        singleton_tokens: If true, tokens must describe one global context and
            are broadcast to the RGB batch size after token encoding.
        compute_dtype: JAX/Flax compute dtype for Transformer math.
        cnn_depth: RGB CNN base depth when ``image_key`` is set.
        cnn_kernel: RGB CNN kernel size when ``image_key`` is set.
        cnn_mults: RGB CNN channel multipliers when ``image_key`` is set.

    Input shapes:
        Tokens may be ``(B, num_tokens, token_dim)``, ``(num_tokens, token_dim)``,
        or flattened ``(B, num_tokens * token_dim)``.

    Returns:
        Without ``image_key``: ``(B, embed_dim)`` token embedding, or
        ``(embed_dim,)`` for a single unbatched token sequence. With
        ``image_key``: ``(B, cnn_embed_dim + embed_dim)`` concatenated embedding.
    """

    embed_dim: int = 1024
    token_dim: int = 1024
    num_tokens: int = AGG_TOKEN_TOKENS
    model_dim: int | None = None
    layers: int = 2
    heads: int = 8
    mlp_ratio: int = 2
    dropout: float = 0.0
    readout: ReadoutKind = "mean"
    norm_kind: NormKind = "layer"
    activation: ActivationKind = "gelu"
    keep_register_tokens: bool = True
    register_tokens: int = AGG_REGISTER_TOKENS
    token_key: str | None = None
    image_key: str | None = None
    singleton_tokens: bool = False
    compute_dtype: DTypeLike = jnp.float32
    cnn_depth: int = 16
    cnn_kernel: int = 5
    cnn_mults: tuple[int, ...] = (2, 3, 4, 4)

    def _model_dim(self) -> int:
        """Return the residual token width after the optional input projection."""
        if self.model_dim is None:
            return self.token_dim
        return self.model_dim

    def _kept_tokens(self) -> int:
        """Return token count after optional register-token removal."""
        if self.keep_register_tokens:
            return self.num_tokens
        return self.num_tokens - self.register_tokens

    def _validate_config(self) -> None:
        """Fail early for inconsistent Transformer/readout settings."""
        if self.num_tokens <= 0 or self.token_dim <= 0 or self.embed_dim <= 0:
            raise ValueError("num_tokens, token_dim, and embed_dim must be positive")
        if self.layers < 0:
            raise ValueError(f"layers must be non-negative, got {self.layers}")
        if self.heads <= 0 or self._model_dim() % self.heads != 0:
            raise ValueError(
                f"model_dim={self._model_dim()} must be divisible by heads={self.heads}"
            )
        if self.register_tokens < 0 or 1 + self.register_tokens >= self.num_tokens:
            raise ValueError(
                "register_tokens must leave at least one camera token and one patch token"
            )
        if self.readout == "camera_register_patch" and not self.keep_register_tokens:
            raise ValueError("camera_register_patch readout requires kept register tokens")
        if self.readout not in ("mean", "camera_patch", "camera_register_patch"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.norm_kind not in ("rms", "layer"):
            raise ValueError(f"unknown norm kind {self.norm_kind!r}")
        if self.activation not in ("gelu", "silu"):
            raise ValueError(f"unknown activation {self.activation!r}")

    def _extract_image_and_tokens(
        self, obs: EncoderObs
    ) -> tuple[jax.Array | None, jax.Array]:
        """Resolve image and token tensors from either a dict or direct tensor input."""
        if self.token_key is None:
            if isinstance(obs, Mapping):
                raise TypeError("token_key=None expects obs to be the token tensor")
            return None, obs
        if not isinstance(obs, Mapping):
            raise TypeError("token_key requires obs to be a mapping")
        image = None if self.image_key is None else obs[self.image_key]
        return image, obs[self.token_key]

    def _reshape_tokens(self, tokens: jax.Array) -> tuple[jax.Array, bool]:
        """Convert supported token layouts to ``(B, num_tokens, token_dim)``."""
        tokens = jnp.asarray(tokens, dtype=self.compute_dtype)
        squeeze = False
        if tokens.ndim == 2 and tokens.shape == (self.num_tokens, self.token_dim):
            tokens = tokens[None]
            squeeze = True
        elif tokens.ndim == 2 and tokens.shape[-1] == self.num_tokens * self.token_dim:
            tokens = tokens.reshape(tokens.shape[0], self.num_tokens, self.token_dim)
        elif tokens.ndim != 3 or tokens.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError(
                "expected tokens with shape "
                f"(B, {self.num_tokens}, {self.token_dim}), "
                f"({self.num_tokens}, {self.token_dim}), or "
                f"(B, {self.num_tokens * self.token_dim}); got {tokens.shape}"
            )
        if self.singleton_tokens and tokens.shape[0] != 1:
            raise ValueError(
                f"singleton_tokens expects one token context, got batch {tokens.shape[0]}"
            )
        return tokens, squeeze

    def _drop_or_keep_registers(self, tokens: jax.Array) -> tuple[jax.Array, int]:
        """Apply the register-token policy and return the patch-token start index."""
        if self.keep_register_tokens:
            return tokens, 1 + self.register_tokens
        kept = jnp.concatenate(
            [tokens[:, :1], tokens[:, 1 + self.register_tokens :]], axis=1
        )
        return kept, 1

    def _readout(self, tokens: jax.Array, patch_start: int) -> jax.Array:
        """Pool the Transformer sequence according to ``self.readout``."""
        if self.readout == "mean":
            return tokens.mean(axis=1)

        camera = tokens[:, 0]
        patches = tokens[:, patch_start:].mean(axis=1)
        if self.readout == "camera_patch":
            return jnp.concatenate([camera, patches], axis=-1)

        registers = tokens[:, 1:patch_start].mean(axis=1)
        return jnp.concatenate([camera, registers, patches], axis=-1)

    def _encode_tokens(self, tokens: jax.Array, *, train: bool = False) -> jax.Array:
        """Encode only the token branch into ``(B, embed_dim)`` or ``(embed_dim,)``."""
        self._validate_config()
        tokens, squeeze = self._reshape_tokens(tokens)
        tokens, patch_start = self._drop_or_keep_registers(tokens)
        x = tokens
        if self.model_dim is not None:
            x = nn.Dense(self._model_dim(), dtype=self.compute_dtype, name="token_proj")(x)

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self._kept_tokens(), self._model_dim()),
        ).astype(self.compute_dtype)
        x = x + pos_embed

        for i in range(self.layers):
            x = _TransformerBlock(
                model_dim=self._model_dim(),
                heads=self.heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                norm_kind=self.norm_kind,
                activation=self.activation,
                compute_dtype=self.compute_dtype,
                name=f"block{i}",
            )(x, train=train)

        readout = self._readout(x, patch_start)
        readout = _make_norm(self.norm_kind, self.compute_dtype, "readout_norm")(readout)
        encoded = nn.Dense(self.embed_dim, dtype=self.compute_dtype, name="proj")(readout)
        if squeeze:
            return encoded[0]
        return encoded

    def _branches(self, obs: EncoderObs, *, train: bool = False) -> tuple[jax.Array, jax.Array]:
        """Return ``(cnn_embed, token_embed)`` for RGB+token configurations."""
        image, tokens = self._extract_image_and_tokens(obs)
        if image is None:
            raise ValueError("branches() requires image_key to be configured")
        cnn_embed = make_rgb_conv_encoder(
            depth=self.cnn_depth,
            kernel_size=self.cnn_kernel,
            mults=self.cnn_mults,
            name="rgb_cnn",
        )(image)
        token_embed = self._encode_tokens(tokens, train=train)
        if self.singleton_tokens:
            token_embed = jnp.broadcast_to(token_embed, (cnn_embed.shape[0], self.embed_dim))
        elif token_embed.shape[0] != cnn_embed.shape[0]:
            raise ValueError(
                "token batch must match image batch unless singleton_tokens=True: "
                f"token batch {token_embed.shape[0]}, image batch {cnn_embed.shape[0]}"
            )
        return cnn_embed, token_embed

    @nn.compact
    def __call__(self, obs: EncoderObs, *, train: bool = False) -> jax.Array:
        """Encode token-only or RGB+token observations."""
        image, tokens = self._extract_image_and_tokens(obs)
        if image is None:
            return self._encode_tokens(tokens, train=train)
        cnn_embed, token_embed = self._branches(obs, train=train)
        return jnp.concatenate([cnn_embed, token_embed], axis=-1)

    @nn.compact
    def branches(self, obs: EncoderObs, *, train: bool = False) -> tuple[jax.Array, jax.Array]:
        """Diagnostic split for RGB+token observations."""
        return self._branches(obs, train=train)
