"""Transformer block + its small helpers (LayerScale, MLP).

Matches ``streamvggt.layers.block.Block`` structure:
    norm1 -> attn -> ls1 -> residual
    norm2 -> mlp  -> ls2 -> residual

LayerScale (``ls1``, ``ls2``) multiplies the residual branch by a per-dim
learnable ``gamma`` initialised to ``init_values``. When ``init_values`` is
None or 0, the reference replaces LayerScale with Identity (no params);
callers express that by passing ``init_values=None``.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.vggt.jax.attention import Attention


# --------------------------------------------------------------------------- #
#  LayerScale
# --------------------------------------------------------------------------- #


class LayerScale(nn.Module):
    """Element-wise scaling with a learnable gamma initialised to init_values."""

    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gamma = self.param(
            "gamma",
            lambda _key, shape: jnp.full(shape, self.init_values, dtype=jnp.float32),
            (self.dim,),
        )
        return x * gamma.astype(x.dtype)


# --------------------------------------------------------------------------- #
#  MLP (two Linears + GELU)
# --------------------------------------------------------------------------- #


class Mlp(nn.Module):
    """Standard ViT MLP: fc1 -> GELU -> fc2. No dropout (inference-only)."""

    hidden_features: int
    out_features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_features, use_bias=self.use_bias, name="fc1")(x)
        # PyTorch nn.GELU() default is the exact (non-approximate) GELU.
        x = jax.nn.gelu(x, approximate=False)
        x = nn.Dense(self.out_features, use_bias=self.use_bias, name="fc2")(x)
        return x


# --------------------------------------------------------------------------- #
#  Block
# --------------------------------------------------------------------------- #


class Block(nn.Module):
    """Transformer block (pre-norm) with optional LayerScale and RoPE.

    Attributes:
        dim: Feature dimension.
        num_heads: Attention heads.
        mlp_ratio: Hidden-feature multiplier for the MLP.
        qk_norm: Forward-only flag for Attention.
        init_values: LayerScale init. None/0 disables LayerScale (Identity).
        norm_eps: LayerNorm epsilon for norm1 / norm2. DINOv2 uses 1e-6;
            aggregator and camera-trunk blocks use 1e-5.
        qkv_bias / proj_bias / ffn_bias: Bias flags passed through.
    """

    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qk_norm: bool = False
    init_values: float | None = 0.01
    norm_eps: float = 1e-5
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        rope_tables: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        positions: jnp.ndarray | None = None,
        attn_mask: jnp.ndarray | None = None,
        past_kv: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        use_cache: bool = False,
        cache_budget: int | None = None,
        num_anchor_tokens: int = 0,
    ):
        # Attention residual
        h = nn.LayerNorm(epsilon=self.norm_eps, name="norm1")(x)
        attn_out = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            qk_norm=self.qk_norm,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            name="attn",
        )(
            h,
            rope_tables=rope_tables,
            positions=positions,
            attn_mask=attn_mask,
            past_kv=past_kv,
            use_cache=use_cache,
            cache_budget=cache_budget,
            num_anchor_tokens=num_anchor_tokens,
        )
        scores = None
        if use_cache:
            if cache_budget is not None:
                h, new_kv, scores = attn_out
            else:
                h, new_kv = attn_out
        else:
            h = attn_out
        if self.init_values:
            h = LayerScale(self.dim, init_values=self.init_values, name="ls1")(h)
        x = x + h

        # MLP residual
        h = nn.LayerNorm(epsilon=self.norm_eps, name="norm2")(x)
        h = Mlp(
            hidden_features=int(self.dim * self.mlp_ratio),
            out_features=self.dim,
            use_bias=self.ffn_bias,
            name="mlp",
        )(h)
        if self.init_values:
            h = LayerScale(self.dim, init_values=self.init_values, name="ls2")(h)
        x = x + h

        if use_cache:
            if cache_budget is not None:
                return x, new_kv, scores
            return x, new_kv
        return x
