"""Attention with optional QK-norm and RoPE — no-cache forward path.

Matches ``streamvggt.layers.attention.Attention`` in the no-cache branch:
QKV-packed Linear, optional per-head LayerNorm on Q and K, optional 2D RoPE
on Q and K, manual softmax in fp32, output projection.

The KV-cache path (eviction, anchor tokens, dynamic budgets) lands in
Step 6a+; for now ``use_cache=True`` raises NotImplementedError.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from modules.vggt.jax.rope import apply_rope_2d


# Per-head LayerNorm on Q and K uses PyTorch's default eps=1e-5.
# Outer block's norm1 / norm2 may use a different eps (1e-6 for DINOv2) and
# is configured on the Block, not here.
_QK_NORM_EPS = 1e-5


class Attention(nn.Module):
    """Multi-head attention without cache.

    Attributes:
        dim: Token feature dimension.
        num_heads: Number of attention heads. ``dim`` must be divisible by it.
        qk_norm: If True, apply LayerNorm to Q and K (per-head, on head_dim).
        qkv_bias: Whether QKV Linear has a bias term.
        proj_bias: Whether the output projection Linear has a bias term.
        softmax_dtype: Precision used for the attention-score softmax.
            Defaults to fp32 to match PyTorch's ``F.scaled_dot_product_attention``
            semantics even when inputs are bf16.
    """

    dim: int
    num_heads: int
    qk_norm: bool = False
    qkv_bias: bool = True
    proj_bias: bool = True
    softmax_dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        rope_tables: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        positions: jnp.ndarray | None = None,
        attn_mask: jnp.ndarray | None = None,
        use_cache: bool = False,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (B, N, dim) input tokens.
            rope_tables: Optional (cos, sin) pair for 2D RoPE.
            positions: (B, N, 2) integer positions; required iff rope_tables given.
            attn_mask: Optional additive attention mask broadcastable to (B, H, N, N).
            use_cache: Must be False here; cache path lands in Step 6a.

        Returns:
            (B, N, dim) attended tokens.
        """
        if use_cache:
            raise NotImplementedError("KV-cache path lands in Step 6a of the plan.")
        if (rope_tables is None) != (positions is None):
            raise ValueError("rope_tables and positions must be given together.")

        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"input dim {C} != attention dim {self.dim}")
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim {self.dim} not divisible by num_heads {self.num_heads}"
            )
        head_dim = self.dim // self.num_heads

        qkv = nn.Dense(3 * self.dim, use_bias=self.qkv_bias, name="qkv")(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = nn.LayerNorm(epsilon=_QK_NORM_EPS, name="q_norm")(q)
            k = nn.LayerNorm(epsilon=_QK_NORM_EPS, name="k_norm")(k)

        if rope_tables is not None:
            cos_table, sin_table = rope_tables
            q = apply_rope_2d(q, positions, cos_table, sin_table)
            k = apply_rope_2d(k, positions, cos_table, sin_table)

        # Manual attention with explicit fp32 softmax so bf16 inputs behave
        # like PyTorch's F.scaled_dot_product_attention (which internally
        # computes softmax in fp32).
        orig_dtype = v.dtype
        scale = jnp.asarray(1.0 / jnp.sqrt(head_dim), dtype=self.softmax_dtype)
        q_hi = q.astype(self.softmax_dtype) * scale
        k_hi = k.astype(self.softmax_dtype)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q_hi, k_hi)
        if attn_mask is not None:
            scores = scores + attn_mask.astype(self.softmax_dtype)
        probs = jax.nn.softmax(scores, axis=-1).astype(orig_dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)  # (B, H, N, Dh)

        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, N, self.dim)
        out = nn.Dense(self.dim, use_bias=self.proj_bias, name="proj")(out)
        return out
