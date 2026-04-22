"""Attention with optional QK-norm, 2D RoPE, KV-cache + eviction (Step 6b).

Matches ``streamvggt.layers.attention.Attention``:
QKV-packed Linear, optional per-head LayerNorm on Q and K, optional 2D RoPE
on Q and K, manual softmax in fp32, output projection.

Cache semantics:
* ``past_kv`` = ``(past_k, past_v)`` from the previous call (already
  RoPE-applied). Concatenated with the current frame's new K/V before
  attention.
* If ``cache_budget`` is given and the concatenated cache exceeds it, the
  first ``num_anchor_tokens`` entries are preserved as anchors and the
  remaining candidates are pruned down by cosine-similarity-to-mean
  scoring (least-similar are retained — "diverse" tokens beat "redundant"
  ones). Dynamic per-block budget allocation lands in Step 6c.
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

# torch.nn.functional.normalize default eps.
_L2_NORMALIZE_EPS = 1e-12


def _evict_kv(
    k: jnp.ndarray,
    v: jnp.ndarray,
    cache_budget: int,
    num_anchor_tokens: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Port of ``streamvggt.layers.attention.Attention.eviction``.

    If ``k.shape[2] <= cache_budget``: no-op, returns ``(k, v, None)``.

    Otherwise: keeps the first ``num_anchor_tokens`` as anchors and prunes
    the remaining candidates to ``cache_budget - num_anchor_tokens`` by
    cosine-similarity-to-mean scoring. The tokens with the LOWEST similarity
    to the candidate mean are retained (they carry the most unique
    information); high-similarity tokens are redundant and evicted.

    Returns the updated ``(k, v)`` plus a scalar ``avg_scores`` used by the
    dynamic-budget allocator (Step 6c).
    """
    B, H, N, D = k.shape
    if N <= cache_budget:
        return k, v, None

    n_anchor = num_anchor_tokens
    n_keep = cache_budget - n_anchor
    if n_keep <= 0:
        raise ValueError(
            f"cache_budget ({cache_budget}) must exceed num_anchor_tokens "
            f"({num_anchor_tokens})"
        )

    anchor_k, cand_k = k[:, :, :n_anchor], k[:, :, n_anchor:]
    anchor_v, cand_v = v[:, :, :n_anchor], v[:, :, n_anchor:]

    # L2-normalize candidates (matches torch.nn.functional.normalize dim=-1).
    norm = jnp.linalg.norm(cand_k, axis=-1, keepdims=True)
    cand_k_norm = cand_k / jnp.maximum(norm, _L2_NORMALIZE_EPS)

    # Score = cos-sim to mean candidate direction.
    mean_vec = jnp.mean(cand_k_norm, axis=2, keepdims=True)  # (B, H, 1, D)
    scores = jnp.sum(cand_k_norm * mean_vec, axis=-1)  # (B, H, n_cand)
    avg_scores = jnp.mean(scores)

    # Keep lowest-similarity (most diverse) candidates. top_k on -scores.
    _, top_idx = jax.lax.top_k(-scores, n_keep)  # (B, H, n_keep)
    # PyTorch sorts indices ascending to keep temporal order stable.
    top_idx = jnp.sort(top_idx, axis=-1)

    gather_idx = jnp.broadcast_to(top_idx[..., None], (B, H, n_keep, D))
    kept_cand_k = jnp.take_along_axis(cand_k, gather_idx, axis=2)
    kept_cand_v = jnp.take_along_axis(cand_v, gather_idx, axis=2)

    final_k = jnp.concatenate([anchor_k, kept_cand_k], axis=2)
    final_v = jnp.concatenate([anchor_v, kept_cand_v], axis=2)
    return final_k, final_v, avg_scores


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
        past_kv: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        use_cache: bool = False,
        cache_budget: int | None = None,
        num_anchor_tokens: int = 0,
    ):
        """Forward pass.

        Args:
            x: (B, N, dim) input tokens (the *new* frame's tokens when caching).
            rope_tables: Optional (cos, sin) pair for 2D RoPE.
            positions: (B, N, 2) integer positions; required iff rope_tables given.
                In cache mode, these are the positions of the new frame only;
                past_kv already has RoPE applied from prior calls.
            attn_mask: Optional additive attention mask broadcastable to
                (B, H, N, K) where K = past_len + N.
            past_kv: Optional (past_k, past_v), both shape (B, H, past_len, Dh),
                returned by the previous call. Prepended to the new K/V.
            use_cache: If True, return ``(output, (full_k, full_v))`` — or
                ``(output, (full_k, full_v), avg_scores_or_None)`` if
                ``cache_budget`` is provided.
            cache_budget: Optional int. If set and the concatenated cache
                exceeds this, low-diversity candidates are evicted down to
                ``cache_budget`` tokens.
            num_anchor_tokens: Number of leading cache entries that are
                never evicted (typically the first frame's token count).

        Returns:
            Tensor, or tuple depending on flags (see ``use_cache``).
        """
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

        # Prepend cached past K/V (already RoPE-applied) to the new K/V.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = jnp.concatenate([past_k, k], axis=2)
            v = jnp.concatenate([past_v, v], axis=2)

        # Optional eviction: prune the cache back to cache_budget tokens.
        # ``evict_score`` is None when no eviction fired; otherwise a scalar
        # cosine-similarity mean consumed by the dynamic-budget allocator.
        # Named distinctly from the attention-score variable below — they are
        # different quantities and shadowing caused a real bug.
        evict_score: jnp.ndarray | None = None
        if use_cache and cache_budget is not None and k.shape[2] > cache_budget:
            k, v, evict_score = _evict_kv(k, v, cache_budget, num_anchor_tokens)

        # Manual attention with explicit fp32 softmax so bf16 inputs behave
        # like PyTorch's F.scaled_dot_product_attention (which internally
        # computes softmax in fp32).
        orig_dtype = v.dtype
        scale = jnp.asarray(1.0 / jnp.sqrt(head_dim), dtype=self.softmax_dtype)
        q_hi = q.astype(self.softmax_dtype) * scale
        k_hi = k.astype(self.softmax_dtype)
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q_hi, k_hi)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask.astype(self.softmax_dtype)
        probs = jax.nn.softmax(attn_scores, axis=-1).astype(orig_dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)  # (B, H, N, Dh)

        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, N, self.dim)
        out = nn.Dense(self.dim, use_bias=self.proj_bias, name="proj")(out)
        if use_cache:
            if cache_budget is not None:
                return out, (k, v), evict_score
            return out, (k, v)
        return out
