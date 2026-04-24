"""Attention with optional QK-norm, 2D RoPE, KV-cache + eviction.

Matches ``streamvggt.layers.attention.Attention``:
QKV-packed Linear, optional per-head LayerNorm on Q and K, optional 2D RoPE
on Q and K, manual softmax in fp32, output projection.

Cache semantics:
* Legacy 2-tuple ``past_kv = (past_k, past_v)``: variable-length concat +
  compact eviction, manual einsum+softmax. Kept for no-cache parity tests.
* Padded 3-tuple ``past_kv = (k_pad, v_pad, valid_len)``:
  ``k_pad, v_pad`` shape ``(B, H, MAX, Dh)`` fixed, ``valid_len`` int32 scalar.
  Eviction writes back with ``dynamic_update_slice_in_dim``; attention uses
  ``jax.nn.dot_product_attention`` with ``-inf`` bias over padded slots.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from modules.vggt.jax.rope import apply_rope_2d


_QK_NORM_EPS = 1e-5
_L2_NORMALIZE_EPS = 1e-12


def _evict_kv(
    k: jnp.ndarray,
    v: jnp.ndarray,
    cache_budget: int,
    num_anchor_tokens: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Port of ``streamvggt.layers.attention.Attention.eviction`` (legacy path).

    If ``k.shape[2] <= cache_budget``: no-op, returns ``(k, v, None)``.

    Otherwise: keeps the first ``num_anchor_tokens`` as anchors and prunes
    the remaining candidates to ``cache_budget - num_anchor_tokens`` by
    cosine-similarity-to-mean scoring. Lowest-similarity tokens are retained.
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

    norm = jnp.linalg.norm(cand_k, axis=-1, keepdims=True)
    cand_k_norm = cand_k / jnp.maximum(norm, _L2_NORMALIZE_EPS)

    mean_vec = jnp.mean(cand_k_norm, axis=2, keepdims=True)
    scores = jnp.sum(cand_k_norm * mean_vec, axis=-1)
    avg_scores = jnp.mean(scores)

    _, top_idx = jax.lax.top_k(-scores, n_keep)
    top_idx = jnp.sort(top_idx, axis=-1)

    gather_idx = jnp.broadcast_to(top_idx[..., None], (B, H, n_keep, D))
    kept_cand_k = jnp.take_along_axis(cand_k, gather_idx, axis=2)
    kept_cand_v = jnp.take_along_axis(cand_v, gather_idx, axis=2)

    final_k = jnp.concatenate([anchor_k, kept_cand_k], axis=2)
    final_v = jnp.concatenate([anchor_v, kept_cand_v], axis=2)
    return final_k, final_v, avg_scores


def _padded_evict(
    k_pad: jnp.ndarray,
    v_pad: jnp.ndarray,
    valid_len: jnp.ndarray,
    cache_budget: int,
    num_anchor_tokens: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evict inside the padded cache. Returns (k_pad, v_pad, new_valid_len, score).

    ``score`` is the fp32 mean of cosine similarities over valid candidates
    (matches the legacy ``_evict_kv`` semantics, sans the batch/head reduction
    behaviour: both implementations take ``jnp.mean`` over all elements).
    """
    B, H, MAX, Dh = k_pad.shape
    n_anchor = num_anchor_tokens
    n_keep = cache_budget - n_anchor
    if n_keep <= 0:
        raise ValueError(
            f"cache_budget ({cache_budget}) must exceed num_anchor_tokens "
            f"({num_anchor_tokens})"
        )

    anchor_k = k_pad[:, :, :n_anchor]
    anchor_v = v_pad[:, :, :n_anchor]
    cand_k = k_pad[:, :, n_anchor:]
    cand_v = v_pad[:, :, n_anchor:]
    n_cand_max = MAX - n_anchor

    slot_idx = jnp.arange(n_cand_max, dtype=jnp.int32)
    cand_valid = slot_idx < (valid_len - n_anchor)  # (n_cand_max,)

    norm = jnp.linalg.norm(cand_k, axis=-1, keepdims=True)
    cand_k_norm = cand_k / jnp.maximum(norm, _L2_NORMALIZE_EPS)
    valid_f = cand_valid.astype(cand_k_norm.dtype)
    # Mean vector restricted to valid slots.
    valid_count = jnp.maximum(jnp.sum(valid_f), 1.0)
    mean_vec = (
        jnp.sum(cand_k_norm * valid_f[None, None, :, None], axis=2, keepdims=True)
        / valid_count
    )
    scores = jnp.sum(cand_k_norm * mean_vec, axis=-1)
    # Match legacy jnp.mean(scores) semantics: mean over valid candidates only
    # (reference code does mean over the full candidate region, which for a
    # compact 2-tuple is the valid region — here the valid region equals what
    # that would be in the compact form).
    score_sum = jnp.sum(scores * valid_f[None, None, :])
    total_count = valid_count * B * H
    evict_score = (score_sum / jnp.maximum(total_count, 1.0)).astype(jnp.float32)

    # Invalid slots get +inf so top_k(-scores) avoids them.
    scores_masked = jnp.where(cand_valid[None, None, :], scores, jnp.inf)

    _, top_idx = jax.lax.top_k(-scores_masked, n_keep)  # (B, H, n_keep)
    top_idx = jnp.sort(top_idx, axis=-1)

    gather_idx = jnp.broadcast_to(top_idx[..., None], (B, H, n_keep, Dh))
    kept_cand_k = jnp.take_along_axis(cand_k, gather_idx, axis=2)
    kept_cand_v = jnp.take_along_axis(cand_v, gather_idx, axis=2)

    # Write back: anchors at [0, n_anchor), kept candidates at
    # [n_anchor, n_anchor + n_keep), zero out the rest.
    k_pad_out = jnp.zeros_like(k_pad)
    v_pad_out = jnp.zeros_like(v_pad)
    k_pad_out = jax.lax.dynamic_update_slice_in_dim(k_pad_out, anchor_k, 0, axis=2)
    v_pad_out = jax.lax.dynamic_update_slice_in_dim(v_pad_out, anchor_v, 0, axis=2)
    k_pad_out = jax.lax.dynamic_update_slice_in_dim(k_pad_out, kept_cand_k, n_anchor, axis=2)
    v_pad_out = jax.lax.dynamic_update_slice_in_dim(v_pad_out, kept_cand_v, n_anchor, axis=2)
    new_valid_len = jnp.asarray(n_anchor + n_keep, dtype=jnp.int32)
    return k_pad_out, v_pad_out, new_valid_len, evict_score


class Attention(nn.Module):
    """Multi-head attention with optional cache."""

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
        past_kv: tuple | None = None,
        use_cache: bool = False,
        cache_budget: int | None = None,
        num_anchor_tokens: int = 0,
    ):
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

        is_padded_cache = (
            past_kv is not None
            and isinstance(past_kv, tuple)
            and len(past_kv) == 3
        )

        if use_cache and is_padded_cache:
            return self._padded_cache_forward(
                q, k, v, past_kv, cache_budget, num_anchor_tokens, B, N
            )

        # --- Legacy path (no cache, or 2-tuple past_kv) ---
        if past_kv is not None:
            past_k, past_v = past_kv
            k = jnp.concatenate([past_k, k], axis=2)
            v = jnp.concatenate([past_v, v], axis=2)

        evict_score: jnp.ndarray | None = None
        if use_cache and cache_budget is not None and k.shape[2] > cache_budget:
            k, v, evict_score = _evict_kv(k, v, cache_budget, num_anchor_tokens)

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

    def _padded_cache_forward(
        self,
        q: jnp.ndarray,
        new_k: jnp.ndarray,
        new_v: jnp.ndarray,
        past_kv: tuple,
        cache_budget: int | None,
        num_anchor_tokens: int,
        B: int,
        N: int,
    ):
        """Padded-cache path: dynamic_update_slice writes + SDPA with bool mask."""
        past_k_pad, past_v_pad, valid_len = past_kv
        _, _, MAX, Dh = past_k_pad.shape
        cache_dtype = past_k_pad.dtype  # honour whatever dtype the cache was allocated in

        # 1. Append new K/V at offset valid_len (must match cache dtype).
        k_pad = jax.lax.dynamic_update_slice_in_dim(
            past_k_pad, new_k.astype(cache_dtype), valid_len, axis=2
        )
        v_pad = jax.lax.dynamic_update_slice_in_dim(
            past_v_pad, new_v.astype(cache_dtype), valid_len, axis=2
        )
        new_valid_len = valid_len + N

        # 2. Optional eviction.
        did_evict = jnp.asarray(False)
        evict_score_payload = jnp.asarray(0.0, dtype=jnp.float32)
        if cache_budget is not None:
            # Branch: if new_valid_len > cache_budget, evict; else passthrough.
            def _do_evict(operand):
                kp, vp, vl = operand
                return _padded_evict(kp, vp, vl, cache_budget, num_anchor_tokens)

            def _no_evict(operand):
                kp, vp, vl = operand
                return (kp, vp, vl, jnp.asarray(0.0, dtype=jnp.float32))

            k_pad, v_pad, new_valid_len, evict_score_payload = jax.lax.cond(
                new_valid_len > cache_budget,
                _do_evict,
                _no_evict,
                operand=(k_pad, v_pad, new_valid_len),
            )
            did_evict = new_valid_len <= cache_budget  # after clamp: True iff eviction ran
            # Note: the cleaner check is "new_valid_len_before > cache_budget".
            # Recompute that:
            # Actually we want did_evict = (valid_len + N > cache_budget).
            # Compute from the original pre-cond values:
            did_evict = (valid_len + N) > jnp.asarray(cache_budget, dtype=jnp.int32)

        # 3. Attention via jax.nn.dot_product_attention with bool mask.
        # Cast all inputs to cache_dtype (bf16 when the extractor uses bf16)
        # so cuDNN flash attention can lower to its kernel.  Replace the
        # explicit -inf bias with a bool mask (True = valid slot).
        q_tnhd = jnp.transpose(q.astype(cache_dtype), (0, 2, 1, 3))  # (B, N, H, Dh)
        k_tnhd = jnp.transpose(k_pad, (0, 2, 1, 3))  # (B, MAX, H, Dh) — already cache_dtype
        v_tnhd = jnp.transpose(v_pad, (0, 2, 1, 3))

        head_dim_f = float(q.shape[-1])
        scale = 1.0 / head_dim_f ** 0.5

        # cuDNN flash attention with key_value_seq_lengths: tells cuDNN how
        # many K/V slots are valid without materialising a (B,H,N,MAX) bias.
        # Shape: (B,) int32 per-sample valid length.
        B_dim = q_tnhd.shape[0]
        kv_len = jnp.broadcast_to(new_valid_len.reshape(1), (B_dim,)).astype(jnp.int32)
        # cuDNN flash supports only fp16/bf16/fp8. Fall back to XLA for fp32.
        if q_tnhd.dtype in (jnp.bfloat16, jnp.float16):
            out_tnhd = jax.nn.dot_product_attention(
                q_tnhd, k_tnhd, v_tnhd,
                scale=scale, is_causal=False,
                key_value_seq_lengths=kv_len,
                implementation='cudnn',
            )
        else:
            pos = jnp.arange(MAX, dtype=jnp.int32)
            mask_xla = (pos < new_valid_len).reshape(1, 1, 1, MAX)
            out_tnhd = jax.nn.dot_product_attention(
                q_tnhd, k_tnhd, v_tnhd,
                scale=scale, is_causal=False,
                mask=mask_xla, implementation='xla',
            )
        out = jnp.transpose(out_tnhd, (0, 2, 1, 3)).astype(q.dtype)
        out = out.reshape(B, N, self.dim)
        out = nn.Dense(self.dim, use_bias=self.proj_bias, name="proj")(out)

        new_kv = (k_pad, v_pad, new_valid_len)
        if cache_budget is not None:
            # In padded-jit mode we can't use Python-None sentinels across
            # compile boundaries, so return the score paired with a did_evict
            # flag; the aggregator decodes with jnp.where.
            return out, new_kv, (did_evict, evict_score_payload)
        return out, new_kv
