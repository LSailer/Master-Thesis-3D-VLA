"""Aggregator — alternating frame/global attention over S frames.

Mirrors ``streamvggt.models.aggregator.Aggregator``:

* **No-cache path** (``use_cache=False``): Processes all S frames at once,
  global attention uses an S*P-length causal mask so frame i only sees
  frames 0..i.
* **Streaming cache path** (``use_cache=True``): Processes S=1 new frame per
  call. Global blocks receive the prior frames' K/V via ``past_kvs[i]`` and
  return an updated cache.

  Two cache formats are supported:
    - Legacy 2-tuple ``(k, v)``: variable-length; eviction yields a compact
      concat of anchors + kept candidates.
    - Padded 3-tuple ``(k_pad, v_pad, valid_len)``: fixed ``(B, H, MAX, Dh)``
      shape with a traced int32 scalar ``valid_len``. Enables jit without
      shape-polymorphism. Feature_extractor uses this form; parity tests
      call the aggregator directly with 2-tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.vggt.jax.backbone import DinoV2Backbone
from src.vggt.jax.block import Block
from src.vggt.jax.rope import compute_1d_rope_tables


# Frozen ResNet statistics used by the reference (aggregator.py:20-21).
_RESNET_MEAN = (0.485, 0.456, 0.406)
_RESNET_STD = (0.229, 0.224, 0.225)


def _slice_expand_and_flatten(token_tensor: jnp.ndarray, B: int, S: int) -> jnp.ndarray:
    """Port of ``aggregator.slice_expand_and_flatten``.

    token_tensor: (1, 2, X, C). First slice is used for frame 0, second for
    frames 1..S-1. Output: (B*S, X, C).
    """
    _, _, X, C = token_tensor.shape
    query = jnp.broadcast_to(token_tensor[:, 0:1], (B, 1, X, C))
    if S > 1:
        others = jnp.broadcast_to(token_tensor[:, 1:2], (B, S - 1, X, C))
        combined = jnp.concatenate([query, others], axis=1)
    else:
        combined = query
    return combined.reshape(B * S, X, C)


def _make_position_grid(
    B: int, S: int, grid_h: int, grid_w: int, patch_start_idx: int
) -> jnp.ndarray:
    """Build positions like ``PositionGetter`` + aggregator's +1 shift.

    Output: (B*S, P, 2) where P = patch_start_idx + grid_h*grid_w. Special
    tokens (camera + register) get position 0; patch tokens get (y+1, x+1).
    """
    ys, xs = jnp.meshgrid(jnp.arange(grid_h), jnp.arange(grid_w), indexing="ij")
    patch_pos = jnp.stack([ys.reshape(-1), xs.reshape(-1)], axis=-1)  # (P_patch, 2)
    patch_pos = patch_pos + 1
    patch_pos = jnp.broadcast_to(patch_pos, (B * S, patch_pos.shape[0], 2))
    special = jnp.zeros((B * S, patch_start_idx, 2), dtype=patch_pos.dtype)
    return jnp.concatenate([special, patch_pos], axis=1)


def _make_causal_global_mask(S: int, P: int, dtype=jnp.float32) -> jnp.ndarray:
    """Per-block causal frame mask for global attention over S*P tokens."""
    L = S * P
    frame_ids = jnp.arange(L) // P
    future = frame_ids[:, None] < frame_ids[None, :]
    return future.astype(dtype) * jnp.finfo(dtype).min


def _calculate_dynamic_budgets(
    last_scores: jnp.ndarray, total_budget: int
) -> jnp.ndarray:
    """Port of ``aggregator._calculate_dynamic_budgets``.

    Blocks that evicted high-similarity tokens last frame (high
    ``last_scores``) have low "diversity" and receive a smaller share of
    the global budget. The per-block budget is ``softmax(2*(1-score)) *
    total_budget`` truncated to int.
    """
    total_budget = max(int(total_budget), 0)
    diversity = 1.0 - last_scores
    scaled = diversity / 0.5
    proportions = jax.nn.softmax(scaled, axis=0)
    budgets = proportions * total_budget
    return budgets.astype(jnp.int32)


@dataclass(frozen=True)
class _CacheState:
    """Streaming-cache settings shared by all global attention blocks."""

    enabled: bool
    past_kvs: list | None
    last_scores: jnp.ndarray | None
    padded_mode: bool
    current_budgets: list[int] | jnp.ndarray | None


@dataclass(frozen=True)
class _TokenLayout:
    """Static token dimensions used to switch between frame/global layouts."""

    B: int
    S: int
    P: int
    embed_dim: int

    def to_frame(self, tokens: jnp.ndarray) -> jnp.ndarray:
        if tokens.shape in (
            (self.B, self.S, self.P, self.embed_dim),
            (self.B, self.S * self.P, self.embed_dim),
        ):
            return tokens.reshape(self.B * self.S, self.P, self.embed_dim)
        return tokens

    def to_global(self, frame_tokens: jnp.ndarray) -> jnp.ndarray:
        return frame_tokens.reshape(self.B, self.S, self.P, self.embed_dim).reshape(
            self.B, self.S * self.P, self.embed_dim
        )

    def split_layers(self, tokens: jnp.ndarray) -> jnp.ndarray:
        return tokens.reshape(self.B, self.S, self.P, self.embed_dim)


def _validate_image_shape(images: jnp.ndarray, *, img_size: int) -> tuple[int, ...]:
    """Validate image channels/spatial size and return static input dimensions."""
    B, S, C_in, H, W = images.shape
    if C_in != 3:
        raise ValueError(f"expected 3 input channels, got {C_in}")
    if H != img_size or W != img_size:
        raise NotImplementedError(
            f"Aggregator is fixed at {img_size}x{img_size}; got {H}x{W}."
        )
    return B, S, C_in, H, W


def _prepare_cache_state(
    *,
    use_cache: bool,
    past_kvs: list | None,
    last_scores: jnp.ndarray | None,
    total_budget: int | None,
    current_budgets_static: tuple[int, ...] | None,
    depth: int,
    S: int,
) -> _CacheState:
    """Validate and collect all streaming-cache metadata for one call."""
    if not use_cache:
        return _CacheState(False, None, None, False, None)
    if S != 1:
        raise ValueError(f"use_cache expects S=1 per call, got S={S}")
    if past_kvs is None:
        past_kvs = [None] * depth
    if len(past_kvs) != depth:
        raise ValueError(f"past_kvs length {len(past_kvs)} != depth {depth}")
    if last_scores is None:
        last_scores = jnp.zeros((depth,), dtype=jnp.float32)

    padded_mode = any(
        entry is not None and isinstance(entry, tuple) and len(entry) == 3
        for entry in past_kvs
    )
    current_budgets = None
    if total_budget is not None:
        current_budgets = (
            list(current_budgets_static)
            if current_budgets_static is not None
            else _calculate_dynamic_budgets(last_scores, total_budget)
        )
    return _CacheState(True, past_kvs, last_scores, padded_mode, current_budgets)


def _normalise_resnet_images(images: jnp.ndarray) -> jnp.ndarray:
    """Apply the reference ResNet normalisation to [0, 1] RGB images."""
    mean = jnp.asarray(_RESNET_MEAN, dtype=images.dtype).reshape(1, 1, 3, 1, 1)
    std = jnp.asarray(_RESNET_STD, dtype=images.dtype).reshape(1, 1, 3, 1, 1)
    return (images - mean) / std


def _make_special_tokens(
    *,
    camera_token_full: jnp.ndarray,
    register_token_full: jnp.ndarray,
    patch_tokens: jnp.ndarray,
    B: int,
    S: int,
    use_cache: bool,
    past_frame_idx: int,
    num_register_tokens: int,
    embed_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Broadcast camera/register tokens for either batched or streaming input."""
    if use_cache:
        slot = 0 if past_frame_idx == 0 else 1
        camera = jnp.broadcast_to(
            camera_token_full[0, slot : slot + 1].astype(patch_tokens.dtype),
            (B, 1, embed_dim),
        )
        register = jnp.broadcast_to(
            register_token_full[0, slot : slot + 1]
            .reshape(1, num_register_tokens, embed_dim)
            .astype(patch_tokens.dtype),
            (B, num_register_tokens, embed_dim),
        )
        return camera, register
    camera = _slice_expand_and_flatten(
        camera_token_full.astype(patch_tokens.dtype), B, S
    )
    register = _slice_expand_and_flatten(
        register_token_full.astype(patch_tokens.dtype), B, S
    )
    return camera, register


def _prepare_attention_geometry(
    *,
    layout: _TokenLayout,
    img_size: int,
    patch_size: int,
    patch_start_idx: int,
    num_heads: int,
    rope_freq: float,
    dtype,
    use_cache: bool,
) -> tuple[
    jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray | None
]:
    """Build token positions, RoPE tables, and the no-cache causal mask."""
    grid = img_size // patch_size
    positions_bs_p = _make_position_grid(
        layout.B, layout.S, grid, grid, patch_start_idx
    )
    positions_b_sp = positions_bs_p.reshape(layout.B, layout.S, layout.P, 2).reshape(
        layout.B, layout.S * layout.P, 2
    )
    head_dim = layout.embed_dim // num_heads
    cos_t, sin_t = compute_1d_rope_tables(
        dim=head_dim // 2,
        max_pos=grid + 1,
        frequency=rope_freq,
        dtype=dtype,
    )
    global_mask = (
        None
        if use_cache
        else _make_causal_global_mask(layout.S, layout.P, dtype=jnp.float32)
    )
    return positions_bs_p, positions_b_sp, (cos_t, sin_t), global_mask


def _record_cache_scores(
    *,
    new_scores: list,
    any_evicted: bool,
    scores,
    fallback_score: jnp.ndarray,
) -> bool:
    """Record eviction scores while preserving old scores when no eviction occurred."""
    if isinstance(scores, tuple):
        did_evict, score_scalar = scores
        new_scores.append((did_evict, score_scalar))
        return any_evicted
    if scores is not None:
        new_scores.append(scores)
        return True
    new_scores.append(fallback_score)
    return any_evicted


def _finalise_last_scores(
    *,
    cache_state: _CacheState,
    new_scores: list,
    any_evicted: bool,
) -> jnp.ndarray:
    """Merge per-block eviction scores into the state returned to the caller."""
    if cache_state.padded_mode and cache_state.current_budgets is not None:
        if cache_state.last_scores is None:
            raise ValueError("padded cache requires last_scores")
        per_block_new = []
        for b, entry in enumerate(new_scores):
            if isinstance(entry, tuple):
                did_evict, score = entry
                per_block_new.append(
                    jnp.where(did_evict, score, cache_state.last_scores[b])
                )
            else:
                per_block_new.append(jnp.asarray(entry, dtype=jnp.float32))
        return jnp.stack(per_block_new).astype(jnp.float32)
    if any_evicted:
        return jnp.stack([jnp.asarray(s, dtype=jnp.float32) for s in new_scores])
    if cache_state.last_scores is None:
        raise ValueError("cache_state.last_scores is required")
    return cache_state.last_scores


class Aggregator(nn.Module):
    """Alternating frame/global attention tower."""

    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    init_values: float = 0.01
    norm_eps: float = 1e-5
    rope_freq: float = 100.0
    aa_order: tuple[str, ...] = ("frame", "global")

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        *,
        use_cache: bool = False,
        past_kvs: list | None = None,
        past_frame_idx: int = 0,
        total_budget: int | None = None,
        last_scores: jnp.ndarray | None = None,
        current_budgets_static: tuple[int, ...] | None = None,
    ):
        """Forward pass.

        Args:
            images: (B, S, 3, H, W) float in [0, 1]. In cache mode, S must be 1.
            use_cache: If True, take the streaming cache path.
            past_kvs: List of length ``depth``; entries are None, a 2-tuple
                (legacy compact), or a 3-tuple (padded).
            past_frame_idx: Zero-based frame index.
            total_budget: Optional global cache size. If None, eviction off.
            last_scores: Optional (depth,) fp32 array from prior call.
            current_budgets_static: Optional tuple of Python ints (length
                ``depth``) giving the per-block budgets. When provided, used
                directly; allows the caller to compute budgets outside jit
                and pass them as static args so top_k's k stays Python-int.

        Returns:
            No-cache:  ``(output_list, patch_start_idx)``.
            Cache:     ``(output_list, patch_start_idx, new_past_kvs,
                          new_last_scores)``.
        """
        B, S, C_in, H, W = _validate_image_shape(images, img_size=self.img_size)
        cache_state = _prepare_cache_state(
            use_cache=use_cache,
            past_kvs=past_kvs,
            last_scores=last_scores,
            total_budget=total_budget,
            current_budgets_static=current_budgets_static,
            depth=self.depth,
            S=S,
        )
        images = _normalise_resnet_images(images)

        x = images.reshape(B * S, C_in, H, W)
        patch_tokens = DinoV2Backbone(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            num_register_tokens=self.num_register_tokens,
            init_values=1.0,  # DINOv2 default
            norm_eps=1e-6,  # DINOv2 override
            name="patch_embed",
        )(x)

        camera_token_full = self.param(
            "camera_token",
            lambda _k, shape: jnp.zeros(shape, dtype=jnp.float32),
            (1, 2, 1, self.embed_dim),
        )
        register_token_full = self.param(
            "register_token",
            lambda _k, shape: jnp.zeros(shape, dtype=jnp.float32),
            (1, 2, self.num_register_tokens, self.embed_dim),
        )
        camera, register = _make_special_tokens(
            camera_token_full=camera_token_full,
            register_token_full=register_token_full,
            patch_tokens=patch_tokens,
            B=B,
            S=S,
            use_cache=cache_state.enabled,
            past_frame_idx=past_frame_idx,
            num_register_tokens=self.num_register_tokens,
            embed_dim=self.embed_dim,
        )
        tokens = jnp.concatenate([camera, register, patch_tokens], axis=1)
        patch_start_idx = 1 + self.num_register_tokens  # 5
        layout = _TokenLayout(
            B=B,
            S=S,
            P=tokens.shape[1],
            embed_dim=self.embed_dim,
        )

        positions_bs_p, positions_b_sp, rope_tables, global_mask = (
            _prepare_attention_geometry(
                layout=layout,
                img_size=self.img_size,
                patch_size=self.patch_size,
                patch_start_idx=patch_start_idx,
                num_heads=self.num_heads,
                rope_freq=self.rope_freq,
                dtype=images.dtype,
                use_cache=cache_state.enabled,
            )
        )

        output_list: list[jnp.ndarray] = []
        new_past_kvs: list = []
        new_scores: list = []
        any_evicted = False

        for b in range(self.depth):
            tokens_frame = layout.to_frame(tokens)
            tokens_frame = cast(
                jnp.ndarray,
                Block(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qk_norm=True,
                    init_values=self.init_values,
                    norm_eps=self.norm_eps,
                    name=f"frame_blocks_{b}",
                )(tokens_frame, rope_tables=rope_tables, positions=positions_bs_p),
            )
            frame_inter = layout.split_layers(tokens_frame)

            tokens_global = layout.to_global(tokens_frame)
            global_block = Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qk_norm=True,
                init_values=self.init_values,
                norm_eps=self.norm_eps,
                name=f"global_blocks_{b}",
            )
            if cache_state.enabled:
                if cache_state.past_kvs is None:
                    raise ValueError("cache enabled requires past_kvs")
                past_entry = cache_state.past_kvs[b]
                current_budgets = cache_state.current_budgets
                if current_budgets is not None:
                    if cache_state.last_scores is None:
                        raise ValueError("budgeted cache requires last_scores")
                    tokens_global, new_kv, scores = cast(
                        tuple[jnp.ndarray, tuple, object],
                        global_block(
                            tokens_global,
                            rope_tables=rope_tables,
                            positions=positions_b_sp,
                            attn_mask=None,
                            past_kv=past_entry,
                            use_cache=True,
                            cache_budget=int(current_budgets[b]),
                            num_anchor_tokens=layout.P,
                        ),
                    )
                    any_evicted = _record_cache_scores(
                        new_scores=new_scores,
                        any_evicted=any_evicted,
                        scores=scores,
                        fallback_score=cache_state.last_scores[b],
                    )
                else:
                    tokens_global, new_kv = cast(
                        tuple[jnp.ndarray, tuple],
                        global_block(
                            tokens_global,
                            rope_tables=rope_tables,
                            positions=positions_b_sp,
                            attn_mask=None,
                            past_kv=past_entry,
                            use_cache=True,
                        ),
                    )
                new_past_kvs.append(new_kv)
            else:
                tokens_global = cast(
                    jnp.ndarray,
                    global_block(
                        tokens_global,
                        rope_tables=rope_tables,
                        positions=positions_b_sp,
                        attn_mask=global_mask,
                    ),
                )
            global_inter = layout.split_layers(tokens_global)
            tokens = layout.to_frame(global_inter)

            output_list.append(jnp.concatenate([frame_inter, global_inter], axis=-1))

        if cache_state.enabled:
            new_last_scores = _finalise_last_scores(
                cache_state=cache_state,
                new_scores=new_scores,
                any_evicted=any_evicted,
            )
            return output_list, patch_start_idx, new_past_kvs, new_last_scores
        return output_list, patch_start_idx
