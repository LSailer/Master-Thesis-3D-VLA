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
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"expected 3 input channels, got {C_in}")
        if H != self.img_size or W != self.img_size:
            raise NotImplementedError(
                f"Aggregator is fixed at {self.img_size}x{self.img_size}; got {H}x{W}."
            )
        if use_cache and S != 1:
            raise ValueError(f"use_cache expects S=1 per call, got S={S}")
        if use_cache and past_kvs is None:
            past_kvs = [None] * self.depth
        if use_cache and len(past_kvs) != self.depth:
            raise ValueError(
                f"past_kvs length {len(past_kvs)} != depth {self.depth}"
            )
        if use_cache and last_scores is None:
            last_scores = jnp.zeros((self.depth,), dtype=jnp.float32)

        # Detect whether any past cache entry is padded (3-tuple) — all must
        # match. None + any 3-tuples means mixed first-call semantics.
        padded_mode = False
        if use_cache and past_kvs is not None:
            for entry in past_kvs:
                if entry is not None and isinstance(entry, tuple) and len(entry) == 3:
                    padded_mode = True
                    break

        # Per-block budgets.
        if use_cache and total_budget is not None:
            if current_budgets_static is not None:
                # Static Python tuple — used for the jitted path.
                current_budgets = list(current_budgets_static)
            else:
                # Compute from last_scores (non-jit path).
                current_budgets = _calculate_dynamic_budgets(last_scores, total_budget)
        else:
            current_budgets = None

        # ResNet normalisation on [0, 1] images.
        mean = jnp.asarray(_RESNET_MEAN, dtype=images.dtype).reshape(1, 1, 3, 1, 1)
        std = jnp.asarray(_RESNET_STD, dtype=images.dtype).reshape(1, 1, 3, 1, 1)
        images = (images - mean) / std

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
            norm_eps=1e-6,     # DINOv2 override
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
        if use_cache:
            slot = 0 if past_frame_idx == 0 else 1
            n_reg = self.num_register_tokens
            camera = jnp.broadcast_to(
                camera_token_full[0, slot : slot + 1].astype(patch_tokens.dtype),
                (B, 1, self.embed_dim),
            )
            register = jnp.broadcast_to(
                register_token_full[0, slot : slot + 1].reshape(1, n_reg, self.embed_dim)
                .astype(patch_tokens.dtype),
                (B, n_reg, self.embed_dim),
            )
        else:
            camera = _slice_expand_and_flatten(
                camera_token_full.astype(patch_tokens.dtype), B, S
            )
            register = _slice_expand_and_flatten(
                register_token_full.astype(patch_tokens.dtype), B, S
            )
        tokens = jnp.concatenate([camera, register, patch_tokens], axis=1)
        P = tokens.shape[1]
        patch_start_idx = 1 + self.num_register_tokens  # 5

        grid = self.img_size // self.patch_size
        positions_bs_p = _make_position_grid(B, S, grid, grid, patch_start_idx)
        head_dim = self.embed_dim // self.num_heads
        cos_t, sin_t = compute_1d_rope_tables(
            dim=head_dim // 2,
            max_pos=grid + 1,
            frequency=self.rope_freq,
            dtype=images.dtype,
        )
        rope_tables = (cos_t, sin_t)

        global_mask = None if use_cache else _make_causal_global_mask(
            S, P, dtype=jnp.float32
        )

        def _to_frame(t):
            return t.reshape(B * S, P, self.embed_dim)

        def _to_global(t_flat):
            return t_flat.reshape(B, S, P, self.embed_dim).reshape(
                B, S * P, self.embed_dim
            )

        positions_b_sp = positions_bs_p.reshape(B, S, P, 2).reshape(B, S * P, 2)

        output_list: list[jnp.ndarray] = []
        new_past_kvs: list = []
        new_scores: list = []
        any_evicted = False
        any_evicted_traced: jnp.ndarray | None = None  # padded mode

        for b in range(self.depth):
            tokens_frame = (
                _to_frame(tokens)
                if tokens.shape == (B, S * P, self.embed_dim)
                else tokens
            )
            tokens_frame = Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qk_norm=True,
                init_values=self.init_values,
                norm_eps=self.norm_eps,
                name=f"frame_blocks_{b}",
            )(tokens_frame, rope_tables=rope_tables, positions=positions_bs_p)
            frame_inter = tokens_frame.reshape(B, S, P, self.embed_dim)

            tokens_global = _to_global(tokens_frame)
            global_block = Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qk_norm=True,
                init_values=self.init_values,
                norm_eps=self.norm_eps,
                name=f"global_blocks_{b}",
            )
            if use_cache:
                # In padded mode, initialize zero-pad on the first frame.
                past_entry = past_kvs[b] if past_kvs is not None else None
                if padded_mode and past_entry is None:
                    # Shouldn't happen — padded mode implies entries are 3-tuples.
                    # But keep safe: fall through to 2-tuple (legacy) path.
                    pass

                if current_budgets is not None:
                    if isinstance(current_budgets, list):
                        per_block_budget = int(current_budgets[b])
                    else:
                        # jax array — only safe outside jit.
                        per_block_budget = int(current_budgets[b])
                    tokens_global, new_kv, scores = global_block(
                        tokens_global,
                        rope_tables=rope_tables,
                        positions=positions_b_sp,
                        attn_mask=None,
                        past_kv=past_entry,
                        use_cache=True,
                        cache_budget=per_block_budget,
                        num_anchor_tokens=P,
                    )
                    # Decode `scores`: legacy → None or scalar; padded →
                    # (did_evict_bool, score_scalar).
                    if isinstance(scores, tuple):
                        did_evict, score_scalar = scores
                        # Accumulate per-block record.
                        new_scores.append((did_evict, score_scalar))
                        if any_evicted_traced is None:
                            any_evicted_traced = did_evict
                        else:
                            any_evicted_traced = any_evicted_traced | did_evict
                    else:
                        if scores is not None:
                            new_scores.append(scores)
                            any_evicted = True
                        else:
                            new_scores.append(last_scores[b])
                else:
                    tokens_global, new_kv = global_block(
                        tokens_global,
                        rope_tables=rope_tables,
                        positions=positions_b_sp,
                        attn_mask=None,
                        past_kv=past_entry,
                        use_cache=True,
                    )
                new_past_kvs.append(new_kv)
            else:
                tokens_global = global_block(
                    tokens_global,
                    rope_tables=rope_tables,
                    positions=positions_b_sp,
                    attn_mask=global_mask,
                )
            global_inter = tokens_global.reshape(B, S, P, self.embed_dim)
            tokens = _to_frame(global_inter)

            output_list.append(jnp.concatenate([frame_inter, global_inter], axis=-1))

        if use_cache:
            if padded_mode and current_budgets is not None:
                # Padded mode: new_scores is a list of (did_evict, score) pairs.
                # Build new_last_scores with jnp.where to preserve last_scores
                # where no eviction fired.
                per_block_new = []
                for b, entry in enumerate(new_scores):
                    if isinstance(entry, tuple):
                        did_evict, score = entry
                        per_block_new.append(
                            jnp.where(did_evict, score, last_scores[b])
                        )
                    else:
                        # Shouldn't occur in padded mode, but be safe.
                        per_block_new.append(jnp.asarray(entry, dtype=jnp.float32))
                new_last_scores = jnp.stack(per_block_new).astype(jnp.float32)
            elif any_evicted:
                new_last_scores = jnp.stack(
                    [jnp.asarray(s, dtype=jnp.float32) for s in new_scores]
                )
            else:
                new_last_scores = last_scores
            return output_list, patch_start_idx, new_past_kvs, new_last_scores
        return output_list, patch_start_idx
