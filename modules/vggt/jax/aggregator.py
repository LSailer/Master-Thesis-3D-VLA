"""Aggregator — alternating frame/global attention over S frames.

Mirrors ``streamvggt.models.aggregator.Aggregator`` for the **no-cache** path
(``use_cache=False`` in the reference). The cache/eviction/dynamic-budget
branches land in Step 6a+.

Forward pipeline:
    images (B, S, 3, H, W) in [0, 1]
      -> ResNet normalise
      -> (B*S, 3, H, W) patch conv via DinoV2Backbone
      -> prepend camera_token + register_token per frame (slice-expand-flatten)
      -> 24 alternating (frame, global) block pairs
      -> output_list: [(B, S, P, 2*C)] * depth  +  patch_start_idx = 5

Each global block in the no-cache path uses a per-block causal mask over
``S * P`` tokens (frame i attends only to frames 0..i).
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from modules.vggt.jax.backbone import DinoV2Backbone
from modules.vggt.jax.block import Block
from modules.vggt.jax.rope import compute_1d_rope_tables


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
    """Per-block causal frame mask for global attention over S*P tokens.

    Token at position k belongs to frame ``k // P``. Future frames are masked
    with ``finfo(dtype).min`` additive (pre-softmax). Returns (S*P, S*P) which
    broadcasts against (B, H, L, L) attention scores.
    """
    L = S * P
    frame_ids = jnp.arange(L) // P
    future = frame_ids[:, None] < frame_ids[None, :]
    return future.astype(dtype) * jnp.finfo(dtype).min


class Aggregator(nn.Module):
    """Alternating frame/global attention tower (no-cache path).

    Attributes:
        img_size: Fixed input side length (518 for thesis).
        patch_size: Patch conv stride (14).
        embed_dim: Token feature dim (1024).
        depth: Number of (frame, global) block pairs (24).
        num_heads: Attention heads (16).
        mlp_ratio: MLP expansion factor (4).
        num_register_tokens: Register tokens prepended per frame (4).
        init_values: LayerScale init for aggregator blocks (0.01).
        norm_eps: LayerNorm epsilon used by frame/global blocks (1e-5).
        rope_freq: 2D RoPE base frequency (100.0).
        aa_order: Tuple of attention types; defaults to ("frame", "global").
    """

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
    def __call__(self, images: jnp.ndarray) -> tuple[list[jnp.ndarray], int]:
        """Forward pass.

        Args:
            images: (B, S, 3, H, W) float in [0, 1].

        Returns:
            (output_list, patch_start_idx) where output_list has ``depth``
            tensors of shape (B, S, P, 2*embed_dim) and patch_start_idx = 5.
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"expected 3 input channels, got {C_in}")
        if H != self.img_size or W != self.img_size:
            raise NotImplementedError(
                f"Aggregator is fixed at {self.img_size}x{self.img_size}; got {H}x{W}."
            )

        # ResNet normalisation on [0, 1] images.
        mean = jnp.asarray(_RESNET_MEAN, dtype=images.dtype).reshape(1, 1, 3, 1, 1)
        std = jnp.asarray(_RESNET_STD, dtype=images.dtype).reshape(1, 1, 3, 1, 1)
        images = (images - mean) / std

        # Flatten batch/frame for patch conv.
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
        # (B*S, P_patch, embed_dim) where P_patch = (H/patch_size)^2
        P_patch = patch_tokens.shape[1]

        # Camera + register tokens (aggregator-level).
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
        camera = _slice_expand_and_flatten(camera_token_full.astype(patch_tokens.dtype), B, S)
        register = _slice_expand_and_flatten(
            register_token_full.astype(patch_tokens.dtype), B, S
        )
        tokens = jnp.concatenate([camera, register, patch_tokens], axis=1)
        P = tokens.shape[1]
        patch_start_idx = 1 + self.num_register_tokens  # 5

        # Positions + RoPE tables (precomputed for fixed 37x37 grid).
        grid = self.img_size // self.patch_size
        positions_bs_p = _make_position_grid(B, S, grid, grid, patch_start_idx)  # (B*S, P, 2)
        head_dim = self.embed_dim // self.num_heads
        cos_t, sin_t = compute_1d_rope_tables(
            dim=head_dim // 2,
            max_pos=grid + 1,  # patches shifted to [1, grid+1)
            frequency=self.rope_freq,
            dtype=jnp.float32,
        )
        rope_tables = (cos_t, sin_t)

        # Precompute per-block causal mask for global attention (S*P length).
        global_mask = _make_causal_global_mask(S, P, dtype=jnp.float32)

        # Pre-reshape helpers
        def _to_frame(t):  # (B, S, P, C) -> (B*S, P, C)
            return t.reshape(B * S, P, self.embed_dim)

        def _to_global(t_flat):  # (B*S, P, C) -> (B, S*P, C)
            return t_flat.reshape(B, S, P, self.embed_dim).reshape(B, S * P, self.embed_dim)

        positions_b_sp = positions_bs_p.reshape(B, S, P, 2).reshape(B, S * P, 2)

        output_list: list[jnp.ndarray] = []
        # In the reference, aa_block_size defaults to 1 so each outer step
        # runs exactly one frame-block then one global-block. depth == aa_block_num.
        for b in range(self.depth):
            # ---- Frame attention: (B*S, P, C) ----
            tokens_frame = _to_frame(tokens) if tokens.shape == (B, S * P, self.embed_dim) else tokens
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

            # ---- Global attention: (B, S*P, C) ----
            tokens_global = _to_global(tokens_frame)
            tokens_global = Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qk_norm=True,
                init_values=self.init_values,
                norm_eps=self.norm_eps,
                name=f"global_blocks_{b}",
            )(
                tokens_global,
                rope_tables=rope_tables,
                positions=positions_b_sp,
                attn_mask=global_mask,
            )
            global_inter = tokens_global.reshape(B, S, P, self.embed_dim)

            # Carry for next iteration (flat-frame layout).
            tokens = _to_frame(global_inter)

            # Each level emits concat([frame, global], axis=-1) -> (B, S, P, 2C).
            output_list.append(jnp.concatenate([frame_inter, global_inter], axis=-1))

        return output_list, patch_start_idx
