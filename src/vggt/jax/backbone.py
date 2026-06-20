"""DINOv2 ViT-L/14-reg backbone — the Aggregator's ``patch_embed`` module.

Mirrors ``streamvggt.layers.vision_transformer.vit_large`` with
``num_register_tokens=4`` and ``init_values=1.0`` (the defaults passed by
``Aggregator.__build_patch_embed__``).

**Fixed 518x518 input.** For the thesis, the aggregator always sees the
full 518x518 frame. Since 518/14 = 37 exactly and the checkpoint's
``pos_embed`` has 1+37*37=1370 slots, the PyTorch ``interpolate_pos_encoding``
fast-path triggers (``npatch == N and w == h``) and returns ``pos_embed``
unchanged. We therefore skip the bicubic-AA interpolation math entirely
and add ``pos_embed`` as-is. Reintroducing interpolation for variable
input sizes is a follow-up.

**What we expose.** Like the PyTorch reference, the forward returns only
the patch tokens (the normalized ``x_norm_patchtokens`` slice): cls and
register tokens are dropped because the aggregator concatenates its own
camera/register tokens downstream.

Param tree matches ``src/vggt/jax/weight_transfer.py``:
    patch_embed/                       # DinoV2Backbone
      cls_token                         # raw param (1, 1, 1024)
      mask_token                        # raw param (1, 1024)  -- unused at inference
      register_tokens                   # raw param (1, 4, 1024)
      pos_embed                         # raw param (1, 1370, 1024)
      norm/ {scale, bias}               # final LayerNorm (eps=1e-6)
      patch_embed/                      # inner PatchEmbed
        proj/ {kernel, bias}            # Conv2d 14x14 stride 14
      blocks_0/ ... blocks_23/          # 24 ViT blocks (qk_norm=False, LayerScale init=1.0)
"""

from __future__ import annotations

from typing import cast

import flax.linen as nn
import jax.numpy as jnp

from src.vggt.jax.block import Block


# --------------------------------------------------------------------------- #
#  Inner PatchEmbed (Conv2d 14x14 stride 14)
# --------------------------------------------------------------------------- #


class PatchEmbed(nn.Module):
    """Conv-based patch embedding: (B, H, W, 3) -> (B, H/p, W/p, embed_dim)."""

    embed_dim: int
    patch_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is already NHWC (caller transposed).
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="proj",
        )(x)
        return x


# --------------------------------------------------------------------------- #
#  DINOv2 ViT-L/14-reg backbone
# --------------------------------------------------------------------------- #


class DinoV2Backbone(nn.Module):
    """DINOv2 ViT-L/14-reg frozen inference, fixed 518x518 input.

    Forward: NCHW image in [0, 1] (after ResNet normalisation done upstream)
    -> (B, num_patches, embed_dim) patch-token features.

    Attributes match the config baked into ``Aggregator.__build_patch_embed__``
    for ``patch_embed="dinov2_vitl14_reg"``: depth=24, embed_dim=1024,
    num_heads=16, mlp_ratio=4, num_register_tokens=4, init_values=1.0.
    qk_norm stays False (DINOv2 defaults).
    """

    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    init_values: float = 1.0
    norm_eps: float = 1e-6  # DINOv2 uses eps=1e-6 explicitly for norm1/norm2/norm

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Input: NCHW to match the PyTorch reference's calling convention.
        # The caller (aggregator) passes ``images.reshape(B*S, C, H, W)``.
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            raise NotImplementedError(
                f"DinoV2Backbone is fixed at {self.img_size}x{self.img_size}; "
                f"got {H}x{W}. Reintroducing bicubic-AA pos_embed interpolation "
                "is a follow-up."
            )
        if C != 3:
            raise ValueError(f"expected 3 input channels, got {C}")

        # Declare all top-level params up front so they land in the param tree
        # even when unused (mask_token is loaded but not used at inference).
        cls_token = self.param(
            "cls_token",
            lambda _k, shape: jnp.zeros(shape, dtype=jnp.float32),
            (1, 1, self.embed_dim),
        )
        _mask_token = self.param(  # noqa: F841 -- loaded from checkpoint, unused here
            "mask_token",
            lambda _k, shape: jnp.zeros(shape, dtype=jnp.float32),
            (1, self.embed_dim),
        )
        register_tokens = self.param(
            "register_tokens",
            lambda _k, shape: jnp.zeros(shape, dtype=jnp.float32),
            (1, self.num_register_tokens, self.embed_dim),
        )
        num_patches = (self.img_size // self.patch_size) ** 2
        pos_embed = self.param(
            "pos_embed",
            lambda _k, shape: jnp.zeros(shape, dtype=jnp.float32),
            (1, num_patches + 1, self.embed_dim),
        )

        # --- PatchEmbed (Conv2d 14x14 stride 14) ---
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = PatchEmbed(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            name="patch_embed",
        )(x_nhwc)
        # (B, 37, 37, 1024) -> (B, 1369, 1024)
        x = x.reshape(B, -1, self.embed_dim)

        # --- prepend cls_token ---
        cls = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim)).astype(x.dtype)
        x = jnp.concatenate([cls, x], axis=1)  # (B, 1+P, 1024)

        # --- add pos_embed ---
        # Reference fast-path: if npatch == N and square, return pos_embed as-is.
        # Our fixed 518 input always satisfies this, so plain addition.
        x = x + pos_embed.astype(x.dtype)

        # --- insert register tokens between cls and patches ---
        reg = jnp.broadcast_to(
            register_tokens, (B, self.num_register_tokens, self.embed_dim)
        ).astype(x.dtype)
        x = jnp.concatenate([x[:, :1], reg, x[:, 1:]], axis=1)
        # Now (B, 1 + num_register + num_patches, embed_dim) = (B, 1374, 1024)

        # --- 24 ViT-L blocks (DINOv2 has qk_norm=False, LayerScale init=1.0) ---
        for i in range(self.depth):
            x = cast(
                jnp.ndarray,
                Block(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qk_norm=False,
                    init_values=self.init_values,
                    norm_eps=self.norm_eps,
                    name=f"blocks_{i}",
                )(x),
            )  # no rope for DINOv2

        # --- final LayerNorm (eps=1e-6) ---
        x = nn.LayerNorm(epsilon=self.norm_eps, name="norm")(x)

        # Return patch tokens only: drop cls (index 0) and register tokens.
        # This mirrors the reference's `x_norm_patchtokens` slice.
        return x[:, 1 + self.num_register_tokens :]
