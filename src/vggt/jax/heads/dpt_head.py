"""DPTHead — port of ``streamvggt.heads.dpt_head.DPTHead`` for the point head.

Used as the **point head** in StreamVGGT (``output_dim=4``,
``activation="inv_log"``, ``conf_activation="expp1"``). The depth head — a
second DPTHead instance with ``output_dim=2, activation="exp"`` — is out of
scope for v1 and not wired here.

Pipeline, per frame chunk:
    1. ``norm(x)`` on tokens, permute to (B*S, C, H_patch, W_patch).
    2. ``projects[i]`` (1x1 Conv) -> ``+ pos_embed`` -> ``resize_layers[i]``
       for each of 4 intermediate layer indices [4, 11, 17, 23].
    3. ``scratch_forward``: fuse via layer{1..4}_rn Conv3x3 (no bias) and 4
       ``FeatureFusionBlock`` stages (refinenet4/3/2/1) with 2x bilinear
       upsampling (align_corners=True), then ``output_conv1``.
    4. Bilinear resize to image resolution (518x518 for the thesis).
    5. ``output_conv2`` (Conv3x3 -> ReLU -> Conv1x1) -> 4-channel map.
    6. ``activate_head`` splits into pts3d (inv_log on xyz) + conf (expp1).
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


_DEFAULT_INTERMEDIATE_LAYER_IDX: tuple[int, ...] = (4, 11, 17, 23)
_OUT_CHANNELS: tuple[int, int, int, int] = (256, 512, 1024, 1024)
_FEATURES = 256
_POS_EMBED_RATIO = 0.1
_POS_EMBED_OMEGA_0 = 100.0


# --------------------------------------------------------------------------- #
#  Sinusoidal positional embedding (non-learnable; heads/utils.py port)
# --------------------------------------------------------------------------- #


def _make_sincos_pos_embed(
    embed_dim: int, pos: jnp.ndarray, omega_0: float = _POS_EMBED_OMEGA_0
) -> jnp.ndarray:
    """Port of ``heads.utils.make_sincos_pos_embed`` — (M,) pos -> (M, embed_dim)."""
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    # The reference computes omega in float64; we keep fp32 since the final
    # scaling by ratio=0.1 and subsequent convs dominate any precision loss.
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (omega_0**omega)
    pos = pos.reshape(-1)
    out = jnp.einsum("m,d->md", pos, omega)
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=-1)


def _position_grid_to_embed(
    pos_grid: jnp.ndarray, embed_dim: int, omega_0: float = _POS_EMBED_OMEGA_0
) -> jnp.ndarray:
    """Port of ``heads.utils.position_grid_to_embed`` — (H, W, 2) -> (H, W, embed_dim)."""
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0)
    emb_y = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0)
    emb = jnp.concatenate([emb_x, emb_y], axis=-1)
    return emb.reshape(H, W, embed_dim)


def _create_uv_grid(width: int, height: int, aspect_ratio: float) -> jnp.ndarray:
    """Port of ``heads.utils.create_uv_grid``.

    Reference uses ``torch.meshgrid(x_coords, y_coords, indexing='xy')`` which
    yields shape ``(height, width, 2)`` tensors (xy indexing swaps to row-y
    col-x). Matches that.
    """
    diag = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag
    span_y = 1.0 / diag
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bot_y = span_y * (height - 1) / height
    x_coords = jnp.linspace(left_x, right_x, width, dtype=jnp.float32)
    y_coords = jnp.linspace(top_y, bot_y, height, dtype=jnp.float32)
    # torch meshgrid indexing='xy' returns (H, W) shaped outputs.
    yy, xx = jnp.meshgrid(y_coords, x_coords, indexing="ij")
    # But we want shape (H, W, 2) with [uu, vv] stacked. Using indexing='xy':
    xx, yy = jnp.meshgrid(x_coords, y_coords, indexing="xy")
    return jnp.stack([xx, yy], axis=-1)  # (H, W, 2)


def _apply_pos_embed(
    x: jnp.ndarray, W: int, H: int, ratio: float = _POS_EMBED_RATIO
) -> jnp.ndarray:
    """x: (N, C, H_feat, W_feat). Adds ratio * sinusoidal 2D pos embed (fixed)."""
    patch_w = x.shape[-1]
    patch_h = x.shape[-2]
    pos = _create_uv_grid(patch_w, patch_h, aspect_ratio=W / H)
    emb = _position_grid_to_embed(pos, x.shape[1])  # (H_feat, W_feat, C)
    emb = jnp.transpose(emb, (2, 0, 1))[None]  # (1, C, H_feat, W_feat)
    emb = (emb * ratio).astype(x.dtype)
    return x + emb


# --------------------------------------------------------------------------- #
#  Bilinear resize with align_corners=True (jax.image.resize defaults differ).
# --------------------------------------------------------------------------- #


def _bilinear_align_corners(x: jnp.ndarray, out_h: int, out_w: int) -> jnp.ndarray:
    """Bilinear upsample NCHW with PyTorch's align_corners=True convention.

    jax.image.resize implements the align_corners=False (pixel-centered)
    sampling — not compatible with the DPT reference which explicitly uses
    align_corners=True. We implement it directly via fancy indexing.
    """
    N, C, H, W = x.shape
    if H == out_h and W == out_w:
        return x

    scale_h = (H - 1) / max(out_h - 1, 1) if out_h > 1 else 0.0
    scale_w = (W - 1) / max(out_w - 1, 1) if out_w > 1 else 0.0
    y_src = jnp.arange(out_h, dtype=jnp.float32) * scale_h
    x_src = jnp.arange(out_w, dtype=jnp.float32) * scale_w

    y0 = jnp.floor(y_src).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, H - 1)
    x0 = jnp.floor(x_src).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W - 1)
    yf = (y_src - y0.astype(jnp.float32))[:, None]  # (out_h, 1)
    xf = (x_src - x0.astype(jnp.float32))[None, :]  # (1, out_w)

    y0_b = y0[:, None]
    y1_b = y1[:, None]
    x0_b = x0[None, :]
    x1_b = x1[None, :]
    tl = x[:, :, y0_b, x0_b]
    tr = x[:, :, y0_b, x1_b]
    bl = x[:, :, y1_b, x0_b]
    br = x[:, :, y1_b, x1_b]
    top = tl * (1 - xf) + tr * xf
    bot = bl * (1 - xf) + br * xf
    return top * (1 - yf) + bot * yf


# --------------------------------------------------------------------------- #
#  activate_head: inv_log on pts3d, expp1 on conf (head_act.py:52-103).
# --------------------------------------------------------------------------- #


def _inv_log_transform(y: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(y) * jnp.expm1(jnp.abs(y))


def _activate_head_inv_log_expp1(
    out: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """out: (N, 4, H, W). Returns (pts3d (N, H, W, 3), conf (N, H, W))."""
    fmap = jnp.transpose(out, (0, 2, 3, 1))  # (N, H, W, 4)
    xyz = fmap[..., :-1]
    conf = fmap[..., -1]
    pts3d = _inv_log_transform(xyz)
    conf_out = 1.0 + jnp.exp(conf)
    return pts3d, conf_out


# --------------------------------------------------------------------------- #
#  Residual conv unit
# --------------------------------------------------------------------------- #


class _ResidualConvUnit(nn.Module):
    """ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> **skip-add-with-relu'd-x**.

    The reference (``streamvggt.heads.dpt_head.ResidualConvUnit``) does:
        out = self.activation(x)   # nn.ReLU(inplace=True) -- MUTATES x in place
        out = self.conv1(out); out = self.activation(out); out = self.conv2(out)
        return self.skip_add.add(out, x)   # x is now the relu'd version

    The in-place ReLU is a silent side effect: the residual branch literally
    adds ``out + relu(x)``, not ``out + x``. We reproduce that semantics
    explicitly with ``relu(x) + h``. Missing this was the Step-5 parity bug.
    """

    features: int

    def setup(self):
        self.conv1 = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            name="conv1",
        )
        self.conv2 = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            name="conv2",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # NCHW throughout; Flax convs are NHWC so we transpose in / out.
        x_relu = jax.nn.relu(
            x
        )  # also the "x" that skip-add uses (in-place side effect)
        h = jnp.transpose(x_relu, (0, 2, 3, 1))
        h = self.conv1(h)
        h = jnp.transpose(h, (0, 3, 1, 2))
        h = jax.nn.relu(h)
        h = jnp.transpose(h, (0, 2, 3, 1))
        h = self.conv2(h)
        h = jnp.transpose(h, (0, 3, 1, 2))
        return x_relu + h


# --------------------------------------------------------------------------- #
#  Feature fusion block
# --------------------------------------------------------------------------- #


class _FeatureFusionBlock(nn.Module):
    """FFB: optional resConfUnit1(skip) + resConfUnit2(running) + 2x up + 1x1 out."""

    features: int
    has_residual: bool

    def setup(self):
        if self.has_residual:
            self.resConfUnit1 = _ResidualConvUnit(self.features, name="resConfUnit1")
        self.resConfUnit2 = _ResidualConvUnit(self.features, name="resConfUnit2")
        self.out_conv = nn.Conv(
            self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            name="out_conv",
        )

    def __call__(
        self,
        running: jnp.ndarray,
        skip: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        if self.has_residual:
            assert skip is not None
            res = self.resConfUnit1(skip)
            running = running + res
        running = self.resConfUnit2(running)

        if size is None:
            out_h = running.shape[2] * 2
            out_w = running.shape[3] * 2
        else:
            out_h, out_w = size
        running = _bilinear_align_corners(running, out_h, out_w)

        running = jnp.transpose(running, (0, 2, 3, 1))
        running = self.out_conv(running)
        running = jnp.transpose(running, (0, 3, 1, 2))
        return running


# --------------------------------------------------------------------------- #
#  Scratch (rn layers + refinenets + output_conv1)
# --------------------------------------------------------------------------- #


class _Scratch(nn.Module):
    features: int = _FEATURES
    in_channels: tuple[int, int, int, int] = _OUT_CHANNELS

    def setup(self):
        # layer{i}_rn: Conv3x3 stride=1 padding=1, bias=False
        self.layer1_rn = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )
        self.layer2_rn = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )
        self.layer3_rn = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )
        self.layer4_rn = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )
        # refinenet4 has has_residual=False (only the running branch, no skip).
        self.refinenet4 = _FeatureFusionBlock(
            self.features, has_residual=False, name="refinenet4"
        )
        self.refinenet3 = _FeatureFusionBlock(
            self.features, has_residual=True, name="refinenet3"
        )
        self.refinenet2 = _FeatureFusionBlock(
            self.features, has_residual=True, name="refinenet2"
        )
        self.refinenet1 = _FeatureFusionBlock(
            self.features, has_residual=True, name="refinenet1"
        )
        # output_conv1: Conv3x3 features -> features//2  (256 -> 128)
        self.output_conv1 = nn.Conv(
            self.features // 2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
        )
        # output_conv2 is a Sequential(Conv3x3 128->32, ReLU, Conv1x1 32->out_dim=4).
        # We store as two separate submodules named output_conv2_0 / output_conv2_2.
        self.output_conv2_0 = nn.Conv(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
        )
        self.output_conv2_2 = nn.Conv(
            4,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
        )

    def _nchw_conv(self, conv: nn.Conv, x: jnp.ndarray) -> jnp.ndarray:
        """Helper: NCHW -> NHWC for Flax Conv -> NCHW back."""
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = conv(x)
        return jnp.transpose(x, (0, 3, 1, 2))

    def fuse(self, features: list[jnp.ndarray]) -> jnp.ndarray:
        layer_1, layer_2, layer_3, layer_4 = features

        l1 = self._nchw_conv(self.layer1_rn, layer_1)
        l2 = self._nchw_conv(self.layer2_rn, layer_2)
        l3 = self._nchw_conv(self.layer3_rn, layer_3)
        l4 = self._nchw_conv(self.layer4_rn, layer_4)

        out = self.refinenet4(l4, size=(l3.shape[2], l3.shape[3]))
        out = self.refinenet3(out, l3, size=(l2.shape[2], l2.shape[3]))
        out = self.refinenet2(out, l2, size=(l1.shape[2], l1.shape[3]))
        out = self.refinenet1(out, l1)
        out = self._nchw_conv(self.output_conv1, out)
        return out

    def head(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply output_conv2 (Conv3x3 -> ReLU -> Conv1x1)."""
        x = self._nchw_conv(self.output_conv2_0, x)
        x = jax.nn.relu(x)
        x = self._nchw_conv(self.output_conv2_2, x)
        return x


# --------------------------------------------------------------------------- #
#  DPTHead
# --------------------------------------------------------------------------- #


class DPTHead(nn.Module):
    """DPT decoder head producing (pts3d, conf) from aggregator tokens.

    Configured for the point head: ``output_dim=4``, ``activation="inv_log"``,
    ``conf_activation="expp1"``, ``pos_embed=True`` (matches the trained
    checkpoint — adds a fixed sinusoidal embedding at each of the 4 resized
    features and again after the final upsample).
    """

    patch_size: int = 14
    features: int = _FEATURES
    out_channels: tuple[int, int, int, int] = _OUT_CHANNELS
    intermediate_layer_idx: tuple[int, ...] = _DEFAULT_INTERMEDIATE_LAYER_IDX
    dim_in: int = 2048
    pos_embed: bool = True

    def setup(self):
        self.norm = nn.LayerNorm(epsilon=1e-5, name="norm")

        # projects: 1x1 Conv per scale
        self.projects_0 = nn.Conv(
            self.out_channels[0],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
        )
        self.projects_1 = nn.Conv(
            self.out_channels[1],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
        )
        self.projects_2 = nn.Conv(
            self.out_channels[2],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
        )
        self.projects_3 = nn.Conv(
            self.out_channels[3],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
        )

        # resize_layers: ConvTranspose (idx 0,1), Identity (idx 2 -> skip), Conv 3x3 stride 2 (idx 3)
        self.resize_layers_0 = nn.ConvTranspose(
            self.out_channels[0],
            kernel_size=(4, 4),
            strides=(4, 4),
            padding="VALID",
            use_bias=True,
        )
        self.resize_layers_1 = nn.ConvTranspose(
            self.out_channels[1],
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="VALID",
            use_bias=True,
        )
        # resize_layers_2 is Identity — no submodule
        self.resize_layers_3 = nn.Conv(
            self.out_channels[3],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=True,
        )

        self.scratch = _Scratch(
            features=self.features, in_channels=self.out_channels, name="scratch"
        )

    def __call__(
        self,
        aggregated_tokens_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            aggregated_tokens_list: length ``depth`` list of (B, S, P, dim_in).
            images: (B, S, 3, H, W) — used only for target H, W, batch shape.
            patch_start_idx: Starting index of patch tokens (5 for the aggregator).

        Returns:
            (pts3d, conf) with shapes (B, S, H, W, 3) and (B, S, H, W).
        """
        B, S, _, H, W = images.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        features: list[jnp.ndarray] = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][
                :, :, patch_start_idx:
            ]  # (B, S, P_patch, dim_in)
            x = x.reshape(B * S, patch_h * patch_w, x.shape[-1])
            x = self.norm(x)
            # -> (B*S, dim_in, patch_h, patch_w)
            x = jnp.transpose(x, (0, 2, 1)).reshape(
                B * S, x.shape[-1], patch_h, patch_w
            )
            # projects -> (optional) pos_embed -> resize_layers
            x = jnp.transpose(x, (0, 2, 3, 1))
            if dpt_idx == 0:
                x = self.projects_0(x)
            elif dpt_idx == 1:
                x = self.projects_1(x)
            elif dpt_idx == 2:
                x = self.projects_2(x)
            elif dpt_idx == 3:
                x = self.projects_3(x)
            x = jnp.transpose(x, (0, 3, 1, 2))  # back to NCHW

            if self.pos_embed:
                x = _apply_pos_embed(x, W, H)

            # resize_layers
            if dpt_idx == 0:
                x = jnp.transpose(x, (0, 2, 3, 1))
                x = self.resize_layers_0(x)
                x = jnp.transpose(x, (0, 3, 1, 2))
            elif dpt_idx == 1:
                x = jnp.transpose(x, (0, 2, 3, 1))
                x = self.resize_layers_1(x)
                x = jnp.transpose(x, (0, 3, 1, 2))
            elif dpt_idx == 2:
                pass  # identity
            elif dpt_idx == 3:
                x = jnp.transpose(x, (0, 2, 3, 1))
                x = self.resize_layers_3(x)
                x = jnp.transpose(x, (0, 3, 1, 2))
            features.append(x)

        fused = self.scratch.fuse(features)
        # Upsample to image resolution.
        fused = _bilinear_align_corners(fused, H, W)
        if self.pos_embed:
            fused = _apply_pos_embed(fused, W, H)
        out = self.scratch.head(fused)  # (B*S, 4, H, W)

        pts3d, conf = _activate_head_inv_log_expp1(out)
        pts3d = pts3d.reshape(B, S, H, W, 3)
        conf = conf.reshape(B, S, H, W)
        return pts3d, conf
