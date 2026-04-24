"""Parity tests for the JAX StreamVGGT reimplementation.

This file implements Levels 1-3 from the plan:

- **Level 1** (Step 1): weight-transfer correctness.
- **Level 2** (Steps 1.5-5): per-layer numerical equivalence, fp32 reference.
  Step 1.5 covers the shared block infrastructure via a single aggregator
  ``frame_blocks.0`` parity check which exercises Attention+LayerNorm+MLP+
  LayerScale+RoPE in one shot.

Subsequent levels (streaming, integration) land as later steps come online.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax

# JAX on Ampere+ GPUs (H100) defaults to TF32 matmul precision, which caps
# accumulated fp32 errors around 1e-3 after a few layers. PyTorch on CPU (our
# parity baseline) runs full fp32. Force highest precision for parity tests so
# the tolerance ladder reflects algorithmic equivalence, not TF32 vs fp32.
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from modules.vggt.jax.weight_transfer import (
    V1_EXCLUDE_PREFIXES,
    count_leaves,
    load_checkpoint,
    load_pytorch_weights,
    sum_numel,
    verify_coverage,
    verify_per_leaf_roundtrip,
)


# Downloading the 1.26B-param HF checkpoint is slow on first run; cache the
# numpy state_dict to a scratch path so repeated test runs are quick.
_CACHE_DIR = Path(os.environ.get("VGGT_TEST_CACHE", "/tmp/vggt_test_cache"))
_CACHE_NPZ = _CACHE_DIR / "streamvggt_state_dict.npz"


@pytest.fixture(scope="module")
def state_dict() -> dict[str, np.ndarray]:
    """Cached numpy state_dict from the HF StreamVGGT checkpoint."""
    if _CACHE_NPZ.exists():
        with np.load(_CACHE_NPZ, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    sd = load_checkpoint()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_CACHE_NPZ, **sd)
    return sd


# --------------------------------------------------------------------------- #
#  Level 1
# --------------------------------------------------------------------------- #


class TestLevel1WeightTransfer:
    """Exit criterion: every in-scope key mapped, per-leaf roundtrip exact."""

    def test_coverage(self, state_dict):
        _tree, report = load_pytorch_weights(state_dict, include_v1_only=True)
        verify_coverage(state_dict, report)

    def test_no_v1_keys_unmapped(self, state_dict):
        _tree, report = load_pytorch_weights(state_dict, include_v1_only=True)
        assert report["unmapped"] == [], (
            f"Unmapped keys: {report['unmapped'][:10]}"
        )

    def test_v1_skips_depth_and_track(self, state_dict):
        _tree, report = load_pytorch_weights(state_dict, include_v1_only=True)
        for k in report["skipped"]:
            assert k.startswith(V1_EXCLUDE_PREFIXES), k
        # Every out-of-scope key was indeed skipped (not silently mapped)
        out_of_scope = [k for k in state_dict if k.startswith(V1_EXCLUDE_PREFIXES)]
        assert set(report["skipped"]) == set(out_of_scope)

    def test_per_leaf_exact_roundtrip(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        # Bit-exact: atol=0 enforced inside verify_per_leaf_roundtrip via array_equal.
        verify_per_leaf_roundtrip(state_dict, tree)

    def test_param_count_matches(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        in_scope_sum = sum(
            int(t.size) for k, t in state_dict.items() if not k.startswith(V1_EXCLUDE_PREFIXES)
        )
        assert sum_numel(tree) == in_scope_sum

    def test_leaf_count_matches(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        in_scope_n = sum(1 for k in state_dict if not k.startswith(V1_EXCLUDE_PREFIXES))
        assert count_leaves(tree) == in_scope_n

    def test_v1_expected_tensor_count(self, state_dict):
        # From the checkpoint inventory: aggregator (1210) + camera_head (69)
        # + point_head (62) = 1341 tensors in v1 scope. This guards against
        # silently dropping keys if the checkpoint structure shifts upstream.
        in_scope_n = sum(1 for k in state_dict if not k.startswith(V1_EXCLUDE_PREFIXES))
        assert in_scope_n == 1341, f"expected 1341 v1 tensors, got {in_scope_n}"


# --------------------------------------------------------------------------- #
#  Transposition unit tests (no checkpoint needed)
# --------------------------------------------------------------------------- #


class TestTranspositionRules:
    """Cheap property-style tests for the four transposition families."""

    def test_conv2d_layout(self):
        from modules.vggt.jax.weight_transfer import _conv2d_to_flax

        pt = np.random.randn(8, 4, 3, 3).astype(np.float32)  # (O, I, H, W)
        jx = _conv2d_to_flax(pt)
        assert jx.shape == (3, 3, 4, 8)
        # element at (kh, kw, i, o) must equal source (o, i, kh, kw)
        for o in range(8):
            for i in range(4):
                for kh in range(3):
                    for kw in range(3):
                        assert jx[kh, kw, i, o] == pt[o, i, kh, kw]

    def test_conv_transpose2d_layout(self):
        from modules.vggt.jax.weight_transfer import _conv_transpose2d_to_flax

        # Use H=W=3 to catch spatial-axis flip (2x2 happens to be symmetric-ish).
        pt = np.random.randn(4, 8, 3, 3).astype(np.float32)  # (I, O, H, W)
        jx = _conv_transpose2d_to_flax(pt)
        # Flax nn.ConvTranspose uses (H, W, in, out) layout (same as nn.Conv),
        # but requires spatially-flipped kernel vs PyTorch's ConvTranspose2d.
        assert jx.shape == (3, 3, 4, 8)
        H, W = 3, 3
        for i in range(4):
            for o in range(8):
                for kh in range(H):
                    for kw in range(W):
                        # Spatial flip: jax[kh, kw] corresponds to pt[:, :, H-1-kh, W-1-kw].
                        assert jx[kh, kw, i, o] == pt[i, o, H - 1 - kh, W - 1 - kw]

    def test_linear_layout(self):
        from modules.vggt.jax.weight_transfer import _linear_to_flax

        pt = np.random.randn(32, 8).astype(np.float32)  # (O, I)
        jx = _linear_to_flax(pt)
        assert jx.shape == (8, 32)
        np.testing.assert_array_equal(jx, pt.T)


# --------------------------------------------------------------------------- #
#  Level 2 -- single aggregator frame_block parity
# --------------------------------------------------------------------------- #


# ATOL from the repo's established ladder. The per-block budget at fp32 is
# "single attention block: atol <= 1e-4" per the plan's Success Criteria.
ATOL_SINGLE_BLOCK_FP32 = 1e-4


@pytest.fixture(scope="module")
def pytorch_ivggt_module():
    """Import the PyTorch StreamVGGT layer module once (adds external to sys.path)."""
    import sys

    ext = str(
        Path(__file__).resolve().parents[3] / "external" / "InfiniteVGGT" / "src"
    )
    if ext not in sys.path:
        sys.path.insert(0, ext)
    import streamvggt  # noqa: F401
    return streamvggt


class TestLevel2SharedBlockParity:
    """Single-block parity: covers Attention, LayerNorm, MLP, LayerScale, RoPE.

    We pick ``aggregator.frame_blocks.0`` because it exercises every piece of
    shared infra (qk_norm=True, RoPE wired, LayerScale present).
    """

    @pytest.fixture(scope="class")
    def fixtures(self, state_dict, pytorch_ivggt_module):
        """Load weights, build matched PyTorch and JAX single-block modules."""
        import torch
        from streamvggt.layers.block import Block as PtBlock
        from streamvggt.layers.rope import RotaryPositionEmbedding2D

        # Aggregator config for vit_large path (see aggregator.py:50-68).
        dim = 1024
        num_heads = 16
        head_dim = dim // num_heads  # 64
        mlp_ratio = 4.0
        init_values = 0.01
        qk_norm = True
        rope_frequency = 100.0

        # --- PyTorch module ---
        pt_rope = RotaryPositionEmbedding2D(frequency=rope_frequency)
        pt_block = PtBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            init_values=init_values,
            qk_norm=qk_norm,
            rope=pt_rope,
        ).eval()

        # Load block-0 weights from the full state_dict into pt_block.
        prefix = "aggregator.frame_blocks.0."
        pt_sd = {
            k[len(prefix):]: torch.from_numpy(np.asarray(v))
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        missing, unexpected = pt_block.load_state_dict(pt_sd, strict=False)
        assert not missing, f"missing keys: {missing}"
        assert not unexpected, f"unexpected keys: {unexpected}"

        # --- JAX module ---
        from modules.vggt.jax.block import Block as JxBlock
        from modules.vggt.jax.rope import compute_1d_rope_tables
        from modules.vggt.jax.weight_transfer import load_pytorch_weights

        jx_block = JxBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qk_norm=qk_norm,
            init_values=init_values,
            norm_eps=1e-5,
        )

        full_tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        jx_params = {"params": full_tree["aggregator"]["frame_blocks_0"]}
        jx_params = jax.tree.map(jnp.asarray, jx_params)

        # --- Inputs common to both ---
        # B=1, S=1; grid 37x37 -> P = 1 (cam) + 4 (reg) + 1369 (patch) = 1374.
        rng = np.random.RandomState(1234)
        x_np = rng.randn(1, 1374, dim).astype(np.float32)

        # Build positions the same way aggregator.forward does (aggregator.py:247-254):
        # patch positions from cartesian product of [0, 37) x [0, 37), shifted by +1,
        # special tokens (first patch_start_idx=5) get zeros.
        ys, xs = np.meshgrid(np.arange(37), np.arange(37), indexing="ij")
        patch_pos = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=-1)  # (1369, 2)
        patch_pos = patch_pos + 1  # aggregator shifts by 1
        special = np.zeros((5, 2), dtype=patch_pos.dtype)
        positions_np = np.concatenate([special, patch_pos], axis=0)[None]  # (1, 1374, 2)

        return dict(
            pt_block=pt_block,
            pt_rope=pt_rope,
            jx_block=jx_block,
            jx_params=jx_params,
            x_np=x_np,
            positions_np=positions_np,
            head_dim=head_dim,
            rope_frequency=rope_frequency,
        )

    def test_single_frame_block_matches_pytorch(self, fixtures):
        import torch

        x_np = fixtures["x_np"]
        positions_np = fixtures["positions_np"]
        head_dim = fixtures["head_dim"]
        freq = fixtures["rope_frequency"]

        # --- PyTorch forward ---
        with torch.no_grad():
            pt_out = fixtures["pt_block"](
                torch.from_numpy(x_np),
                pos=torch.from_numpy(positions_np.astype(np.int64)),
            ).numpy()

        # --- JAX forward (same RoPE tables the PyTorch version would build) ---
        from modules.vggt.jax.rope import compute_1d_rope_tables

        max_pos = int(positions_np.max()) + 1  # 38
        cos_t, sin_t = compute_1d_rope_tables(
            dim=head_dim // 2,  # 2D RoPE splits per-head dim in half
            max_pos=max_pos,
            frequency=freq,
            dtype=jnp.float32,
        )
        jx_out = fixtures["jx_block"].apply(
            fixtures["jx_params"],
            jnp.asarray(x_np),
            rope_tables=(cos_t, sin_t),
            positions=jnp.asarray(positions_np),
        )
        jx_out_np = np.asarray(jx_out)

        # Shape first so mismatches are obvious before value diffs.
        assert jx_out_np.shape == pt_out.shape, (jx_out_np.shape, pt_out.shape)
        max_abs = np.max(np.abs(jx_out_np - pt_out))
        assert max_abs <= ATOL_SINGLE_BLOCK_FP32, (
            f"single-block parity: max_abs={max_abs:.3e} > {ATOL_SINGLE_BLOCK_FP32:.0e}"
        )


# --------------------------------------------------------------------------- #
#  Level 2 -- DINOv2 backbone parity (Step 2)
# --------------------------------------------------------------------------- #


ATOL_PATCH_EMBED_FP32 = 1e-5
ATOL_FULL_BACKBONE_FP32 = 1e-3  # 24 blocks of accumulated fp32 matmul drift


class TestLevel2DinoV2Backbone:
    """Step 2 parity gate: DINOv2 ViT-L/14-reg backbone at 518x518."""

    @pytest.fixture(scope="class")
    def pt_backbone(self, state_dict, pytorch_ivggt_module):
        """PyTorch DINOv2 ViT-L loaded with checkpoint weights under the same
        config the Aggregator builds (num_register_tokens=4, init_values=1.0,
        interpolate_antialias=True, interpolate_offset=0.0, block_chunks=0)."""
        import torch
        from streamvggt.layers.vision_transformer import vit_large

        model = vit_large(
            patch_size=14,
            img_size=518,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            block_chunks=0,
            init_values=1.0,
        ).eval()
        # Force naive attention path for clean parity (matches JAX manual attention).
        for blk in model.blocks:
            blk.attn.fused_attn = False

        prefix = "aggregator.patch_embed."
        pt_sd = {
            k[len(prefix):]: torch.from_numpy(np.asarray(v))
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        missing, unexpected = model.load_state_dict(pt_sd, strict=False)
        # mask_token is loaded; prepare_tokens_with_masks skips it when masks is None.
        assert not missing, f"missing: {missing}"
        assert not unexpected, f"unexpected: {unexpected}"
        return model

    @pytest.fixture(scope="class")
    def jx_params(self, state_dict):
        """JAX param tree for the DinoV2Backbone sub-module."""
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        return jax.tree.map(jnp.asarray, {"params": tree["aggregator"]["patch_embed"]})

    @pytest.fixture(scope="class")
    def sample_image(self):
        """Random normalized NCHW image matching the aggregator's pipeline output."""
        rng = np.random.RandomState(4242)
        # Values in roughly the range of a normalized image.
        return rng.randn(1, 3, 518, 518).astype(np.float32)

    def test_patch_embed_conv_matches_pytorch(
        self, pt_backbone, jx_params, sample_image
    ):
        """Inner 14x14 Conv patch embed only (before cls/pos/register)."""
        import torch

        with torch.no_grad():
            # PyTorch reference: `model.patch_embed(images)` returns flattened
            # (B, num_patches, embed_dim). We match that output shape.
            pt_out = pt_backbone.patch_embed(torch.from_numpy(sample_image)).numpy()

        from modules.vggt.jax.backbone import PatchEmbed

        pe_params = {"params": jx_params["params"]["patch_embed"]}
        # NCHW -> NHWC for Flax Conv
        x_nhwc = jnp.asarray(np.transpose(sample_image, (0, 2, 3, 1)))
        jx_conv = PatchEmbed(embed_dim=1024, patch_size=14).apply(pe_params, x_nhwc)
        # Flatten spatial dims to match PyTorch's (B, num_patches, embed_dim)
        jx_out = np.asarray(jx_conv).reshape(1, -1, 1024)

        assert jx_out.shape == pt_out.shape, (jx_out.shape, pt_out.shape)
        max_abs = np.max(np.abs(jx_out - pt_out))
        assert max_abs <= ATOL_PATCH_EMBED_FP32, (
            f"patch-embed parity: max_abs={max_abs:.3e} > {ATOL_PATCH_EMBED_FP32:.0e}"
        )

    def test_full_backbone_matches_pytorch(
        self, pt_backbone, jx_params, sample_image
    ):
        """Full DINOv2 backbone (24 blocks + final norm), patch-tokens slice."""
        import torch

        with torch.no_grad():
            pt_dict = pt_backbone(torch.from_numpy(sample_image), is_training=True)
        pt_patch_tokens = pt_dict["x_norm_patchtokens"].numpy()  # (B, num_patches, 1024)

        from modules.vggt.jax.backbone import DinoV2Backbone

        jx_out = DinoV2Backbone().apply(jx_params, jnp.asarray(sample_image))
        jx_out_np = np.asarray(jx_out)

        assert jx_out_np.shape == pt_patch_tokens.shape, (
            jx_out_np.shape,
            pt_patch_tokens.shape,
        )
        max_abs = np.max(np.abs(jx_out_np - pt_patch_tokens))
        assert max_abs <= ATOL_FULL_BACKBONE_FP32, (
            f"full-backbone parity: max_abs={max_abs:.3e} > {ATOL_FULL_BACKBONE_FP32:.0e}"
        )


# --------------------------------------------------------------------------- #
#  Level 2 -- aggregator no-cache parity (Step 3)
# --------------------------------------------------------------------------- #


ATOL_AGGREGATOR_NO_CACHE_FP32 = 1e-3


class TestLevel2AggregatorNoCache:
    """Step 3 parity gate: full aggregator forward without KV-cache.

    Uses S=2 so the global attention's causal frame mask is actually exercised
    (S=1 would leave it a no-op).
    """

    @pytest.fixture(scope="class")
    def pt_aggregator(self, state_dict, pytorch_ivggt_module):
        import torch
        from streamvggt.models.aggregator import Aggregator as PtAggregator

        pt_agg = PtAggregator(img_size=518, patch_size=14, embed_dim=1024).eval()
        # Force naive attention in every block (DINOv2 patch_embed + frame + global).
        for blk in pt_agg.patch_embed.blocks:
            blk.attn.fused_attn = False
        for blk in pt_agg.frame_blocks:
            blk.attn.fused_attn = False
        for blk in pt_agg.global_blocks:
            blk.attn.fused_attn = False

        prefix = "aggregator."
        pt_sd = {
            k[len(prefix):]: torch.from_numpy(np.asarray(v))
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        missing, unexpected = pt_agg.load_state_dict(pt_sd, strict=False)
        assert not missing, f"missing: {missing}"
        assert not unexpected, f"unexpected: {unexpected}"
        return pt_agg

    @pytest.fixture(scope="class")
    def jx_params(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        return jax.tree.map(jnp.asarray, {"params": tree["aggregator"]})

    @pytest.fixture(scope="class")
    def sample_frames(self):
        # B=1, S=2 frames, 518x518, values in [0, 1] like the aggregator expects.
        rng = np.random.RandomState(7)
        return rng.uniform(0.0, 1.0, size=(1, 2, 3, 518, 518)).astype(np.float32)

    def test_aggregator_last_level_matches_pytorch(
        self, pt_aggregator, jx_params, sample_frames
    ):
        import torch
        from modules.vggt.jax.aggregator import Aggregator

        with torch.no_grad():
            pt_out_list, pt_patch_start = pt_aggregator(torch.from_numpy(sample_frames))
        pt_last = pt_out_list[-1].numpy()  # (B, S, P, 2*C)

        jx_out_list, jx_patch_start = Aggregator().apply(
            jx_params, jnp.asarray(sample_frames)
        )
        assert jx_patch_start == pt_patch_start == 5
        jx_last = np.asarray(jx_out_list[-1])

        assert jx_last.shape == pt_last.shape, (jx_last.shape, pt_last.shape)
        max_abs = np.max(np.abs(jx_last - pt_last))
        assert max_abs <= ATOL_AGGREGATOR_NO_CACHE_FP32, (
            f"aggregator no-cache parity: max_abs={max_abs:.3e} "
            f"> {ATOL_AGGREGATOR_NO_CACHE_FP32:.0e}"
        )


# --------------------------------------------------------------------------- #
#  Level 2 -- camera head parity (Step 4)
# --------------------------------------------------------------------------- #


ATOL_CAMERA_HEAD_FP32 = 1e-3


class TestLevel2CameraHead:
    """Step 4 parity gate: iterative AdaLN camera head on synthetic tokens.

    Random ``aggregated_tokens_list`` suffices because the head is a pure
    function of its input; we skip the 3-minute aggregator forward here.
    """

    @pytest.fixture(scope="class")
    def pt_camera_head(self, state_dict, pytorch_ivggt_module):
        import torch
        from streamvggt.heads.camera_head import CameraHead as PtCameraHead

        head = PtCameraHead(dim_in=2048).eval()
        for blk in head.trunk:
            blk.attn.fused_attn = False
        prefix = "camera_head."
        pt_sd = {
            k[len(prefix):]: torch.from_numpy(np.asarray(v))
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        missing, unexpected = head.load_state_dict(pt_sd, strict=False)
        assert not missing, f"missing: {missing}"
        assert not unexpected, f"unexpected: {unexpected}"
        return head

    @pytest.fixture(scope="class")
    def jx_params(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        return jax.tree.map(jnp.asarray, {"params": tree["camera_head"]})

    @pytest.fixture(scope="class")
    def aggregated_tokens(self):
        # (B=1, S=2, P=1374, dim_in=2048) — realistic scale.
        rng = np.random.RandomState(17)
        last = rng.randn(1, 2, 1374, 2048).astype(np.float32) * 0.1
        # head only uses the last list element; pass a singleton list.
        return last

    def test_camera_head_last_iter_matches_pytorch(
        self, pt_camera_head, jx_params, aggregated_tokens
    ):
        import torch
        from modules.vggt.jax.heads.camera_head import CameraHead

        pt_tokens = torch.from_numpy(aggregated_tokens)
        with torch.no_grad():
            pt_list = pt_camera_head([pt_tokens])
        pt_last = pt_list[-1].numpy()  # (B, S, 9)

        jx_list = CameraHead().apply(jx_params, [jnp.asarray(aggregated_tokens)])
        jx_last = np.asarray(jx_list[-1])

        assert jx_last.shape == pt_last.shape, (jx_last.shape, pt_last.shape)
        max_abs = np.max(np.abs(jx_last - pt_last))
        assert max_abs <= ATOL_CAMERA_HEAD_FP32, (
            f"camera-head parity: max_abs={max_abs:.3e} > {ATOL_CAMERA_HEAD_FP32:.0e}"
        )


# --------------------------------------------------------------------------- #
#  Level 2 -- DPT point head parity (Step 5)
# --------------------------------------------------------------------------- #


ATOL_POINT_HEAD_FP32 = 1e-3


class TestLevel2PointHead:
    """Step 5 parity gate: DPT decoder producing pts3d + conf.

    Random aggregated_tokens_list (24 entries, only indices 4/11/17/23 are
    actually read by the head) and a dummy image tensor are enough here: the
    head is a pure function of those inputs, so we skip the 3-minute
    aggregator forward.
    """

    @pytest.fixture(scope="class")
    def pt_point_head(self, state_dict, pytorch_ivggt_module):
        import torch
        from streamvggt.heads.dpt_head import DPTHead as PtDPTHead

        head = PtDPTHead(
            dim_in=2048,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1",
            pos_embed=True,
        ).eval()
        prefix = "point_head."
        pt_sd = {
            k[len(prefix):]: torch.from_numpy(np.asarray(v))
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        missing, unexpected = head.load_state_dict(pt_sd, strict=False)
        assert not missing, f"missing: {missing}"
        assert not unexpected, f"unexpected: {unexpected}"
        return head

    @pytest.fixture(scope="class")
    def jx_params(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        return jax.tree.map(jnp.asarray, {"params": tree["point_head"]})

    @pytest.fixture(scope="class")
    def synthetic_inputs(self):
        # 24 aggregated levels, only 4/11/17/23 are consumed.
        rng = np.random.RandomState(101)
        B, S, P, dim_in = 1, 1, 1374, 2048  # 5 special + 1369 patch
        agg_list = [
            rng.randn(B, S, P, dim_in).astype(np.float32) * 0.1
            for _ in range(24)
        ]
        # Images are only used for H, W, B, S in the DPT forward.
        images = np.zeros((B, S, 3, 518, 518), dtype=np.float32)
        return agg_list, images

    def test_point_head_matches_pytorch(
        self, pt_point_head, jx_params, synthetic_inputs
    ):
        import torch
        from modules.vggt.jax.heads.dpt_head import DPTHead

        agg_list, images = synthetic_inputs
        pt_agg = [torch.from_numpy(x) for x in agg_list]
        with torch.no_grad():
            pt_pts3d, pt_conf = pt_point_head(
                pt_agg, images=torch.from_numpy(images), patch_start_idx=5
            )
        pt_pts3d_np = pt_pts3d.numpy()
        pt_conf_np = pt_conf.numpy()

        jx_pts3d, jx_conf = DPTHead().apply(
            jx_params,
            [jnp.asarray(x) for x in agg_list],
            jnp.asarray(images),
            5,
        )
        jx_pts3d_np = np.asarray(jx_pts3d)
        jx_conf_np = np.asarray(jx_conf)

        assert jx_pts3d_np.shape == pt_pts3d_np.shape
        assert jx_conf_np.shape == pt_conf_np.shape
        max_abs_pts = np.max(np.abs(jx_pts3d_np - pt_pts3d_np))
        max_abs_conf = np.max(np.abs(jx_conf_np - pt_conf_np))
        assert max_abs_pts <= ATOL_POINT_HEAD_FP32, (
            f"point-head pts3d: max_abs={max_abs_pts:.3e} > {ATOL_POINT_HEAD_FP32:.0e}"
        )
        assert max_abs_conf <= ATOL_POINT_HEAD_FP32, (
            f"point-head conf: max_abs={max_abs_conf:.3e} > {ATOL_POINT_HEAD_FP32:.0e}"
        )


# --------------------------------------------------------------------------- #
#  Level 3 -- streaming cache parity (Step 6a: eviction disabled)
# --------------------------------------------------------------------------- #


# Cache-mode parity bound. The cache path rearranges the order of arithmetic
# (each frame's global attention now uses a rectangular Q x [past_K | new_K]
# matmul instead of the block-diagonal-causal S*P x S*P version) so rounding
# patterns differ slightly. 1e-3 fp32 matches the aggregator-no-cache bound.
ATOL_AGGREGATOR_CACHE_FP32 = 1e-3
ATOL_CAMERA_CACHE_FP32 = 1e-3

# Frames used across Level-3 tests. Keep small so PyTorch fp32 / no-fused
# attention completes quickly; Step 6c will add 500-frame sequences once
# eviction is wired up to keep memory bounded.
_L3_NUM_FRAMES = 3


class TestLevel3AggregatorCache:
    """Step 6a gate: aggregator streaming with eviction disabled.

    Two independent checks:

    1. **JAX self-consistency** — running the cache path frame-by-frame should
       reproduce the no-cache path applied to all frames at once. This catches
       bugs in cache threading without involving PyTorch.
    2. **JAX vs PyTorch cache** — mirrors the production streaming path. Uses
       ``total_budget`` large enough that the reference's eviction never fires.
    """

    @pytest.fixture(scope="class")
    def pt_aggregator(self, state_dict, pytorch_ivggt_module):
        import torch
        from streamvggt.models.aggregator import Aggregator as PtAggregator

        pt_agg = PtAggregator(img_size=518, patch_size=14, embed_dim=1024).eval()
        for blk in pt_agg.patch_embed.blocks:
            blk.attn.fused_attn = False
        for blk in pt_agg.frame_blocks:
            blk.attn.fused_attn = False
        for blk in pt_agg.global_blocks:
            blk.attn.fused_attn = False

        prefix = "aggregator."
        pt_sd = {
            k[len(prefix):]: torch.from_numpy(np.asarray(v))
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        missing, unexpected = pt_agg.load_state_dict(pt_sd, strict=False)
        assert not missing, f"missing: {missing}"
        assert not unexpected, f"unexpected: {unexpected}"
        return pt_agg

    @pytest.fixture(scope="class")
    def jx_params(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        return jax.tree.map(jnp.asarray, {"params": tree["aggregator"]})

    @pytest.fixture(scope="class")
    def frames(self):
        # N frames of (3, 518, 518) each; returned as numpy (N, 3, 518, 518).
        rng = np.random.RandomState(11)
        return rng.uniform(
            0.0, 1.0, size=(_L3_NUM_FRAMES, 3, 518, 518)
        ).astype(np.float32)

    # --------------------------------------------------------------------- #
    #  (1) JAX cache vs JAX no-cache
    # --------------------------------------------------------------------- #
    def test_jax_cache_matches_jax_nocache(self, jx_params, frames):
        from modules.vggt.jax.aggregator import Aggregator

        # No-cache: process all frames at once with shape (1, N, 3, 518, 518).
        all_frames = frames[None]  # (1, N, 3, H, W)
        nc_out_list, _ = Aggregator().apply(jx_params, jnp.asarray(all_frames))
        nc_last = np.asarray(nc_out_list[-1])  # (1, N, P, 2C)

        # Cache: loop frame-by-frame with S=1 each.
        past_kvs = None
        last_scores = None
        cached_last_per_frame = []
        for i in range(frames.shape[0]):
            one = jnp.asarray(frames[i : i + 1][None])  # (1, 1, 3, H, W)
            out_list, _, past_kvs, last_scores = Aggregator().apply(
                jx_params,
                one,
                use_cache=True,
                past_kvs=past_kvs,
                past_frame_idx=i,
                last_scores=last_scores,
            )
            cached_last_per_frame.append(np.asarray(out_list[-1]))
        cached_last = np.concatenate(cached_last_per_frame, axis=1)  # (1, N, P, 2C)

        assert cached_last.shape == nc_last.shape
        max_abs = np.max(np.abs(cached_last - nc_last))
        assert max_abs <= ATOL_AGGREGATOR_CACHE_FP32, (
            f"JAX cache vs JAX no-cache: max_abs={max_abs:.3e} "
            f"> {ATOL_AGGREGATOR_CACHE_FP32:.0e}"
        )

    # --------------------------------------------------------------------- #
    #  (2) JAX cache vs PyTorch cache
    # --------------------------------------------------------------------- #
    def test_jax_cache_matches_pytorch_cache(
        self, pt_aggregator, jx_params, frames
    ):
        import torch
        from modules.vggt.jax.aggregator import Aggregator

        # total_budget is distributed across the 24 global blocks via softmax
        # of 1-last_scores. With last_scores initialised to 0, the proportions
        # are ~uniform, so per-block budget ~= total_budget / depth. We need
        # per-block budget >= N*P to keep eviction from firing in 6a. Add 2x
        # slack to absorb softmax rounding.
        P = 5 + (518 // 14) ** 2  # 5 specials + 1369 patch tokens = 1374
        N = frames.shape[0]
        depth = pt_aggregator.depth
        total_budget = N * P * depth * 2

        # ---- PyTorch: loop with use_cache=True ----
        pt_past_kvs = [None] * pt_aggregator.depth
        pt_last_per_frame = []
        with torch.no_grad():
            for i in range(N):
                one = torch.from_numpy(frames[i : i + 1][None])  # (1, 1, 3, H, W)
                out_list, _, pt_past_kvs = pt_aggregator(
                    one,
                    past_key_values=pt_past_kvs,
                    use_cache=True,
                    past_frame_idx=i,
                    total_budget=total_budget,
                )
                pt_last_per_frame.append(out_list[-1].cpu().numpy())
        pt_last = np.concatenate(pt_last_per_frame, axis=1)

        # ---- JAX: loop with use_cache=True ----
        jx_past_kvs = None
        jx_last_scores = None
        jx_last_per_frame = []
        for i in range(N):
            one = jnp.asarray(frames[i : i + 1][None])
            out_list, _, jx_past_kvs, jx_last_scores = Aggregator().apply(
                jx_params,
                one,
                use_cache=True,
                past_kvs=jx_past_kvs,
                past_frame_idx=i,
                last_scores=jx_last_scores,
            )
            jx_last_per_frame.append(np.asarray(out_list[-1]))
        jx_last = np.concatenate(jx_last_per_frame, axis=1)

        assert jx_last.shape == pt_last.shape, (jx_last.shape, pt_last.shape)
        max_abs = np.max(np.abs(jx_last - pt_last))
        assert max_abs <= ATOL_AGGREGATOR_CACHE_FP32, (
            f"JAX cache vs PyTorch cache: max_abs={max_abs:.3e} "
            f"> {ATOL_AGGREGATOR_CACHE_FP32:.0e}"
        )


class TestLevel3CameraHeadCache:
    """Step 6a gate: camera head streaming cache.

    Works on the aggregator's output tokens, so we can feed it random tensors
    (like TestLevel2CameraHead) instead of running the full aggregator.
    """

    @pytest.fixture(scope="class")
    def pt_camera_head(self, state_dict, pytorch_ivggt_module):
        import torch
        from streamvggt.heads.camera_head import CameraHead as PtCamera

        head = PtCamera(dim_in=2048).eval()
        for blk in head.trunk:
            blk.attn.fused_attn = False
        prefix = "camera_head."
        pt_sd = {
            k[len(prefix):]: torch.from_numpy(np.asarray(v))
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        missing, unexpected = head.load_state_dict(pt_sd, strict=False)
        assert not missing, f"missing: {missing}"
        assert not unexpected, f"unexpected: {unexpected}"
        return head

    @pytest.fixture(scope="class")
    def jx_params(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        return jax.tree.map(jnp.asarray, {"params": tree["camera_head"]})

    @pytest.fixture(scope="class")
    def agg_tokens_per_frame(self):
        # 24 aggregator levels, N frames each with S=1. Only ``[-1]`` is used.
        rng = np.random.RandomState(29)
        B, P, C = 1, 1374, 2048
        frames = []
        for _ in range(_L3_NUM_FRAMES):
            frames.append(
                [
                    rng.randn(B, 1, P, C).astype(np.float32) * 0.1
                    for _ in range(24)
                ]
            )
        return frames

    def test_camera_head_cache_matches_pytorch(
        self, pt_camera_head, jx_params, agg_tokens_per_frame
    ):
        import torch
        from modules.vggt.jax.heads.camera_head import CameraHead

        # ---- PyTorch cache loop ----
        pt_cache = [None] * pt_camera_head.trunk_depth
        pt_poses = []
        with torch.no_grad():
            for frame_tokens in agg_tokens_per_frame:
                pt_input = [torch.from_numpy(x) for x in frame_tokens]
                pose_list, pt_cache = pt_camera_head(
                    pt_input, past_key_values_camera=pt_cache, use_cache=True
                )
                pt_poses.append(pose_list[-1].cpu().numpy())
        pt_poses = np.concatenate(pt_poses, axis=1)  # (B, N, 9)

        # ---- JAX cache loop ----
        jx_cache = None
        jx_poses = []
        for frame_tokens in agg_tokens_per_frame:
            jx_input = [jnp.asarray(x) for x in frame_tokens]
            pose_list, jx_cache = CameraHead().apply(
                jx_params,
                jx_input,
                use_cache=True,
                past_kvs_camera=jx_cache,
            )
            jx_poses.append(np.asarray(pose_list[-1]))
        jx_poses = np.concatenate(jx_poses, axis=1)

        assert jx_poses.shape == pt_poses.shape
        max_abs = np.max(np.abs(jx_poses - pt_poses))
        assert max_abs <= ATOL_CAMERA_CACHE_FP32, (
            f"camera-head cache: max_abs={max_abs:.3e} "
            f"> {ATOL_CAMERA_CACHE_FP32:.0e}"
        )


# --------------------------------------------------------------------------- #
#  Level 3 -- eviction parity (Step 6b: static uniform budget)
# --------------------------------------------------------------------------- #


class TestLevel3Eviction:
    """Step 6b gate: small-budget uniform eviction.

    Runs exactly 4 frames so eviction fires **once** on the last frame:
      * frames 0, 1, 2: pre-evict cache <= 3*P, no eviction fires.
      * frame 3: pre-evict cache = 4*P > 3*P, eviction prunes to 3*P.

    Stopping at 4 frames is deliberate: the reference updates its
    ``self.last_scores`` buffer after eviction, after which subsequent
    frames compute a NON-uniform per-block budget via softmax. That
    dynamic-budget path lands in Step 6c. For 6b we verify that the
    uniform branch (last_scores still zero on frame 3) matches PyTorch
    exactly.

    The first *P* tokens (frame 0) are anchors — never evicted. The
    candidate scoring + top_k(-scores) path must match between PyTorch and
    JAX for the cache contents to agree on which tokens were retained.
    """

    @pytest.fixture(scope="class")
    def pt_aggregator(self, state_dict, pytorch_ivggt_module):
        import torch
        from streamvggt.models.aggregator import Aggregator as PtAggregator

        pt_agg = PtAggregator(img_size=518, patch_size=14, embed_dim=1024).eval()
        for blk in pt_agg.patch_embed.blocks:
            blk.attn.fused_attn = False
        for blk in pt_agg.frame_blocks:
            blk.attn.fused_attn = False
        for blk in pt_agg.global_blocks:
            blk.attn.fused_attn = False

        prefix = "aggregator."
        pt_sd = {
            k[len(prefix):]: torch.from_numpy(np.asarray(v))
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        pt_agg.load_state_dict(pt_sd, strict=False)
        return pt_agg

    @pytest.fixture(scope="class")
    def jx_params(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        return jax.tree.map(jnp.asarray, {"params": tree["aggregator"]})

    @pytest.fixture(scope="class")
    def frames_and_budget(self):
        # 4 frames; eviction fires exactly once, on the last frame.
        rng = np.random.RandomState(91)
        frames = rng.uniform(0.0, 1.0, size=(4, 3, 518, 518)).astype(np.float32)
        P = 5 + (518 // 14) ** 2  # 1374
        depth = 24
        per_block = 3 * P
        total_budget = per_block * depth
        return frames, total_budget, per_block

    @pytest.fixture(scope="class")
    def pt_run(self, pt_aggregator, frames_and_budget):
        import torch

        frames, total_budget, _ = frames_and_budget
        past = [None] * pt_aggregator.depth
        outs = []
        sizes = []
        with torch.no_grad():
            for i, frame in enumerate(frames):
                one = torch.from_numpy(frame[None, None])
                out_list, _, past = pt_aggregator(
                    one,
                    past_key_values=past,
                    use_cache=True,
                    past_frame_idx=i,
                    total_budget=total_budget,
                )
                outs.append(out_list[-1].cpu().numpy())
                sizes.append(past[0][0].shape[2])
        # Convert past to numpy to release PyTorch tensors.
        past_np = [(k.cpu().numpy(), v.cpu().numpy()) for (k, v) in past]
        return outs, sizes, past_np

    @pytest.fixture(scope="class")
    def jx_run(self, jx_params, frames_and_budget):
        from modules.vggt.jax.aggregator import Aggregator

        frames, total_budget, _ = frames_and_budget
        past = None
        last_scores = None
        outs = []
        sizes = []
        for i, frame in enumerate(frames):
            one = jnp.asarray(frame[None, None])
            out_list, _, past, last_scores = Aggregator().apply(
                jx_params,
                one,
                use_cache=True,
                past_kvs=past,
                past_frame_idx=i,
                total_budget=total_budget,
                last_scores=last_scores,
            )
            outs.append(np.asarray(out_list[-1]))
            sizes.append(past[0][0].shape[2])
        past_np = [(np.asarray(k), np.asarray(v)) for (k, v) in past]
        return outs, sizes, past_np

    def test_eviction_fires_at_same_frame(self, pt_run, jx_run):
        """Both implementations should evict starting at frame 3."""
        _, pt_sizes, _ = pt_run
        _, jx_sizes, _ = jx_run
        assert pt_sizes == jx_sizes, (
            f"cache sizes diverge: PT={pt_sizes} JX={jx_sizes}"
        )
        P = 1374
        expected = [P, 2 * P, 3 * P, 3 * P]
        assert pt_sizes == expected, f"unexpected sizes: {pt_sizes}"

    def test_retained_kv_matches_pytorch(self, pt_run, jx_run):
        """After eviction, retained (K, V) must match bit-close — which is
        equivalent to ``same indices retained`` under the scoring rule."""
        _, _, pt_past = pt_run
        _, _, jx_past = jx_run

        for b in [0, 11, 23]:
            pt_k, pt_v = pt_past[b]
            jx_k, jx_v = jx_past[b]
            assert pt_k.shape == jx_k.shape, (b, pt_k.shape, jx_k.shape)
            max_abs_k = np.max(np.abs(pt_k - jx_k))
            max_abs_v = np.max(np.abs(pt_v - jx_v))
            # 3e-3 absorbs accumulated fp32-noise-through-24-blocks rounding.
            assert max_abs_k <= 3e-3, f"block {b} K mismatch: {max_abs_k:.3e}"
            assert max_abs_v <= 3e-3, f"block {b} V mismatch: {max_abs_v:.3e}"

    def test_per_frame_output_matches_pytorch(self, pt_run, jx_run):
        """Final aggregator outputs per frame within the cache tolerance."""
        pt_outs, _, _ = pt_run
        jx_outs, _, _ = jx_run
        for i, (pt, jx) in enumerate(zip(pt_outs, jx_outs)):
            max_abs = np.max(np.abs(pt - jx))
            assert max_abs <= ATOL_AGGREGATOR_CACHE_FP32, (
                f"frame {i} output: max_abs={max_abs:.3e} "
                f"> {ATOL_AGGREGATOR_CACHE_FP32:.0e}"
            )


# --------------------------------------------------------------------------- #
#  Level 3 -- dynamic budget allocation (Step 6c)
# --------------------------------------------------------------------------- #


# Plan target: dynamic-budget computation matches PyTorch at atol ≤ 1e-4
# (both sides compute in fp32).
ATOL_DYNAMIC_BUDGET_FP32 = 1e-4


class TestLevel3DynamicBudget:
    """Step 6c gate: softmax-of-diversity dynamic budget allocation.

    The reference updates ``self.last_scores`` whenever any block evicts,
    so frames 4+ receive a non-uniform budget per block. JAX must reproduce
    both the budget computation and the resulting outputs bit-close.
    """

    def test_dynamic_budget_formula_matches_pytorch(self, pytorch_ivggt_module):
        """Unit test — ``_calculate_dynamic_budgets`` vs PyTorch reference."""
        import torch
        from streamvggt.models.aggregator import Aggregator as PtAggregator
        from modules.vggt.jax.aggregator import _calculate_dynamic_budgets

        pt_agg = PtAggregator(img_size=518, patch_size=14, embed_dim=1024)
        # Fabricate a non-uniform last_scores to exercise the allocator.
        rng = np.random.RandomState(17)
        last = rng.uniform(0.0, 0.9, size=(24,)).astype(np.float32)
        total_budget = 1_200_000

        pt_agg.last_scores = torch.from_numpy(last)
        pt_budgets = pt_agg._calculate_dynamic_budgets(total_budget).cpu().numpy()

        jx_budgets = np.asarray(
            _calculate_dynamic_budgets(jnp.asarray(last), total_budget)
        )

        assert pt_budgets.shape == jx_budgets.shape == (24,)
        max_abs = np.max(np.abs(pt_budgets - jx_budgets))
        # Both sides run fp32 * int truncation; the softmax may disagree
        # by ~1 unit after round-to-zero. atol 2 absorbs that.
        assert max_abs <= 2, (
            f"dynamic-budget diff: max_abs={max_abs}, "
            f"pt={pt_budgets[:5]} jx={jx_budgets[:5]}"
        )

    @pytest.fixture(scope="class")
    def pt_aggregator(self, state_dict, pytorch_ivggt_module):
        import torch
        from streamvggt.models.aggregator import Aggregator as PtAggregator

        pt_agg = PtAggregator(img_size=518, patch_size=14, embed_dim=1024).eval()
        for blk in pt_agg.patch_embed.blocks:
            blk.attn.fused_attn = False
        for blk in pt_agg.frame_blocks:
            blk.attn.fused_attn = False
        for blk in pt_agg.global_blocks:
            blk.attn.fused_attn = False

        prefix = "aggregator."
        pt_sd = {
            k[len(prefix):]: torch.from_numpy(np.asarray(v))
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        pt_agg.load_state_dict(pt_sd, strict=False)
        return pt_agg

    @pytest.fixture(scope="class")
    def jx_params(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        return jax.tree.map(jnp.asarray, {"params": tree["aggregator"]})

    @pytest.fixture(scope="class")
    def frames_and_budget(self):
        # 6 frames: eviction fires at frame 3 (still uniform), frames 4-5
        # use the dynamic budget derived from updated last_scores.
        rng = np.random.RandomState(33)
        frames = rng.uniform(0.0, 1.0, size=(6, 3, 518, 518)).astype(np.float32)
        P = 5 + (518 // 14) ** 2
        per_block = 3 * P
        total_budget = per_block * 24
        return frames, total_budget

    @pytest.fixture(scope="class")
    def pt_run(self, pt_aggregator, frames_and_budget):
        import torch

        frames, total_budget = frames_and_budget
        past = [None] * pt_aggregator.depth
        outs = []
        budgets_per_frame = []
        with torch.no_grad():
            for i, frame in enumerate(frames):
                # Snapshot the budgets the aggregator will use on this call.
                budgets_per_frame.append(
                    pt_aggregator._calculate_dynamic_budgets(total_budget)
                    .cpu().numpy().copy()
                )
                one = torch.from_numpy(frame[None, None])
                out_list, _, past = pt_aggregator(
                    one,
                    past_key_values=past,
                    use_cache=True,
                    past_frame_idx=i,
                    total_budget=total_budget,
                )
                outs.append(out_list[-1].cpu().numpy())
        past_np = [(k.cpu().numpy(), v.cpu().numpy()) for (k, v) in past]
        final_last_scores = pt_aggregator.last_scores.cpu().numpy().copy()
        return outs, past_np, budgets_per_frame, final_last_scores

    @pytest.fixture(scope="class")
    def jx_run(self, jx_params, frames_and_budget):
        from modules.vggt.jax.aggregator import (
            Aggregator,
            _calculate_dynamic_budgets,
        )

        frames, total_budget = frames_and_budget
        past = None
        last_scores = None
        outs = []
        budgets_per_frame = []
        for i, frame in enumerate(frames):
            ls = (
                last_scores
                if last_scores is not None
                else jnp.zeros((24,), dtype=jnp.float32)
            )
            budgets_per_frame.append(
                np.asarray(_calculate_dynamic_budgets(ls, total_budget))
            )
            one = jnp.asarray(frame[None, None])
            out_list, _, past, last_scores = Aggregator().apply(
                jx_params,
                one,
                use_cache=True,
                past_kvs=past,
                past_frame_idx=i,
                total_budget=total_budget,
                last_scores=last_scores,
            )
            outs.append(np.asarray(out_list[-1]))
        past_np = [(np.asarray(k), np.asarray(v)) for (k, v) in past]
        return outs, past_np, budgets_per_frame, np.asarray(last_scores)

    def test_dynamic_budgets_per_frame(self, pt_run, jx_run):
        """The per-block budget arrays should match frame-by-frame."""
        _, _, pt_bud, _ = pt_run
        _, _, jx_bud, _ = jx_run
        for i in range(len(pt_bud)):
            diff = np.abs(pt_bud[i] - jx_bud[i]).max()
            assert diff <= 2, f"frame {i} budget diff={diff}, pt={pt_bud[i]}"

    def test_last_scores_match_pytorch(self, pt_run, jx_run):
        _, _, _, pt_ls = pt_run
        _, _, _, jx_ls = jx_run
        diff = np.abs(pt_ls - jx_ls).max()
        assert diff <= ATOL_DYNAMIC_BUDGET_FP32, (
            f"last_scores diff={diff:.3e} pt={pt_ls[:5]} jx={jx_ls[:5]}"
        )

    def test_per_frame_output_matches_pytorch(self, pt_run, jx_run):
        pt_outs, _, _, _ = pt_run
        jx_outs, _, _, _ = jx_run
        # 6 frames + dynamic budget → more fp32-noise accumulation than the
        # 4-frame uniform-budget (6b) test. Minor softmax rounding in
        # _calculate_dynamic_budgets can shift a handful of retained
        # candidates across blocks; measured drift is ~1.3e-3 at frame 4.
        atol = 2e-3
        for i, (pt, jx) in enumerate(zip(pt_outs, jx_outs)):
            max_abs = np.max(np.abs(pt - jx))
            assert max_abs <= atol, (
                f"frame {i} output: max_abs={max_abs:.3e} > {atol:.0e}"
            )

    def test_anchor_kv_matches_pytorch(self, pt_run, jx_run):
        """Anchors (first P tokens) are never evicted and should bit-match.

        The candidate portion is deliberately NOT compared here: 6c's
        dynamic-budget allocation rounds proportions to int, and a 1-unit
        budget difference between PT and JX can swap a single candidate
        token at the eviction boundary, producing large per-index K/V
        divergence even though the overall outputs match. Anchors bypass
        that path entirely — their contents are fully determined by frame
        0's forward pass.
        """
        _, pt_past, _, _ = pt_run
        _, jx_past, _, _ = jx_run
        P = 1374  # 5 specials + 37*37 patches
        for b in [0, 11, 23]:
            pt_k, pt_v = pt_past[b]
            jx_k, jx_v = jx_past[b]
            max_abs_k = np.max(np.abs(pt_k[:, :, :P] - jx_k[:, :, :P]))
            max_abs_v = np.max(np.abs(pt_v[:, :, :P] - jx_v[:, :, :P]))
            assert max_abs_k <= 3e-3, f"block {b} anchor K: {max_abs_k:.3e}"
            assert max_abs_v <= 3e-3, f"block {b} anchor V: {max_abs_v:.3e}"


# --------------------------------------------------------------------------- #
#  Level 3: padded-vs-legacy KV-cache parity
# --------------------------------------------------------------------------- #

# The JAXVGGTFeatureExtractor uses a fixed-size padded 3-tuple cache
# (k_pad, v_pad, valid_len) so the aggregator can be jitted once and reused.
# The existing Level-3 tests above all use past_kvs=None and therefore only
# exercise the legacy 2-tuple growing path. These tests lock the padded path
# to the legacy path's numerics so layout/reshape/mask regressions surface
# at a small block-level unit instead of only the 4-min integration test.
ATOL_PADDED_VS_LEGACY_FP32 = 1e-5


class TestLevel3PaddedCacheParity:
    """Padded 3-tuple cache must match legacy 2-tuple cache bit-for-bit.

    Both paths implement the same math — attention over [past tokens | new
    tokens]. The padded path pre-allocates a fixed buffer and masks invalid
    slots; the legacy path grows a list by concat. With no eviction, the
    two must produce identical output and identical stored K/V (at the
    valid prefix).
    """

    @pytest.fixture(scope="class")
    def block_setup(self):
        from modules.vggt.jax.block import Block

        dim = 64
        num_heads = 4
        head_dim = dim // num_heads
        N = 8
        block = Block(
            dim=dim, num_heads=num_heads, mlp_ratio=2.0,
            qk_norm=False, init_values=None,
        )
        rng_key = jax.random.PRNGKey(0)
        x_init = jnp.zeros((1, N, dim), dtype=jnp.float32)
        params = block.init(rng_key, x_init, use_cache=False)
        return dict(block=block, params=params, dim=dim, num_heads=num_heads,
                    head_dim=head_dim, N=N)

    def _empty_padded(self, B, H, MAX, Dh, dtype=jnp.float32):
        return (
            jnp.zeros((B, H, MAX, Dh), dtype=dtype),
            jnp.zeros((B, H, MAX, Dh), dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def test_single_frame_padded_matches_legacy(self, block_setup):
        """Frame 0, no prior cache — padded and legacy must give identical output.

        This is the exact case that caught the (0,2,1,3) transpose bug in
        _padded_cache_forward's output reshape.
        """
        block = block_setup["block"]
        params = block_setup["params"]
        N, H, Dh = block_setup["N"], block_setup["num_heads"], block_setup["head_dim"]
        MAX = 64
        cache_budget = 48

        x = jax.random.normal(jax.random.PRNGKey(42), (1, N, block_setup["dim"]),
                               dtype=jnp.float32)

        out_legacy, kv_legacy, _ = block.apply(
            params, x, past_kv=None, use_cache=True,
            cache_budget=cache_budget, num_anchor_tokens=0,
        )
        out_padded, kv_padded, _ = block.apply(
            params, x, past_kv=self._empty_padded(1, H, MAX, Dh),
            use_cache=True, cache_budget=cache_budget, num_anchor_tokens=0,
        )

        max_abs = np.max(np.abs(np.asarray(out_legacy) - np.asarray(out_padded)))
        assert max_abs <= ATOL_PADDED_VS_LEGACY_FP32, (
            f"single-frame output drift: {max_abs:.3e} > {ATOL_PADDED_VS_LEGACY_FP32:.0e}"
        )
        # Stored K/V must also match at the valid prefix.
        k_leg, v_leg = kv_legacy
        k_pad, v_pad, vl = kv_padded
        vl_int = int(np.asarray(vl))
        assert vl_int == N, (vl_int, N)
        np.testing.assert_allclose(np.asarray(k_leg), np.asarray(k_pad[:, :, :N]),
                                    atol=ATOL_PADDED_VS_LEGACY_FP32)
        np.testing.assert_allclose(np.asarray(v_leg), np.asarray(v_pad[:, :, :N]),
                                    atol=ATOL_PADDED_VS_LEGACY_FP32)

    def test_streaming_rollout_padded_matches_legacy(self, block_setup):
        """5-frame rollout, no eviction — padded must match legacy at every frame.

        Catches incremental-write bugs (off-by-one in valid_len, wrong
        dynamic_update_slice offset, stale mask) that single-frame misses.
        """
        block = block_setup["block"]
        params = block_setup["params"]
        N, H, Dh = block_setup["N"], block_setup["num_heads"], block_setup["head_dim"]
        n_frames = 5
        # Budget > total tokens so eviction never fires (this test is about
        # the write + masked-read path, not eviction).
        cache_budget = n_frames * N + 4
        MAX = cache_budget + N  # room for one more append before any evict

        rng = jax.random.PRNGKey(7)
        xs = [jax.random.normal(k, (1, N, block_setup["dim"]), dtype=jnp.float32)
              for k in jax.random.split(rng, n_frames)]

        legacy_past = None
        padded_past = self._empty_padded(1, H, MAX, Dh)
        for i, x in enumerate(xs):
            out_l, legacy_past, _ = block.apply(
                params, x, past_kv=legacy_past, use_cache=True,
                cache_budget=cache_budget, num_anchor_tokens=0,
            )
            out_p, padded_past, _ = block.apply(
                params, x, past_kv=padded_past, use_cache=True,
                cache_budget=cache_budget, num_anchor_tokens=0,
            )
            max_abs = np.max(np.abs(np.asarray(out_l) - np.asarray(out_p)))
            assert max_abs <= ATOL_PADDED_VS_LEGACY_FP32, (
                f"frame {i} output drift: {max_abs:.3e} > {ATOL_PADDED_VS_LEGACY_FP32:.0e}"
            )
            # Valid prefix of the padded K/V must equal the legacy K/V.
            k_leg, v_leg = legacy_past
            k_pad, v_pad, vl = padded_past
            vl_int = int(np.asarray(vl))
            expected_vl = (i + 1) * N
            assert vl_int == expected_vl, (i, vl_int, expected_vl)
            np.testing.assert_allclose(
                np.asarray(k_leg), np.asarray(k_pad[:, :, :expected_vl]),
                atol=ATOL_PADDED_VS_LEGACY_FP32,
                err_msg=f"frame {i} K prefix mismatch",
            )
            np.testing.assert_allclose(
                np.asarray(v_leg), np.asarray(v_pad[:, :, :expected_vl]),
                atol=ATOL_PADDED_VS_LEGACY_FP32,
                err_msg=f"frame {i} V prefix mismatch",
            )


class TestLevel3CameraHeadPaddedParity:
    """Padded 3-tuple cache on camera_head must match legacy 2-tuple bit-for-bit.

    Locks the new jitted camera-head path: the CameraHead module routes padded
    past_kvs_camera entries through Attention._padded_cache_forward instead of
    the legacy concat path. No eviction here (cache_budget=None), so with
    identical seed inputs the two paths should produce identical pose outputs
    and identical stored K/V prefixes at every frame.
    """

    @pytest.fixture(scope="class")
    def jx_params(self, state_dict):
        tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
        return jax.tree.map(jnp.asarray, {"params": tree["camera_head"]})

    @pytest.fixture(scope="class")
    def agg_tokens_per_frame(self):
        rng = np.random.RandomState(31)
        B, P, C = 1, 1374, 2048
        frames = []
        for _ in range(_L3_NUM_FRAMES):
            frames.append(
                [
                    rng.randn(B, 1, P, C).astype(np.float32) * 0.1
                    for _ in range(24)
                ]
            )
        return frames

    def _empty_padded(self, B, H, MAX, Dh, dtype=jnp.float32):
        return (
            jnp.zeros((B, H, MAX, Dh), dtype=dtype),
            jnp.zeros((B, H, MAX, Dh), dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def test_streaming_rollout_padded_matches_legacy(
        self, jx_params, agg_tokens_per_frame
    ):
        from modules.vggt.jax.heads.camera_head import CameraHead

        head = CameraHead()
        # Camera head: num_heads=16, dim_in=2048 -> head_dim=128, trunk_depth=4,
        # num_iterations=4. Cache grows by num_iterations per frame per block.
        H, Dh = head.num_heads, head.dim_in // head.num_heads
        n_iters = head.num_iterations
        # MAX with headroom for _L3_NUM_FRAMES frames.
        MAX = _L3_NUM_FRAMES * n_iters + n_iters

        legacy_past = None
        padded_past = [
            self._empty_padded(1, H, MAX, Dh) for _ in range(head.trunk_depth)
        ]

        for i, frame_tokens in enumerate(agg_tokens_per_frame):
            jx_input = [jnp.asarray(x) for x in frame_tokens]

            pose_legacy, legacy_past = head.apply(
                jx_params, jx_input,
                use_cache=True, past_kvs_camera=legacy_past,
            )
            pose_padded, padded_past = head.apply(
                jx_params, jx_input,
                use_cache=True, past_kvs_camera=padded_past,
            )

            # Pose output parity (last-iter prediction).
            max_abs = np.max(
                np.abs(np.asarray(pose_legacy[-1]) - np.asarray(pose_padded[-1]))
            )
            assert max_abs <= ATOL_PADDED_VS_LEGACY_FP32, (
                f"frame {i} pose drift: {max_abs:.3e} "
                f"> {ATOL_PADDED_VS_LEGACY_FP32:.0e}"
            )

            # K/V valid-prefix parity for every trunk block.
            expected_vl = (i + 1) * n_iters
            for k in range(head.trunk_depth):
                k_leg, v_leg = legacy_past[k]
                k_pad, v_pad, vl = padded_past[k]
                assert int(np.asarray(vl)) == expected_vl, (
                    i, k, int(np.asarray(vl)), expected_vl
                )
                np.testing.assert_allclose(
                    np.asarray(k_leg),
                    np.asarray(k_pad[:, :, :expected_vl]),
                    atol=ATOL_PADDED_VS_LEGACY_FP32,
                    err_msg=f"frame {i} block {k} K prefix mismatch",
                )
                np.testing.assert_allclose(
                    np.asarray(v_leg),
                    np.asarray(v_pad[:, :, :expected_vl]),
                    atol=ATOL_PADDED_VS_LEGACY_FP32,
                    err_msg=f"frame {i} block {k} V prefix mismatch",
                )
