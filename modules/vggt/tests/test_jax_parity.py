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

        pt = np.random.randn(4, 8, 2, 2).astype(np.float32)  # (I, O, H, W)
        jx = _conv_transpose2d_to_flax(pt)
        assert jx.shape == (2, 2, 4, 8)
        for i in range(4):
            for o in range(8):
                for kh in range(2):
                    for kw in range(2):
                        assert jx[kh, kw, i, o] == pt[i, o, kh, kw]

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
