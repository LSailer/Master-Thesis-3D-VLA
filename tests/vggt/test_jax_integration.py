"""Level-4 integration tests for ``JAXVGGTFeatureExtractor`` (Step 7).

Mirrors the contract tests in ``test_feature_extractor.py`` (shapes, NaN-
free streaming, reset reproducibility) plus a cross-backend drop-in
parity test: a 5-step rollout with the same RGB inputs should match the
PyTorch extractor to within the production tolerance.
"""

from __future__ import annotations

from pathlib import Path

import jax

# Match test_jax_parity: force highest matmul precision so parity isn't
# blown by TF32 drift. Both backends then have a fair fp32 comparison.
jax.config.update("jax_default_matmul_precision", "highest")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


try:
    HAS_CUDA_JAX = bool(jax.devices("gpu"))
except RuntimeError:
    HAS_CUDA_JAX = False
try:
    import torch

    HAS_CUDA_PT = torch.cuda.is_available()
except ImportError:
    HAS_CUDA_PT = False

gpu_jax = pytest.mark.skipif(not HAS_CUDA_JAX, reason="requires CUDA GPU for JAX")
both = pytest.mark.skipif(
    not (HAS_CUDA_JAX and HAS_CUDA_PT), reason="requires both JAX and PyTorch CUDA"
)


def _make_frame(seed: int = 0) -> np.ndarray:
    """Synthetic 518x518 RGB frame (CHW, uint8)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(3, 518, 518), dtype=np.uint8)


def _real_habitat_frame(index: int = 0) -> np.ndarray:
    """Real Habitat RGB frame fixture (CHW, uint8)."""
    fixture = (
        Path(__file__).parents[1]
        / "r2dreamer"
        / "launch"
        / "fixtures"
        / "sample_habitat_obs.npz"
    )
    return np.load(fixture)["frames"][index]


@gpu_jax
class TestJAXFeatureExtractorContract:
    """API-contract tests for the JAX extractor — mirrors PyTorch suite."""

    @pytest.fixture(scope="class")
    def extractor(self):
        from src.vggt.jax import JAXVGGTFeatureExtractor

        return JAXVGGTFeatureExtractor(device="cuda")

    def test_single_frame_shapes(self, extractor):
        extractor.reset()
        out = extractor.extract(_real_habitat_frame(index=0))
        assert out["world_points"].shape == (37, 37, 3)
        assert out["camera_pose"].shape == (9,)
        assert out["aggregator_features"].shape == (1374, 1024)
        assert out["world_points"].dtype == np.float32
        assert out["camera_pose"].dtype == np.float32
        assert out["aggregator_features"].dtype == np.float32
        assert not np.any(np.isnan(out["aggregator_features"]))

    def test_streaming_multiple_frames(self, extractor):
        """Five streaming frames produce NaN-free outputs of the right shape."""
        extractor.reset()
        for i in range(5):
            out = extractor.extract(_make_frame(seed=i))
            assert out["world_points"].shape == (37, 37, 3)
            assert out["camera_pose"].shape == (9,)
            assert not np.any(np.isnan(out["world_points"])), f"NaN at frame {i}"
            assert not np.any(np.isnan(out["camera_pose"])), f"NaN at frame {i}"

    def test_reset_reproduces_first_frame(self, extractor):
        """reset() must fully clear cache + last_scores + frame counter."""
        frame0 = _make_frame(seed=99)
        extractor.reset()
        out_first = extractor.extract(frame0)
        for i in range(3):
            extractor.extract(_make_frame(seed=200 + i))
        extractor.reset()
        out_again = extractor.extract(frame0)

        np.testing.assert_allclose(
            out_first["world_points"],
            out_again["world_points"],
            atol=1e-4,
            err_msg="world_points differ after reset",
        )
        np.testing.assert_allclose(
            out_first["camera_pose"],
            out_again["camera_pose"],
            atol=1e-4,
            err_msg="camera_pose differs after reset",
        )

    def test_camera_pose_not_all_zero_after_second_frame(self, extractor):
        extractor.reset()
        extractor.extract(_make_frame(seed=10))
        out2 = extractor.extract(_make_frame(seed=11))
        assert not np.allclose(out2["camera_pose"], 0.0, atol=1e-6)

    def test_camera_cache_overflow_raises(self):
        """Extracting past max_camera_frames must raise, not silently corrupt.

        Configures a fresh extractor with ``max_camera_frames=3`` (so
        ``_CAM_MAX = 3 × num_iterations = 12``).  Three calls succeed;
        the fourth must raise ``RuntimeError``.
        """
        from src.vggt.jax import JAXVGGTFeatureExtractor

        ext = JAXVGGTFeatureExtractor(device="cuda", max_camera_frames=3)
        ext.reset()
        for i in range(3):
            ext.extract(_make_frame(seed=500 + i))
        with pytest.raises(RuntimeError, match="Camera-head padded cache overflow"):
            ext.extract(_make_frame(seed=599))

    def test_compute_heads_false_returns_only_aggregator(self):
        """compute_heads=False skips camera/point heads and world_points wrapper."""
        from src.vggt.jax import JAXVGGTFeatureExtractor

        ext = JAXVGGTFeatureExtractor(device="cuda", compute_heads=False)
        ext.reset()
        out = ext.extract(_make_frame(seed=7))
        assert set(out.keys()) == {"aggregator_features"}
        assert out["aggregator_features"].shape == (1374, 1024)
        assert out["aggregator_features"].dtype == np.float32
        assert not np.any(np.isnan(out["aggregator_features"]))

    def test_compute_heads_false_aggregator_matches_full(self, extractor):
        """Skipping heads must not change the aggregator output values."""
        from src.vggt.jax import JAXVGGTFeatureExtractor

        frame = _make_frame(seed=11)
        extractor.reset()
        out_full = extractor.extract(frame)
        ext_skip = JAXVGGTFeatureExtractor(device="cuda", compute_heads=False)
        ext_skip.reset()
        out_skip = ext_skip.extract(frame)
        np.testing.assert_allclose(
            np.asarray(out_full["aggregator_features"]),
            np.asarray(out_skip["aggregator_features"]),
            atol=1e-5,
            err_msg="aggregator_features changed when heads were skipped",
        )


@both
class TestJAXvsPyTorchExtractor:
    """Cross-backend drop-in parity — plan Level-4 exit gate."""

    # Level-4 tolerance per the plan: ``atol ≤ 1e-2`` for the rollout.
    # We relax slightly above the per-frame parity test (1e-3) because
    # the pool operation + different float accumulation paths add noise.
    ROLLOUT_ATOL = 1e-2
    N_FRAMES = 5

    @pytest.fixture(scope="class")
    def jax_ext(self):
        import jax.numpy as jnp

        from src.vggt.jax import JAXVGGTFeatureExtractor

        # Force fp32 to line up with the fp32-pinned PyTorch fixture below;
        # default bf16 would exceed ROLLOUT_ATOL against fp32 PyTorch.
        return JAXVGGTFeatureExtractor(device="cuda", dtype=jnp.float32)

    @pytest.fixture(scope="class")
    def pt_ext(self):
        from src.vggt.feature_extractor import VGGTFeatureExtractor

        ext = VGGTFeatureExtractor(device="cuda")
        # Force full-precision attention in the aggregator so the backends
        # line up. Production bf16 parity is part of Step 8's benchmark.
        ext.model.aggregator = ext.model.aggregator.to(torch.float32)
        ext.model.camera_head = ext.model.camera_head.to(torch.float32)
        ext.model.point_head = ext.model.point_head.to(torch.float32)
        for blk in ext.model.aggregator.patch_embed.blocks:
            blk.attn.fused_attn = False
        for blk in ext.model.aggregator.frame_blocks:
            blk.attn.fused_attn = False
        for blk in ext.model.aggregator.global_blocks:
            blk.attn.fused_attn = False
        for blk in ext.model.camera_head.trunk:
            blk.attn.fused_attn = False
        # Disable the bf16 autocast the extractor installs by default.
        ext._amp_dtype = torch.float32
        yield ext
        del ext
        torch.cuda.empty_cache()

    def test_rollout_parity(self, jax_ext, pt_ext):
        """5-frame rollout: JAX outputs match PyTorch within Level-4 atol."""
        frames = [_make_frame(seed=300 + i) for i in range(self.N_FRAMES)]

        jax_ext.reset()
        pt_ext.reset()

        jax_outs = [jax_ext.extract(f) for f in frames]
        pt_outs = [pt_ext.extract(f) for f in frames]

        for i, (jx, pt) in enumerate(zip(jax_outs, pt_outs)):
            wp_err = np.max(np.abs(jx["world_points"] - pt["world_points"]))
            cp_err = np.max(np.abs(jx["camera_pose"] - pt["camera_pose"]))
            assert wp_err <= self.ROLLOUT_ATOL, (
                f"frame {i} world_points err={wp_err:.3e} "
                f"> {self.ROLLOUT_ATOL:.0e}"
            )
            assert cp_err <= self.ROLLOUT_ATOL, (
                f"frame {i} camera_pose err={cp_err:.3e} "
                f"> {self.ROLLOUT_ATOL:.0e}"
            )
