
"""Smoke tests for VGGTFeatureExtractor streaming inference."""

import numpy as np
import pytest

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


gpu = pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA GPU")


def _make_frame(seed: int = 0) -> np.ndarray:
    """Create a synthetic 518x518 RGB frame (CHW, uint8)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(518, 518, 3), dtype=np.uint8)


@gpu
class TestVGGTFeatureExtractor:
    """Tests that require a GPU and the InfiniteVGGT checkpoint."""

    @pytest.fixture(scope="class")
    def extractor(self):
        from src.vggt.reference.feature_extractor import VGGTFeatureExtractor

        ext = VGGTFeatureExtractor(device="cuda")
        yield ext
        # Cleanup GPU memory after all tests in this class.
        del ext


    # ---- shape & dtype tests ------------------------------------------------

    def test_single_frame_shapes(self, extractor):
        """extract() on one frame returns correct shapes and dtypes."""
        frame = _make_frame(seed=42)
        out = extractor.extract(frame)

        assert out["world_points"].shape == (37, 37, 3), (
            f"Expected (37, 37, 3), got {out['world_points'].shape}"
        )
        assert out["camera_pose"].shape == (9,), (
            f"Expected (9,), got {out['camera_pose'].shape}"
        )
        assert out["world_points"].dtype == np.float32
        assert out["camera_pose"].dtype == np.float32

    def test_streaming_multiple_frames(self, extractor):
        """Streaming extraction over 5 frames produces valid outputs each time."""
        extractor.reset()
        for i in range(5):
            frame = _make_frame(seed=i)
            out = extractor.extract(frame)

            assert out["world_points"].shape == (37, 37, 3)
            assert out["camera_pose"].shape == (9,)
            assert not np.any(np.isnan(out["world_points"])), f"NaN in world_points at frame {i}"
            assert not np.any(np.isnan(out["camera_pose"])), f"NaN in camera_pose at frame {i}"

    # ---- KV-cache reset test ------------------------------------------------

    def test_reset_reproduces_first_frame(self, extractor):
        """After reset(), the same first frame should give the same output."""
        frame0 = _make_frame(seed=99)

        # First run: extract frame0 as the first frame of an episode.
        extractor.reset()
        out_first = extractor.extract(frame0)

        # Feed a few more frames to change KV-cache state.
        for i in range(3):
            extractor.extract(_make_frame(seed=200 + i))

        # Reset and re-extract the same first frame.
        extractor.reset()
        out_after_reset = extractor.extract(frame0)

        np.testing.assert_allclose(
            out_first["world_points"],
            out_after_reset["world_points"],
            atol=1e-4,
            err_msg="world_points differ after reset — KV-cache was not fully cleared",
        )
        np.testing.assert_allclose(
            out_first["camera_pose"],
            out_after_reset["camera_pose"],
            atol=1e-4,
            err_msg="camera_pose differs after reset — KV-cache was not fully cleared",
        )

    # ---- value sanity tests -------------------------------------------------

    def test_camera_pose_not_all_zero_after_second_frame(self, extractor):
        """After the second frame, camera_pose should have non-trivial values."""
        extractor.reset()
        extractor.extract(_make_frame(seed=10))
        out2 = extractor.extract(_make_frame(seed=11))

        # At least some pose components should be non-zero after two frames.
        assert not np.allclose(out2["camera_pose"], 0.0, atol=1e-6), (
            "camera_pose is all zeros after second frame — streaming may not be working"
        )
