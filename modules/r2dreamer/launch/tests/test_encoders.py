"""L2 (construction) and L3 (adapter behavior) tests for Encoder classes."""

from pathlib import Path

import numpy as np
import pytest

from modules.r2dreamer.adapters import ObsAdapter, VGGTObsAdapter
from modules.r2dreamer.launch.encoders import CNNEncoder, VGGTEncoder


_FIXTURES = Path(__file__).parent / "fixtures"


class TestCNNEncoder:
    def test_cnn_encoder_constructs(self):
        enc = CNNEncoder()
        adapter = enc.make_adapter()
        assert isinstance(adapter, ObsAdapter)

    def test_cnn_adapter_passthrough(self):
        adapter = CNNEncoder().make_adapter()
        dummy_img = np.zeros((3, 64, 64), dtype=np.uint8)
        obs = {"image": dummy_img, "is_first": True}
        buf_obs, agent_obs = adapter.transform(obs)
        # Default ObsAdapter returns the image unchanged and the full obs dict
        np.testing.assert_array_equal(buf_obs, dummy_img)
        assert agent_obs is obs


@pytest.mark.gpu
class TestVGGTEncoder:
    def test_vggt_encoder_constructs(self):
        enc = VGGTEncoder()
        adapter = enc.make_adapter()
        assert isinstance(adapter, VGGTObsAdapter)

    def test_vggt_adapter_behavior(self):
        """Adapter output must match validated pt_outputs within bf16 tolerance."""
        enc = VGGTEncoder()
        adapter = enc.make_adapter()

        frames_data = np.load(_FIXTURES / "sample_habitat_obs.npz")
        outputs_data = np.load(_FIXTURES / "expected_vggt_outputs.npz")

        # frames: (10, 3, 518, 518) uint8 CHW — matches VGGTFeatureExtractor.extract() input
        frames = frames_data["frames"]
        world_points = outputs_data["world_points"]  # (10, 37, 37, 3)
        camera_pose = outputs_data["camera_pose"]    # (10, 9)

        # Reset KV-cache before the sequence
        adapter._extractor.reset()

        for i in range(len(frames)):
            obs = {"image": frames[i], "is_first": i == 0}
            features, agent_obs = adapter.transform(obs)

            assert features.shape == (4116,), f"frame {i}: expected (4116,), got {features.shape}"
            assert features.dtype == np.float32, f"frame {i}: expected float32"

            expected = np.concatenate([
                world_points[i].reshape(-1),
                camera_pose[i],
            ]).astype(np.float32)

            np.testing.assert_allclose(
                features, expected,
                atol=2e-2, rtol=1e-2,
                err_msg=f"Mismatch at frame {i}",
            )
