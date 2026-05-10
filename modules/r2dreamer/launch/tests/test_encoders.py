"""L2 (construction) and L3 (adapter behavior) tests for Encoder classes."""

from pathlib import Path

import numpy as np
import pytest

from modules.r2dreamer.adapters import ObsAdapter, VGGTObsAdapter
from modules.r2dreamer.launch.encoders import (
    CNNEncoder,
    EncoderSpec,
    VGGTEncoder,
    VGGTAggregatorMLPEncoder,
)


_FIXTURES = Path(__file__).parent / "fixtures"


class TestCNNEncoder:
    def test_cnn_encoder_constructs(self):
        enc = CNNEncoder()
        adapter = enc.make_adapter()
        assert isinstance(adapter, ObsAdapter)

    def test_cnn_encoder_exposes_spec(self):
        spec = CNNEncoder().spec()
        assert isinstance(spec, EncoderSpec)
        assert spec.encoder_type == "cnn"
        assert spec.obs_shape == (3, 64, 64)
        assert spec.env_render_resolution == 64
        assert spec.agent_overrides == {}

    def test_cnn_adapter_passthrough(self):
        adapter = CNNEncoder().make_adapter()
        dummy_img = np.zeros((3, 64, 64), dtype=np.uint8)
        obs = {"image": dummy_img, "is_first": True}
        buf_obs, agent_obs = adapter.transform(obs)
        # Default ObsAdapter returns the image unchanged and the full obs dict
        np.testing.assert_array_equal(buf_obs, dummy_img)
        assert agent_obs is obs


class TestVGGTEncoderConfiguration:
    def test_vggt_encoder_uses_static_jax_budgets(self, monkeypatch):
        """R2Dreamer training must use the fast JAX static-budget VGGT path."""
        constructed_kwargs = {}

        class FakeExtractor:
            aggregator_feature_shape = (1374, 1024)

            def __init__(self, **kwargs):
                constructed_kwargs.update(kwargs)

            def reset(self):
                pass

        monkeypatch.setattr(
            "modules.r2dreamer.launch.encoders.VGGTFeatureExtractor",
            FakeExtractor,
        )

        enc = VGGTEncoder()
        adapter = enc.make_adapter()

        assert isinstance(adapter, VGGTObsAdapter)
        assert constructed_kwargs == {
            "total_budget": 200_000,
            "budgets_static": tuple([8333] * 24),
        }

    def test_vggt_encoder_exposes_wp_cp_spec(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (1374, 1024)

            def __init__(self, **kwargs):
                pass

            def reset(self):
                pass

        monkeypatch.setattr(
            "modules.r2dreamer.launch.encoders.VGGTFeatureExtractor",
            FakeExtractor,
        )

        spec = VGGTEncoder(resolution=518).spec()
        assert spec.encoder_type == "vggt"
        assert spec.obs_shape == (4116,)
        assert spec.env_render_resolution == 518
        assert spec.agent_overrides == {"buffer_capacity": 1_000_000}

    def test_aggregator_encoder_spec_uses_extractor_metadata(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (86, 128)

            def __init__(self, **kwargs):
                pass

            def reset(self):
                pass

        monkeypatch.setattr(
            "modules.r2dreamer.launch.encoders.VGGTFeatureExtractor",
            FakeExtractor,
        )

        enc = VGGTAggregatorMLPEncoder(resolution=256)
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert adapter.buffer_shape == (86, 128)
        assert spec.obs_shape == (86, 128)
        assert spec.env_render_resolution == 256
        assert spec.encoder_type == "vggt_aggregator_mlp"
        assert spec.agent_overrides == {
            "buffer_capacity": 5_000,
            "batch_size": 4,
            "seq_len": 32,
            "train_ratio": 128,
        }
        assert "all-token" in spec.design_notes

    def test_aggregator_adapter_uses_float16_replay_and_float32_agent(self):
        class FakeExtractor:
            aggregator_feature_shape = (6, 4)

            def reset(self):
                pass

            def extract(self, image):
                import jax.numpy as jnp
                return {"aggregator_features": jnp.ones((6, 4), dtype=jnp.float32)}

        adapter = VGGTObsAdapter(FakeExtractor(), feature_kind="aggregator")
        replay_features, agent_obs = adapter.transform({"image": np.zeros((3, 4, 4), dtype=np.uint8)})

        assert replay_features.shape == (6, 4)
        assert replay_features.dtype == np.float16
        assert agent_obs["features"].shape == (6, 4)
        assert agent_obs["features"].dtype.name == "float32"


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
