"""L2 (construction) and L3 (adapter behavior) tests for Encoder classes."""

from pathlib import Path

import numpy as np
import pytest

from src.r2dreamer.adapters import ObsAdapter, VGGTObsAdapter
from src.r2dreamer.encoders import (
    CNNEncoder,
    EncoderSpec,
    VGGTEncoder,
    VGGTAggregatorMLPEncoder,
    VGGTDenseWPEncoder,
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
            "src.r2dreamer.encoders.VGGTFeatureExtractor",
            FakeExtractor,
        )

        enc = VGGTEncoder()
        adapter = enc.make_adapter()

        assert isinstance(adapter, VGGTObsAdapter)
        assert constructed_kwargs == {
            "total_budget": 200_000,
            "budgets_static": tuple([8333] * 24),
            "compute_heads": True,
        }

    def test_vggt_encoder_exposes_wp_cp_spec(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (1374, 1024)

            def __init__(self, **kwargs):
                pass

            def reset(self):
                pass

        monkeypatch.setattr(
            "src.r2dreamer.encoders.VGGTFeatureExtractor",
            FakeExtractor,
        )

        spec = VGGTEncoder(resolution=518).spec()
        assert spec.encoder_type == "vggt"
        assert spec.obs_shape == (4116,)
        assert spec.env_render_resolution == 518
        assert spec.agent_overrides == {"buffer_capacity": 1_000_000}

    def test_aggregator_encoder_spec_uses_pooled_extractor_feature_dim(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (86, 128)

            def __init__(self, **kwargs):
                pass

            def reset(self):
                pass

        monkeypatch.setattr(
            "src.r2dreamer.encoders.VGGTFeatureExtractor",
            FakeExtractor,
        )

        enc = VGGTAggregatorMLPEncoder(resolution=256)
        adapter = enc.make_adapter()
        spec = enc.spec()

        # Adapter stores [cam | mean_patches | max_patches] flat: 3 * embed_dim.
        assert adapter.buffer_shape == (3 * 128,)
        assert adapter.buffer_dtype == "float32"
        assert spec.obs_shape == (3 * 128,)
        assert spec.env_render_resolution == 256
        assert spec.encoder_type == "vggt_aggregator_mlp"
        assert spec.agent_overrides == {
            "buffer_capacity": 5_000,
            "batch_size": 4,
            "seq_len": 32,
            "train_ratio": 128,
        }
        assert "camera token" in spec.design_notes

    def test_dense_wp_encoder_exposes_image_shaped_spec(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (1374, 1024)
            image_size = 518

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def reset(self):
                pass

        monkeypatch.setattr(
            "src.r2dreamer.encoders.VGGTFeatureExtractor",
            FakeExtractor,
        )

        enc = VGGTDenseWPEncoder(resolution=518)
        adapter = enc.make_adapter()
        spec = enc.spec()

        # Dense WP is stored channel-first as a (3, 518, 518) float16 image.
        assert adapter.buffer_shape == (3, 518, 518)
        assert adapter.buffer_dtype == "float16"
        assert spec.obs_shape == (3, 518, 518)
        assert spec.env_render_resolution == 518
        assert spec.encoder_type == "vggt_wp_dense_cnn"
        # Needs the point head (the dense map is its raw output).
        assert enc.vggt_compute_heads is True
        assert spec.agent_overrides == {
            "buffer_capacity": 5_000,
            "batch_size": 4,
            "seq_len": 32,
            "train_ratio": 128,
        }

    def test_dense_wp_adapter_emits_chw_pointmap(self):
        # Fake extractor returns an (H, W, 3) dense map; adapter must transpose
        # to (3, H, W) float16 and NOT divide by 255.
        class FakeExtractor:
            image_size = 4

            def reset(self):
                pass

            def extract(self, image, return_dense=False):
                import jax.numpy as jnp
                assert return_dense is True, "wp_dense must request the dense map"
                dense = jnp.arange(4 * 4 * 3, dtype=jnp.float32).reshape(4, 4, 3)
                return {"dense_world_points": dense}

        adapter = VGGTObsAdapter(FakeExtractor(), feature_kind="wp_dense")
        replay_features, agent_obs = adapter.transform(
            {"image": np.zeros((3, 4, 4), dtype=np.uint8)}
        )

        expected = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3).transpose(2, 0, 1)
        assert replay_features.shape == (3, 4, 4)
        assert replay_features.dtype == np.float16
        np.testing.assert_allclose(replay_features, expected.astype(np.float16))
        assert agent_obs["features"].shape == (3, 4, 4)
        assert agent_obs["features"].dtype.name == "float32"

    def test_aggregator_adapter_emits_cam_mean_max_pools(self):
        # Fake extractor with 1 cam + 4 register + 5 patch tokens, D = 4.
        # tokens = arange(40).reshape(10, 4); patches = tokens[5:].
        class FakeExtractor:
            aggregator_feature_shape = (10, 4)

            def reset(self):
                pass

            def extract(self, image):
                import jax.numpy as jnp
                return {"aggregator_features": jnp.arange(40, dtype=jnp.float32).reshape(10, 4)}

        adapter = VGGTObsAdapter(FakeExtractor(), feature_kind="aggregator")
        replay_features, agent_obs = adapter.transform({"image": np.zeros((3, 4, 4), dtype=np.uint8)})

        tokens = np.arange(40, dtype=np.float32).reshape(10, 4)
        expected_cam = tokens[0]
        expected_mean = tokens[5:].mean(axis=0)
        expected_max = tokens[5:].max(axis=0)
        expected = np.concatenate([expected_cam, expected_mean, expected_max])

        assert replay_features.shape == (3 * 4,)
        assert replay_features.dtype == np.float32
        np.testing.assert_allclose(replay_features, expected)
        assert agent_obs["features"].shape == (3 * 4,)
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
