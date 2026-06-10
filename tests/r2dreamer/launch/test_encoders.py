"""L2 (construction) and L3 (adapter behavior) tests for Encoder classes."""

import ast
from pathlib import Path

import numpy as np
import pytest

from src.r2dreamer.adapters import ObsAdapter, VGGTObsAdapter
from src.r2dreamer.adapters.hybrid_adapter import HybridObsAdapter
from src.r2dreamer.encoders import (
    CNNEncoder,
    EncoderSpec,
    HybridEncoder,
    VGGTAggTokenTransformerEncoder,
    VGGTEncoder,
    VGGTAggregatorMLPEncoder,
    VGGTDenseWPEncoder,
    VGGTWPCP64Encoder,
)
from src.r2dreamer.encoders.specs import VGGT_VARIANTS
from src.r2dreamer.world_model import encoders as wm_encoders
from src.shared.video_utils import resize_chw_uint8


_FIXTURES = Path(__file__).parent / "fixtures"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_encoder_package_init_is_thin_reexport():
    package_init = _REPO_ROOT / "src/r2dreamer/encoders/__init__.py"
    tree = ast.parse(package_init.read_text())
    nodes = [
        node for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]

    for node in nodes:
        assert isinstance(node, (ast.ImportFrom, ast.Assign))
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1
            assert isinstance(node.targets[0], ast.Name)
            assert node.targets[0].id == "__all__"


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
            "src.r2dreamer.encoders.specs.VGGTFeatureExtractor", FakeExtractor
        )

        enc = VGGTEncoder()
        adapter = enc.make_adapter()

        assert isinstance(adapter, VGGTObsAdapter)
        assert constructed_kwargs == {
            "total_budget": 200_000,
            "budgets_static": tuple([8333] * 24),
            "compute_heads": True,
            "wp_pool_size": 37,
        }

    def test_vggt_encoder_exposes_wp_cp_spec(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (1374, 1024)

            def __init__(self, **kwargs):
                pass

            def reset(self):
                pass

        monkeypatch.setattr(
            "src.r2dreamer.encoders.specs.VGGTFeatureExtractor", FakeExtractor
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
            "src.r2dreamer.encoders.specs.VGGTFeatureExtractor", FakeExtractor
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

    def test_agg_token_transformer_spec_keeps_full_tokens_fp16(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (10, 4)

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def reset(self):
                pass

        monkeypatch.setattr(
            "src.r2dreamer.encoders.specs.VGGTFeatureExtractor", FakeExtractor
        )

        enc = VGGTAggTokenTransformerEncoder(resolution=256)
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert adapter.buffer_shape == (10 * 4,)
        assert adapter.buffer_dtype == "float16"
        assert spec.obs_shape == (10 * 4,)
        assert spec.env_render_resolution == 256
        assert spec.encoder_type == "vggt_agg_token_transformer"
        assert spec.module_cls is wm_encoders.VGGTAggTokenTransformerEncoder
        assert enc.vggt_compute_heads is False
        assert spec.agent_overrides == {
            "buffer_capacity": 5_000,
            "batch_size": 1,
            "seq_len": 8,
            "train_ratio": 32,
        }
        assert "1374" in spec.design_notes

    def test_dense_wp_encoder_exposes_image_shaped_spec(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (1374, 1024)
            image_size = 518

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def reset(self):
                pass

        monkeypatch.setattr(
            "src.r2dreamer.encoders.specs.VGGTFeatureExtractor", FakeExtractor
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

    def test_wp_cp_64_encoder_spec(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (1374, 1024)

            def __init__(self, **kwargs):
                self.wp_pool_size = int(kwargs.get("wp_pool_size", 37))

            def reset(self):
                pass

        monkeypatch.setattr(
            "src.r2dreamer.encoders.specs.VGGTFeatureExtractor", FakeExtractor
        )

        enc = VGGTWPCP64Encoder(resolution=518)
        adapter = enc.make_adapter()
        spec = enc.spec()

        # 64x64 WP grid: obs = 64*64*3 + 9 = 12297 (vs 4116 at 37x37).
        assert enc.wp_pool_size == 64
        assert adapter.buffer_shape == (64 * 64 * 3 + 9,)
        assert adapter.buffer_dtype == "float32"
        assert spec.obs_shape == (12297,)
        assert spec.encoder_type == "vggt_wp_cp_64"
        # Same MLP module + 1M buffer as the 37x37 WP+CP run -> resolution-only ablation.
        assert spec.module_cls is wm_encoders.VGGTEncoder
        assert spec.agent_overrides == {"buffer_capacity": 1_000_000}

    def test_pool_dense_world_points(self):
        # 37 divides 518 (exact 14x14 block mean); 64 does not (antialiased resize).
        import jax.numpy as jnp
        from src.vggt.jax.feature_extractor import _pool_dense_world_points

        x = jnp.arange(518 * 518 * 3, dtype=jnp.float32).reshape(1, 518, 518, 3)
        p37 = _pool_dense_world_points(x, 37)
        p64 = _pool_dense_world_points(x, 64)
        assert p37.shape == (1, 37, 37, 3)
        assert p64.shape == (1, 64, 64, 3)
        # 37 path must equal an exact 14x14 block average.
        exact = x.reshape(1, 37, 14, 37, 14, 3).mean(axis=(2, 4))
        np.testing.assert_allclose(np.asarray(p37), np.asarray(exact), rtol=1e-5, atol=1e-3)
        assert np.isfinite(np.asarray(p64)).all()

    def test_wp_cp_64_adapter_flattens_world_points_plus_pose(self):
        import jax.numpy as jnp

        class FakeExtractor:
            wp_pool_size = 64

            def reset(self):
                pass

            def extract(self, image):
                return {
                    "world_points": jnp.ones((64, 64, 3), jnp.float32),
                    "camera_pose": jnp.arange(9, dtype=jnp.float32),
                }

        adapter = VGGTObsAdapter(FakeExtractor(), feature_kind="wp_cp")
        assert adapter.buffer_shape == (12297,)
        rep, agent_obs = adapter.transform({"image": np.zeros((3, 518, 518), np.uint8)})
        assert rep.shape == (12297,)
        assert rep.dtype == np.float32
        # last 9 entries are the pose vector 0..8
        np.testing.assert_allclose(rep[-9:], np.arange(9, dtype=np.float32))
        assert agent_obs["features"].shape == (12297,)

    def test_vggt_launcher_variants_are_centralized(self):
        assert VGGTEncoder.variant is VGGT_VARIANTS["vggt"]
        assert VGGTAggregatorMLPEncoder.variant is VGGT_VARIANTS["vggt_aggregator_mlp"]
        assert VGGTAggTokenTransformerEncoder.variant is VGGT_VARIANTS[
            "vggt_agg_token_transformer"
        ]
        assert VGGTDenseWPEncoder.variant is VGGT_VARIANTS["vggt_wp_dense_cnn"]
        assert VGGTWPCP64Encoder.variant is VGGT_VARIANTS["vggt_wp_cp_64"]

        assert VGGT_VARIANTS["vggt"].compute_heads is True
        assert VGGT_VARIANTS["vggt_aggregator_mlp"].compute_heads is False
        assert VGGT_VARIANTS["vggt_agg_token_transformer"].compute_heads is False
        assert VGGT_VARIANTS["vggt_wp_cp_64"].wp_pool_size == 64

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
        replay_features, agent_obs = adapter.transform(
            {"image": np.zeros((3, 4, 4), dtype=np.uint8)}
        )

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

    def test_agg_token_adapter_keeps_camera_register_and_patch_tokens(self):
        class FakeExtractor:
            aggregator_feature_shape = (10, 4)

            def reset(self):
                pass

            def extract(self, image):
                import jax.numpy as jnp
                return {"aggregator_features": jnp.arange(40, dtype=jnp.float32).reshape(10, 4)}

        adapter = VGGTObsAdapter(FakeExtractor(), feature_kind="agg_tokens")
        replay_features, agent_obs = adapter.transform(
            {"image": np.zeros((3, 4, 4), dtype=np.uint8)}
        )

        expected = np.arange(40, dtype=np.float32)
        assert replay_features.shape == (40,)
        assert replay_features.dtype == np.float16
        np.testing.assert_allclose(replay_features, expected.astype(np.float16))
        assert agent_obs["features"].shape == (40,)
        assert agent_obs["features"].dtype.name == "float32"


class TestHybridEncoder:
    """L2/L3 tests for the hybrid (CNN + gated WP/CP MLP) encoder.

    No GPU / real VGGT: the spec test monkeypatches the extractor used inside
    the spec class, and the adapter test injects a hand-rolled fake extractor
    whose ``.extract()`` returns world_points (37,37,3) + camera_pose (9,).
    """

    def test_hybrid_encoder_exposes_spec(self, monkeypatch):
        class FakeExtractor:
            aggregator_feature_shape = (1374, 1024)

            def __init__(self, **kwargs):
                pass

            def reset(self):
                pass

        monkeypatch.setattr(
            "src.r2dreamer.encoders.specs.VGGTFeatureExtractor", FakeExtractor
        )

        spec = HybridEncoder().spec()
        assert isinstance(spec, EncoderSpec)
        assert spec.encoder_type == "hybrid"
        assert spec.obs_shape == (16404,)
        assert spec.env_render_resolution == 518
        assert spec.module_cls is wm_encoders.HybridEncoder

    def test_hybrid_adapter_builds_rgb_wp_cp_layout(self):
        # Fake VGGT extractor: extract() -> world_points (37,37,3) + camera_pose (9,)
        # so flatten_world_points_camera_pose yields 37*37*3 + 9 = 4116.
        world_points = np.arange(37 * 37 * 3, dtype=np.float32).reshape(37, 37, 3)
        camera_pose = np.arange(9, dtype=np.float32) + 100.0

        class FakeExtractor:
            def reset(self):
                pass

            def extract(self, image):
                import jax.numpy as jnp
                return {
                    "world_points": jnp.asarray(world_points),
                    "camera_pose": jnp.asarray(camera_pose),
                }

        adapter = HybridObsAdapter(FakeExtractor())
        assert isinstance(adapter, ObsAdapter)
        assert adapter.buffer_shape == (16404,)
        assert adapter.buffer_dtype == "float32"
        assert adapter.normalize_on_sample is False

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(3, 518, 518), dtype=np.uint8)
        obs_dict = {"image": image, "is_first": True}

        replay, agent_obs = adapter.transform(obs_dict)

        assert replay.shape == (16404,)
        assert replay.dtype == np.float32

        # RGB slice: normalized 64x64 resize of the input, in [0, 1].
        img64 = resize_chw_uint8(image, 64)  # (3,64,64) uint8
        expected_rgb = (img64.astype(np.float32) / 255.0).reshape(-1)
        assert replay[:12288].min() >= 0.0
        assert replay[:12288].max() <= 1.0
        np.testing.assert_allclose(replay[:12288], expected_rgb)

        # WP/CP slice: flattened world_points then camera_pose.
        expected_wp_cp = np.concatenate(
            [world_points.reshape(-1), camera_pose]
        ).astype(np.float32)
        np.testing.assert_allclose(replay[12288:], expected_wp_cp)

        assert np.asarray(agent_obs["hybrid"]).shape == (16404,)
        assert agent_obs["image"].shape == (3, 64, 64)


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
