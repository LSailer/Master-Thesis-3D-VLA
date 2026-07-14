"""L2 (construction) and L3 (adapter behavior) tests for Encoder classes."""
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,import-outside-toplevel,unused-argument
# pylint: disable=line-too-long,use-implicit-booleaness-not-comparison
# pylint: disable=protected-access,consider-using-enumerate

import json
from pathlib import Path

import numpy as np
import pytest

from src.buffer.replay_buffer import ReplayBatch
from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters import ObsAdapter, VGGTObsAdapter
from src.r2dreamer.adapters.house_context_adapter import VGGTHouseContextObsAdapter
from src.r2dreamer.adapters.house_points_adapter import (
    VGGTHousePointsPoseObsAdapter,
    VGGTHybridHousePointsPoseObsAdapter,
)
from src.r2dreamer.adapters.hybrid_adapter import HybridObsAdapter
from src.r2dreamer.adapters.token_adapters import (
    VGGTHouseFullTokenObsAdapter,
    VGGTHouseGlobalEmbeddingObsAdapter,
)
from src.r2dreamer.encoders import (
    VGGT_VARIANTS,
    CNNEncoder,
    EncoderSpec,
    HybridEncoder,
    VGGTAggregatorMLPEncoder,
    VGGTAggTokenTransformerEncoder,
    VGGTDenseWPEncoder,
    VGGTEncoder,
    VGGTHouseContextEncoder,
    VGGTHouseFullTokenNoGateEncoder,
    VGGTHouseGlobalEmbeddingEncoder,
    VGGTHousePointsPoseEncoder,
    VGGTHybridHousePointsPoseEncoder,
    VGGTWP64CNNCPMLPEncoder,
    VGGTWPCP64Encoder,
)
from src.r2dreamer.encoders.constants import (
    HOUSE_CONTEXT_MAX_POINTS,
    HOUSE_POINT_DIM,
)
from src.r2dreamer.encoders.cnn import ConvEncoder
from src.vggt.jax.feature_extractor import ResetMode
from src.r2dreamer.encoders.mlp import (
    HybridEncoder as ModelHybridEncoder,
)
from src.r2dreamer.encoders.mlp import (
    encode_house_global_obs as ModelHouseGlobalEmbeddingEncoder,
)
from src.r2dreamer.encoders.mlp import (
    MLPEncoder,
    WP64CNNCPMLPEncoder,
)
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    CAMERA_TOKEN_GLOBAL_KEY,
    FULL_TOKENS_KEY,
    GLOBAL_PATCH_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
    WORLD_POINTS_KEY,
)
from src.r2dreamer.observation_preparation import (
    CNNObservationPreparation,
    EncoderInputContract,
)
from src.shared.video_utils import resize_chw_uint8

_FIXTURES = Path(__file__).parent / "fixtures"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_minimal_static_house_ply(path: Path) -> None:
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
        "0.0 0.0 0.0 255 0 0\n"
        "1.0 1.0 1.0 0 0 255\n",
        encoding="ascii",
    )


class _StubExtractor:
    """Minimal stand-in for ``JAXVGGTFeatureExtractor`` in construction/spec tests.

    Records the constructor kwargs on ``self.kwargs`` (so a test can assert the
    budgets/reset-mode the encoder passed), derives ``wp_pool_size`` from them,
    and remembers the last ``reset_for_scene`` id. Per-test overrides such as a
    different ``aggregator_feature_shape`` go through the ``patch_vggt`` fixture.
    """

    aggregator_feature_shape = (1374, 1024)
    image_size = 518

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.wp_pool_size = int(kwargs.get("wp_pool_size", 37))
        self.last_scene_id = None

    def reset(self):
        pass

    def reset_for_scene(self, scene_id="scene"):
        self.last_scene_id = scene_id


@pytest.fixture
def patch_vggt(monkeypatch):
    """Patch ``JAXVGGTFeatureExtractor`` with :class:`_StubExtractor`.

    Returns a callable that installs the stub and returns the installed class;
    keyword arguments override stub class attributes for one test, e.g.
    ``patch_vggt(aggregator_feature_shape=(86, 128))``.
    """

    def install(**attrs):
        stub = type("_PatchedStub", (_StubExtractor,), attrs) if attrs else _StubExtractor
        monkeypatch.setattr(
            "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor", stub
        )
        return stub

    return install


class _WorldPoints64Extractor:
    """Fake extractor whose ``extract`` returns a (64, 64, 3) world map + pose."""

    wp_pool_size = 64

    def reset(self):
        pass

    def extract(self, image):
        import jax.numpy as jnp

        return {
            "world_points": jnp.ones((64, 64, 3), jnp.float32),
            "camera_pose": jnp.arange(9, dtype=jnp.float32),
        }


class _AggregatorTokensExtractor:
    """Fake extractor whose ``extract`` returns arange(40) aggregator tokens."""

    aggregator_feature_shape = (10, 4)

    def reset(self):
        pass

    def extract(self, image):
        import jax.numpy as jnp

        return {
            "aggregator_features": jnp.arange(40, dtype=jnp.float32).reshape(10, 4)
        }


def test_encoder_specs_module_was_folded_into_package_init():
    assert not (_REPO_ROOT / "src/r2dreamer/encoders/specs.py").exists()
    assert VGGTEncoder.variant is VGGT_VARIANTS["vggt"]


class TestCNNEncoder:
    def test_cnn_encoder_constructs(self):
        enc = CNNEncoder()
        adapter = enc.make_adapter()
        assert isinstance(adapter, ObsAdapter)
        assert isinstance(adapter, CNNObservationPreparation)

    def test_cnn_encoder_exposes_spec(self):
        spec = CNNEncoder().spec()
        assert isinstance(spec, EncoderSpec)
        assert spec.encoder_type == "cnn"
        assert spec.obs_shape == (3, 64, 64)
        assert spec.env_render_resolution == 64
        assert spec.module_cls is ConvEncoder
        assert spec.agent_overrides == {}
        assert spec.contract_snapshot["encoder_type"] == "cnn"
        json.dumps(spec.contract_snapshot)

    def test_cnn_adapter_passthrough(self):
        adapter = CNNEncoder().make_adapter()
        dummy_img = np.zeros((3, 64, 64), dtype=np.uint8)
        obs = ObservationFrame(image=dummy_img, is_first=True)
        buf_obs, agent_obs = adapter.transform(obs)
        # CNN Observation Preparation returns explicit replay and agent observations.
        np.testing.assert_array_equal(buf_obs, dummy_img)
        assert agent_obs["image"] is dummy_img
        assert agent_obs["is_first"] is True


class TestVGGTEncoderConfiguration:
    def test_vggt_encoder_uses_static_jax_budgets(self, patch_vggt):
        """R2Dreamer training must use the fast JAX static-budget VGGT path."""
        patch_vggt()

        adapter = VGGTEncoder().make_adapter()

        assert isinstance(adapter, VGGTObsAdapter)
        assert adapter._extractor.kwargs == {
            "total_budget": 1_200_000,
            "budgets_static": tuple([50_000] * 24),
            "compute_heads": True,
            "wp_pool_size": 37,
            "reset_mode": ResetMode.FULL,
        }

    def test_house_points_pose_encoder_uses_persist_scene(self, patch_vggt):
        """The live per-scene house-point path must persist the VGGT cache per
        scene so episodes of one house share one world frame (no ghost copies).
        """
        patch_vggt()

        adapter = VGGTHousePointsPoseEncoder().make_adapter()

        assert adapter._extractor.kwargs["reset_mode"] is ResetMode.PERSIST_SCENE

    def test_house_points_adapter_episode_reset_is_scene_aware(self, patch_vggt):
        """The adapter's on_episode_reset must call reset_for_scene with the
        incoming scene_id (and fall back to "scene" when called with no arg,
        so standalone profiling scripts that call it no-arg still work). This
        is the fix for the prefill-doesn't-fire-reset_for_scene gap.
        """
        patch_vggt()

        enc = VGGTHousePointsPoseEncoder()
        adapter = enc.make_adapter()
        assert adapter.on_episode_reset is not None

        adapter.on_episode_reset("house-7")
        assert adapter._extractor.last_scene_id == "house-7"

        # Backward-compatible no-arg call (profiling/debug scripts) -> "scene".
        adapter.on_episode_reset()
        assert adapter._extractor.last_scene_id == "scene"

    def test_vggt_encoder_exposes_wp_cp_spec(self, patch_vggt):
        patch_vggt()

        enc = VGGTEncoder(resolution=518)
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert isinstance(adapter.contract, EncoderInputContract)
        assert adapter.contract.encoder_input.shape == (4116,)
        assert adapter.contract.replay_observation.fields[WORLD_POINTS_KEY].shape == (
            3,
            37,
            37,
        )
        assert adapter.contract.replay_observation.fields[CAMERA_POSE_KEY].shape == (9,)
        assert spec.encoder_type == "vggt"
        assert spec.obs_shape == (4116,)
        assert spec.env_render_resolution == 518
        assert spec.agent_overrides == {"buffer_capacity": 1_000_000}

    def test_aggregator_encoder_spec_uses_pooled_extractor_feature_dim(
        self, patch_vggt
    ):
        patch_vggt(aggregator_feature_shape=(86, 128))

        enc = VGGTAggregatorMLPEncoder(resolution=256)
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert isinstance(adapter.contract, EncoderInputContract)
        assert adapter.contract.encoder_type == "vggt_aggregator_mlp"
        assert adapter.contract.encoder_input.shape == (3 * 128,)
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

    def test_agg_token_transformer_spec_keeps_full_tokens_fp16(self, patch_vggt):
        patch_vggt(aggregator_feature_shape=(10, 4))

        enc = VGGTAggTokenTransformerEncoder(resolution=256)
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert adapter.buffer_shape == (10 * 4,)
        assert adapter.buffer_dtype == "float16"
        assert spec.obs_shape == (10 * 4,)
        assert spec.env_render_resolution == 256
        assert spec.encoder_type == "vggt_agg_token_transformer"
        assert spec.module_cls is TokenTransformerEncoder
        assert enc.vggt_compute_heads is False
        assert spec.agent_overrides == {
            "buffer_capacity": 5_000,
            "batch_size": 1,
            "seq_len": 8,
            "train_ratio": 32,
        }
        assert "1374" in spec.design_notes

    def test_dense_wp_encoder_exposes_image_shaped_spec(self, patch_vggt):
        patch_vggt()

        enc = VGGTDenseWPEncoder(resolution=518)
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert isinstance(adapter.contract, EncoderInputContract)
        assert adapter.contract.encoder_type == "vggt_wp_dense_cnn"
        assert (
            adapter.contract.replay_observation.fields[WORLD_POINTS_KEY].dtype
            == "float16"
        )
        # Dense WP is stored channel-first as a (3, 518, 518) float16 image plus pose.
        assert adapter.buffer_shape == {
            WORLD_POINTS_KEY: (3, 518, 518),
            CAMERA_POSE_KEY: (9,),
        }
        assert adapter.buffer_dtype == {
            WORLD_POINTS_KEY: "float16",
            CAMERA_POSE_KEY: "float16",
        }
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
                return {
                    "dense_world_points": dense,
                    "camera_pose": jnp.arange(9, dtype=jnp.float32),
                }

        adapter = VGGTObsAdapter(FakeExtractor(), feature_kind="wp_dense")
        replay_features, agent_obs = adapter.transform(
            ObservationFrame(image=np.zeros((3, 4, 4), dtype=np.uint8), is_first=False)
        )

        expected = (
            np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3).transpose(2, 0, 1)
        )
        assert replay_features[WORLD_POINTS_KEY].shape == (3, 4, 4)
        assert replay_features[WORLD_POINTS_KEY].dtype == np.float16
        np.testing.assert_allclose(
            replay_features[WORLD_POINTS_KEY], expected.astype(np.float16)
        )
        assert replay_features[CAMERA_POSE_KEY].shape == (9,)
        assert agent_obs[WORLD_POINTS_KEY].shape == (3, 4, 4)
        assert agent_obs[WORLD_POINTS_KEY].dtype.name == "float16"
        assert agent_obs[CAMERA_POSE_KEY].shape == (9,)

    def test_wp_cp_64_encoder_spec(self, patch_vggt):
        patch_vggt()

        enc = VGGTWPCP64Encoder(resolution=518)
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert isinstance(adapter.contract, EncoderInputContract)
        assert adapter.contract.encoder_type == "vggt_wp_cp_64"
        assert adapter.contract.encoder_input.shape == (12297,)
        # 64x64 WP grid: obs = 64*64*3 + 9 = 12297 (vs 4116 at 37x37).
        assert enc.wp_pool_size == 64
        assert adapter.buffer_shape == {
            WORLD_POINTS_KEY: (3, 64, 64),
            CAMERA_POSE_KEY: (9,),
        }
        assert adapter.buffer_dtype == {
            WORLD_POINTS_KEY: "float16",
            CAMERA_POSE_KEY: "float16",
        }
        assert spec.obs_shape == (12297,)
        assert spec.encoder_type == "vggt_wp_cp_64"
        # Same MLP module + 1M buffer as the 37x37 WP+CP run -> resolution-only ablation.
        assert spec.module_cls is MLPEncoder
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
        np.testing.assert_allclose(
            np.asarray(p37), np.asarray(exact), rtol=1e-5, atol=1e-3
        )
        assert np.isfinite(np.asarray(p64)).all()

    def test_wp_cp_64_adapter_flattens_world_points_plus_pose(self):
        adapter = VGGTObsAdapter(_WorldPoints64Extractor(), feature_kind="wp_cp")
        assert adapter.buffer_shape == {
            WORLD_POINTS_KEY: (3, 64, 64),
            CAMERA_POSE_KEY: (9,),
        }
        rep, agent_obs = adapter.transform(
            ObservationFrame(image=np.zeros((3, 518, 518), np.uint8), is_first=False)
        )
        assert rep[WORLD_POINTS_KEY].shape == (3, 64, 64)
        assert rep[WORLD_POINTS_KEY].dtype == np.float16
        np.testing.assert_allclose(rep[CAMERA_POSE_KEY], np.arange(9, dtype=np.float16))
        assert agent_obs[WORLD_POINTS_KEY].shape == (3, 64, 64)
        assert agent_obs[CAMERA_POSE_KEY].shape == (9,)

    def test_wp64_cnn_cp_mlp_adapter_emits_float16_world_points_and_pose(self):
        adapter = VGGTObsAdapter(
            _WorldPoints64Extractor(),
            feature_kind="wp64_cp",
            encoder_type="vggt_wp64_cnn_cp_mlp",
            encoder_module_cls=WP64CNNCPMLPEncoder,
        )

        assert adapter.buffer_shape == {
            WORLD_POINTS_KEY: (3, 64, 64),
            CAMERA_POSE_KEY: (9,),
        }
        assert adapter.buffer_dtype == {
            WORLD_POINTS_KEY: "float16",
            CAMERA_POSE_KEY: "float16",
        }
        rep, agent_obs = adapter.transform(
            ObservationFrame(image=np.zeros((3, 518, 518), np.uint8), is_first=False)
        )
        assert rep[WORLD_POINTS_KEY].shape == (3, 64, 64)
        assert rep[WORLD_POINTS_KEY].dtype == np.float16
        assert rep[CAMERA_POSE_KEY].shape == (9,)
        assert rep[CAMERA_POSE_KEY].dtype == np.float16
        assert agent_obs[WORLD_POINTS_KEY].shape == (3, 64, 64)
        assert agent_obs[CAMERA_POSE_KEY].shape == (9,)

    def test_wp64_cnn_cp_mlp_encoder_spec_uses_structured_obs_shape(self, patch_vggt):
        patch_vggt()

        enc = VGGTWP64CNNCPMLPEncoder(resolution=518)
        adapter = enc.make_adapter()
        spec = enc.spec()

        expected_shape = {
            WORLD_POINTS_KEY: (3, 64, 64),
            CAMERA_POSE_KEY: (9,),
        }
        assert adapter.contract.encoder_type == "vggt_wp64_cnn_cp_mlp"
        assert adapter.contract.encoder_input.buffer_shape() == expected_shape
        assert spec.obs_shape == expected_shape
        assert spec.encoder_type == "vggt_wp64_cnn_cp_mlp"
        assert spec.module_cls is WP64CNNCPMLPEncoder
        assert spec.agent_overrides == {"buffer_capacity": 1_000_000}

    def test_vggt_launcher_variants_are_centralized(self):
        assert VGGTEncoder.variant is VGGT_VARIANTS["vggt"]
        assert VGGTAggregatorMLPEncoder.variant is VGGT_VARIANTS["vggt_aggregator_mlp"]
        assert (
            VGGTAggTokenTransformerEncoder.variant
            is VGGT_VARIANTS["vggt_agg_token_transformer"]
        )
        assert VGGTDenseWPEncoder.variant is VGGT_VARIANTS["vggt_wp_dense_cnn"]
        assert VGGTWPCP64Encoder.variant is VGGT_VARIANTS["vggt_wp_cp_64"]
        assert VGGTWP64CNNCPMLPEncoder.variant is VGGT_VARIANTS["vggt_wp64_cnn_cp_mlp"]

        assert VGGT_VARIANTS["vggt"].compute_heads is True
        assert VGGT_VARIANTS["vggt_aggregator_mlp"].compute_heads is False
        assert VGGT_VARIANTS["vggt_agg_token_transformer"].compute_heads is False
        assert VGGT_VARIANTS["vggt_wp_cp_64"].wp_pool_size == 64

    def test_aggregator_adapter_emits_cam_mean_max_pools(self):
        # Fake extractor with 1 cam + 4 register + 5 patch tokens, D = 4.
        # tokens = arange(40).reshape(10, 4); patches = tokens[5:].
        adapter = VGGTObsAdapter(_AggregatorTokensExtractor(), feature_kind="aggregator")
        replay_features, agent_obs = adapter.transform(
            ObservationFrame(image=np.zeros((3, 4, 4), dtype=np.uint8), is_first=False)
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
        adapter = VGGTObsAdapter(_AggregatorTokensExtractor(), feature_kind="agg_tokens")
        replay_features, agent_obs = adapter.transform(
            ObservationFrame(image=np.zeros((3, 4, 4), dtype=np.uint8), is_first=False)
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

    def test_hybrid_encoder_exposes_spec(self, patch_vggt):
        patch_vggt()

        enc = HybridEncoder()
        adapter = enc.make_adapter()
        spec = enc.spec()
        assert isinstance(adapter.contract, EncoderInputContract)
        assert adapter.contract.encoder_type == "hybrid"
        assert adapter.contract.decoder_target is not None
        assert isinstance(spec, EncoderSpec)
        assert spec.encoder_type == "hybrid"
        assert spec.obs_shape == (16404,)
        assert spec.env_render_resolution == 518
        assert spec.module_cls is ModelHybridEncoder

    def test_hybrid_adapter_builds_rgb_wp_cp_layout(self):
        # Fake VGGT extractor: extract() -> world_points (37,37,3) + camera_pose (9,)
        # so the hybrid WP/CP readout yields 37*37*3 + 9 = 4116.
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
        assert adapter.buffer_shape == {
            HYBRID_IMAGE_KEY: (3, 64, 64),
            HYBRID_WP_CP_KEY: (4116,),
        }
        assert adapter.buffer_dtype == {
            HYBRID_IMAGE_KEY: "uint8",
            HYBRID_WP_CP_KEY: "float32",
        }
        assert adapter.normalize_on_sample == {
            HYBRID_IMAGE_KEY: False,
            HYBRID_WP_CP_KEY: False,
        }
        assert adapter.encoder_obs_shape == (16404,)

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(3, 518, 518), dtype=np.uint8)
        env_obs = ObservationFrame(image=image, is_first=True)

        replay, agent_obs = adapter.transform(env_obs)

        assert set(replay) == {HYBRID_IMAGE_KEY, HYBRID_WP_CP_KEY}
        assert replay[HYBRID_IMAGE_KEY].shape == (3, 64, 64)
        assert replay[HYBRID_IMAGE_KEY].dtype == np.uint8
        assert replay[HYBRID_WP_CP_KEY].shape == (4116,)
        assert replay[HYBRID_WP_CP_KEY].dtype == np.float32

        # RGB field: raw 64x64 uint8 resize of the input.
        img64 = resize_chw_uint8(image, 64)  # (3,64,64) uint8
        np.testing.assert_array_equal(replay[HYBRID_IMAGE_KEY], img64)

        # WP/CP field: flattened world_points then camera_pose.
        expected_wp_cp = np.concatenate([world_points.reshape(-1), camera_pose]).astype(
            np.float32
        )
        np.testing.assert_allclose(replay[HYBRID_WP_CP_KEY], expected_wp_cp)

        assert agent_obs[HYBRID_IMAGE_KEY].shape == (3, 64, 64)
        assert np.asarray(agent_obs[HYBRID_WP_CP_KEY]).shape == (4116,)


class TestVGGTHouseContextEncoder:
    def test_house_context_encoder_static_path_skips_vggt_extractor(
        self, monkeypatch, tmp_path
    ):
        class FailingExtractor:
            def __init__(self, **kwargs):
                raise AssertionError("static context path should not build VGGT")

        monkeypatch.setattr(
            "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor", FailingExtractor
        )
        ply_path = tmp_path / "house.ply"
        _write_minimal_static_house_ply(ply_path)

        enc = VGGTHouseContextEncoder(static_house_context_path=str(ply_path))
        adapter = enc.make_adapter()
        spec = enc.spec()
        image = np.random.default_rng(1).integers(
            0, 256, size=(3, 518, 518), dtype=np.uint8
        )

        replay, agent_obs = adapter.transform(
            ObservationFrame(image=image, is_first=True)
        )

        assert isinstance(adapter, VGGTHouseContextObsAdapter)
        assert spec.obs_shape == (13312,)
        assert "static RGB point-cloud" in spec.design_notes
        assert replay[HOUSE_CONTEXT_KEY].shape == (1024,)
        assert replay[HOUSE_CONTEXT_KEY].dtype == np.float16
        assert agent_obs[HOUSE_CONTEXT_KEY].shape == (1024,)

    def test_house_context_encoder_exposes_rgb_replay_spec(self, patch_vggt):
        patch_vggt()

        enc = VGGTHouseContextEncoder()
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert isinstance(adapter, VGGTHouseContextObsAdapter)
        assert spec.encoder_type == "vggt_house_context"
        assert adapter.buffer_shape == {
            HYBRID_IMAGE_KEY: (3, 64, 64),
            HOUSE_CONTEXT_KEY: (1024,),
        }
        assert spec.obs_shape == (13312,)
        assert spec.env_render_resolution == 518
        assert spec.module_cls is ModelHybridEncoder
        assert spec.agent_overrides["buffer_capacity"] == 1_000_000
        assert spec.agent_overrides["vggt_feature_dim"] == 1024
        assert spec.agent_overrides["vggt_token_dim"] == 2048
        assert spec.agent_overrides["vggt_token_transformer_layers"] == 2
        assert spec.agent_overrides["vggt_token_transformer_heads"] == 8
        assert spec.agent_overrides["vggt_token_transformer_dropout"] == 0.0
        assert adapter.on_episode_reset is None

    def test_house_context_adapter_stores_rgb_and_context_in_replay(self):
        full_tokens = (
            np.arange(1374 * 2048, dtype=np.float32).reshape(1374, 2048) / 1000.0
        )
        context = np.arange(1024, dtype=np.float32)

        class FakeExtractor:
            def extract(self, image):
                import jax.numpy as jnp

                return {"aggregator_full_tokens": jnp.asarray(full_tokens)}

        class FakeContextTransformer:
            def init(self, rng, tokens, *, train=False):
                assert tokens.shape == (1, 1374, 2048)
                return {"params": {}}

            def apply(self, params, tokens, *, train=False):
                assert tokens.shape == (1374, 2048)
                import jax.numpy as jnp

                return jnp.asarray(context)

        adapter = VGGTHouseContextObsAdapter(
            FakeExtractor(), context_transformer=FakeContextTransformer()
        )
        image = np.random.default_rng(0).integers(
            0, 256, size=(3, 518, 518), dtype=np.uint8
        )

        replay, agent_obs = adapter.transform(
            ObservationFrame(image=image, is_first=True)
        )

        assert set(replay) == {HYBRID_IMAGE_KEY, HOUSE_CONTEXT_KEY}
        assert replay[HYBRID_IMAGE_KEY].shape == (3, 64, 64)
        assert replay[HYBRID_IMAGE_KEY].dtype == np.uint8
        assert replay[HOUSE_CONTEXT_KEY].shape == (1024,)
        assert replay[HOUSE_CONTEXT_KEY].dtype == np.float16
        assert set(agent_obs) == {HYBRID_IMAGE_KEY, HOUSE_CONTEXT_KEY, "is_first"}
        assert agent_obs[HOUSE_CONTEXT_KEY].shape == (1024,)

        batch = ReplayBatch(
            obs={
                HYBRID_IMAGE_KEY: np.zeros((2, 3, 3, 64, 64), dtype=np.uint8),
                HOUSE_CONTEXT_KEY: np.zeros((2, 3, 1024), dtype=np.float16),
            },
            actions=np.zeros((2, 3), dtype=np.int32),
            rewards=np.zeros((2, 3), dtype=np.float32),
            is_episode_end=np.zeros((2, 3), dtype=bool),
            is_first=np.zeros((2, 3), dtype=np.float32),
        )
        augmented = adapter.augment_replay_batch(batch)

        assert set(augmented.obs) == {HYBRID_IMAGE_KEY, HOUSE_CONTEXT_KEY}
        assert augmented.obs[HYBRID_IMAGE_KEY].shape == (2, 3, 3, 64, 64)
        assert augmented.obs[HOUSE_CONTEXT_KEY].shape == (2, 3, 1024)

        np.testing.assert_allclose(np.asarray(agent_obs[HOUSE_CONTEXT_KEY]), context)


class _FakeHousePointsExtractor:
    """Fake VGGT extractor returning aligned live world points / pose.

    ``extract`` returns a small VGGT-like output whose ``world_points`` (H, W, 3)
    and ``confidence`` (H, W) flatten in lockstep with an (3, H, W) frame image
    (H = W = 8 -> flat length 64). Each call advances an internal RNG so distinct
    world points are produced, letting growth/isolation tests add new voxels.
    """

    aggregator_feature_shape = (1374, 1024)

    def __init__(self, seed: int = 0, **kwargs):
        self.kwargs = kwargs
        self._rng = np.random.default_rng(seed)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def extract(self, image):
        del image
        # Spread points across a few metres so distinct voxels are created.
        world_points = self._rng.uniform(-1.0, 1.0, size=(8, 8, 3)).astype(
            np.float32
        )
        confidence = np.full((8, 8), 5.0, dtype=np.float32)
        camera_pose = np.arange(9, dtype=np.float32)

        class _Out:
            pass

        out = _Out()
        out.world_points = world_points
        out.confidence = confidence
        out.camera_pose = camera_pose
        return out


class _MappingHousePointsExtractor(_FakeHousePointsExtractor):
    """Legacy mapping-style fake output for adapter compatibility coverage."""

    def extract(self, image):
        out = super().extract(image)
        return {
            "dense_world_points": out.world_points,
            "confidence": out.confidence,
            CAMERA_POSE_KEY: out.camera_pose,
        }


def _house_frame(seed: int, *, is_first=False, scene_id="houseA"):
    image = np.random.default_rng(seed).integers(0, 256, size=(3, 8, 8), dtype=np.uint8)
    return ObservationFrame(image=image, is_first=is_first, scene_id=scene_id)


class TestVGGTHousePointsPoseEncoder:
    def test_house_points_pose_adapter_replays_only_camera_pose(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor",
            _FakeHousePointsExtractor,
        )
        ply_path = tmp_path / "house.ply"
        _write_minimal_static_house_ply(ply_path)

        enc = VGGTHousePointsPoseEncoder(house_points_path=str(ply_path))
        adapter = enc.make_adapter()
        spec = enc.spec()

        replay, agent_obs = adapter.transform(_house_frame(2))

        assert isinstance(adapter, VGGTHousePointsPoseObsAdapter)
        assert spec.encoder_type == "vggt_house_points_pose"
        assert spec.obs_shape == {
            CAMERA_POSE_KEY: (9,),
            HOUSE_CONTEXT_KEY: (HOUSE_CONTEXT_MAX_POINTS, HOUSE_POINT_DIM),
            HOUSE_CONTEXT_SIZE_KEY: (),
        }
        assert replay.keys() == {CAMERA_POSE_KEY}
        assert replay[CAMERA_POSE_KEY].dtype == np.float16
        assert set(agent_obs) == {
            CAMERA_POSE_KEY,
            HOUSE_CONTEXT_KEY,
            HOUSE_CONTEXT_SIZE_KEY,
            "is_first",
        }
        assert agent_obs[HOUSE_CONTEXT_KEY].shape == (
            HOUSE_CONTEXT_MAX_POINTS,
            HOUSE_POINT_DIM,
        )
        assert agent_obs[HOUSE_CONTEXT_SIZE_KEY].shape == ()
        assert agent_obs[HOUSE_CONTEXT_SIZE_KEY].dtype == np.int32
        assert int(agent_obs[HOUSE_CONTEXT_SIZE_KEY]) > 0

    def test_augment_replay_batch_injects_two_dim_house_context(self, monkeypatch):
        monkeypatch.setattr(
            "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor",
            _FakeHousePointsExtractor,
        )
        enc = VGGTHousePointsPoseEncoder()
        adapter = enc.make_adapter()

        adapter.transform(_house_frame(2))

        batch = ReplayBatch(
            obs={CAMERA_POSE_KEY: np.zeros((1, 2, 9), dtype=np.float16)},
            actions=np.zeros((1, 2), dtype=np.int32),
            rewards=np.zeros((1, 2), dtype=np.float32),
            is_episode_end=np.zeros((1, 2), dtype=bool),
            is_first=np.zeros((1, 2), dtype=np.float32),
        )
        augmented = adapter.augment_replay_batch(batch)

        assert set(augmented.obs) == {
            CAMERA_POSE_KEY,
            HOUSE_CONTEXT_KEY,
            HOUSE_CONTEXT_SIZE_KEY,
        }
        assert augmented.obs[CAMERA_POSE_KEY].shape == (1, 2, 9)
        assert augmented.obs[HOUSE_CONTEXT_KEY].shape == (
            HOUSE_CONTEXT_MAX_POINTS,
            HOUSE_POINT_DIM,
        )
        assert augmented.obs[HOUSE_CONTEXT_SIZE_KEY].shape == ()
        assert augmented.obs[HOUSE_CONTEXT_SIZE_KEY].dtype == np.int32

    def test_mapping_vggt_output_is_supported(self):
        adapter = VGGTHousePointsPoseObsAdapter(
            _MappingHousePointsExtractor(seed=4),
            confidence_score=1.0,
            voxel_size_m=0.05,
        )

        replay, agent_obs = adapter.transform(_house_frame(5))

        assert replay.keys() == {CAMERA_POSE_KEY}
        assert replay[CAMERA_POSE_KEY].shape == (9,)
        assert agent_obs[HOUSE_CONTEXT_KEY].shape == (
            HOUSE_CONTEXT_MAX_POINTS,
            HOUSE_POINT_DIM,
        )
        assert int(agent_obs[HOUSE_CONTEXT_SIZE_KEY]) > 0

    def test_max_input_points_stride_caps_subsampled_frame(self):
        adapter = VGGTHousePointsPoseObsAdapter(
            _FakeHousePointsExtractor(seed=9),
            max_input_points=10,
        )

        stride = adapter._input_stride(height=8, width=8)

        assert stride == 3
        assert len(range(0, 8, stride)) ** 2 <= 10

    def test_buffer_grows_across_steps_for_same_scene(self):
        extractor = _FakeHousePointsExtractor(seed=7)
        adapter = VGGTHousePointsPoseObsAdapter(
            extractor, confidence_score=1.0, voxel_size_m=0.05
        )

        adapter.transform(_house_frame(10, scene_id="houseA"))
        first = adapter._buffers["houseA"].points_xyz.shape[0]
        adapter.transform(_house_frame(11, scene_id="houseA"))
        second = adapter._buffers["houseA"].points_xyz.shape[0]

        assert set(adapter._buffers) == {"houseA"}
        assert second > first

    def test_buffers_are_isolated_per_scene(self):
        extractor = _FakeHousePointsExtractor(seed=3)
        adapter = VGGTHousePointsPoseObsAdapter(
            extractor, confidence_score=1.0, voxel_size_m=0.05
        )

        adapter.transform(_house_frame(20, scene_id="houseA"))
        adapter.transform(_house_frame(21, scene_id="houseB"))

        assert set(adapter._buffers) == {"houseA", "houseB"}
        assert adapter._buffers["houseA"].points_xyz.shape[0] > 0
        assert adapter._buffers["houseB"].points_xyz.shape[0] > 0
        assert adapter._buffers["houseA"] is not adapter._buffers["houseB"]

    def test_house_points_path_is_optional_seed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor",
            _FakeHousePointsExtractor,
        )

        # No path: encoder/adapter build and transform without any seed PLY.
        enc = VGGTHousePointsPoseEncoder()
        adapter = enc.make_adapter()
        assert isinstance(adapter, VGGTHousePointsPoseObsAdapter)
        replay, agent_obs = adapter.transform(_house_frame(2))
        assert replay.keys() == {CAMERA_POSE_KEY}
        assert agent_obs[HOUSE_CONTEXT_KEY].shape == (
            HOUSE_CONTEXT_MAX_POINTS,
            HOUSE_POINT_DIM,
        )

        # With a small static PLY warm-start seed: still valid.
        ply_path = tmp_path / "house.ply"
        _write_minimal_static_house_ply(ply_path)
        seeded_enc = VGGTHousePointsPoseEncoder(house_points_path=str(ply_path))
        seeded_adapter = seeded_enc.make_adapter()
        assert isinstance(seeded_adapter, VGGTHousePointsPoseObsAdapter)
        seeded_adapter.transform(_house_frame(2))
        # The seed points are registered in the per-scene buffer.
        assert seeded_adapter._buffers["houseA"].points_xyz.shape[0] > 0


class TestVGGTHybridHousePointsPoseEncoder:
    def test_hybrid_adapter_replays_camera_pose_and_image(self, monkeypatch):
        monkeypatch.setattr(
            "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor",
            _FakeHousePointsExtractor,
        )
        enc = VGGTHybridHousePointsPoseEncoder()
        adapter = enc.make_adapter()
        spec = enc.spec()

        replay, agent_obs = adapter.transform(_house_frame(2))

        assert isinstance(adapter, VGGTHybridHousePointsPoseObsAdapter)
        assert spec.encoder_type == "vggt_hybrid_house_points_pose"
        assert spec.obs_shape == {
            CAMERA_POSE_KEY: (9,),
            HOUSE_CONTEXT_KEY: (HOUSE_CONTEXT_MAX_POINTS, HOUSE_POINT_DIM),
            HOUSE_CONTEXT_SIZE_KEY: (),
            HYBRID_IMAGE_KEY: (3, 64, 64),
        }
        assert replay.keys() == {CAMERA_POSE_KEY, HYBRID_IMAGE_KEY}
        assert replay[CAMERA_POSE_KEY].dtype == np.float16
        assert replay[HYBRID_IMAGE_KEY].dtype == np.uint8
        assert replay[HYBRID_IMAGE_KEY].shape == (3, 64, 64)
        assert set(agent_obs) == {
            CAMERA_POSE_KEY,
            HOUSE_CONTEXT_KEY,
            HOUSE_CONTEXT_SIZE_KEY,
            HYBRID_IMAGE_KEY,
            "is_first",
        }
        assert agent_obs[HYBRID_IMAGE_KEY].shape == (3, 64, 64)
        assert int(agent_obs[HOUSE_CONTEXT_SIZE_KEY]) > 0

    def test_hybrid_adapter_keeps_live_house_context_injection(self, monkeypatch):
        monkeypatch.setattr(
            "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor",
            _FakeHousePointsExtractor,
        )
        enc = VGGTHybridHousePointsPoseEncoder()
        adapter = enc.make_adapter()
        adapter.transform(_house_frame(2))

        batch = ReplayBatch(
            obs={
                CAMERA_POSE_KEY: np.zeros((1, 2, 9), dtype=np.float16),
                HYBRID_IMAGE_KEY: np.zeros((1, 2, 3, 64, 64), dtype=np.float32),
            },
            actions=np.zeros((1, 2), dtype=np.int32),
            rewards=np.zeros((1, 2), dtype=np.float32),
            is_episode_end=np.zeros((1, 2), dtype=bool),
            is_first=np.zeros((1, 2), dtype=np.float32),
        )
        augmented = adapter.augment_replay_batch(batch)

        assert set(augmented.obs) == {
            CAMERA_POSE_KEY,
            HYBRID_IMAGE_KEY,
            HOUSE_CONTEXT_KEY,
            HOUSE_CONTEXT_SIZE_KEY,
        }
        assert augmented.obs[HYBRID_IMAGE_KEY].shape == (1, 2, 3, 64, 64)


class TestVGGTHouseFullTokenNoGateEncoder:
    def test_full_token_nogate_encoder_exposes_image_replay_and_token_obs(
        self, patch_vggt
    ):
        patch_vggt()

        enc = VGGTHouseFullTokenNoGateEncoder()
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert isinstance(adapter, VGGTHouseFullTokenObsAdapter)
        assert spec.encoder_type == "vggt_house_full_tokens_nogate"
        assert adapter.buffer_shape == {
            HYBRID_IMAGE_KEY: (3, 64, 64),
            FULL_TOKENS_KEY: (1374, 2048),
        }
        assert spec.obs_shape == {
            HYBRID_IMAGE_KEY: (3, 64, 64),
            FULL_TOKENS_KEY: (1374, 2048),
        }
        assert spec.module_cls is TokenTransformerEncoder
        assert spec.agent_overrides["buffer_capacity"] == 5_000
        assert spec.agent_overrides["vggt_token_dim"] == 2048
        assert spec.agent_overrides["vggt_token_count"] == 1374
        assert adapter.on_episode_reset is None

    def test_full_token_adapter_stores_rgb_and_tokens_in_replay(self):
        full_tokens = (
            np.arange(1374 * 2048, dtype=np.float32).reshape(1374, 2048) / 1000.0
        )

        class FakeExtractor:
            def extract(self, image):
                import jax.numpy as jnp

                return {"aggregator_full_tokens": jnp.asarray(full_tokens)}

        adapter = VGGTHouseFullTokenObsAdapter(FakeExtractor())
        image = np.random.default_rng(1).integers(
            0, 256, size=(3, 518, 518), dtype=np.uint8
        )

        replay, agent_obs = adapter.transform(
            ObservationFrame(image=image, is_first=True)
        )

        assert set(replay) == {HYBRID_IMAGE_KEY, FULL_TOKENS_KEY}
        assert replay[HYBRID_IMAGE_KEY].shape == (3, 64, 64)
        assert replay[HYBRID_IMAGE_KEY].dtype == np.uint8
        assert replay[FULL_TOKENS_KEY].shape == (1374, 2048)
        assert replay[FULL_TOKENS_KEY].dtype == np.float16
        assert set(agent_obs) == {HYBRID_IMAGE_KEY, FULL_TOKENS_KEY, "is_first"}
        assert agent_obs[FULL_TOKENS_KEY].shape == (1374, 2048)

        batch = ReplayBatch(
            obs={
                HYBRID_IMAGE_KEY: np.zeros((2, 3, 3, 64, 64), dtype=np.uint8),
                FULL_TOKENS_KEY: np.zeros((2, 3, 1374, 2048), dtype=np.float16),
            },
            actions=np.zeros((2, 3), dtype=np.int32),
            rewards=np.zeros((2, 3), dtype=np.float32),
            is_episode_end=np.zeros((2, 3), dtype=bool),
            is_first=np.zeros((2, 3), dtype=np.float32),
        )
        augmented = adapter.augment_replay_batch(batch)

        assert set(augmented.obs) == {HYBRID_IMAGE_KEY, FULL_TOKENS_KEY}
        assert augmented.obs[HYBRID_IMAGE_KEY].shape == (2, 3, 3, 64, 64)
        assert augmented.obs[FULL_TOKENS_KEY].shape == (2, 3, 1374, 2048)


class TestVGGTHouseGlobalTokenNoGateEncoder:
    def test_global_token_nogate_encoder_exposes_rgb_replay_and_token_obs(
        self, patch_vggt
    ):
        from src.r2dreamer.adapters.token_adapters import VGGTHouseGlobalTokenObsAdapter
        from src.r2dreamer.encoders import VGGTHouseGlobalTokenNoGateEncoder

        patch_vggt()

        enc = VGGTHouseGlobalTokenNoGateEncoder()
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert isinstance(adapter, VGGTHouseGlobalTokenObsAdapter)
        assert spec.encoder_type == "vggt_house_global_tokens_nogate"
        assert adapter.buffer_shape == {
            HYBRID_IMAGE_KEY: (3, 64, 64),
            GLOBAL_TOKENS_KEY: (1374, 1024),
        }
        assert spec.obs_shape == {
            HYBRID_IMAGE_KEY: (3, 64, 64),
            GLOBAL_TOKENS_KEY: (1374, 1024),
        }
        assert spec.module_cls is TokenTransformerEncoder
        assert spec.agent_overrides["buffer_capacity"] == 5_000
        assert spec.agent_overrides["vggt_token_dim"] == 1024
        assert spec.agent_overrides["vggt_token_count"] == 1374
        assert adapter.on_episode_reset is None

    def test_global_token_adapter_stores_rgb_and_tokens_in_replay(self):
        from src.r2dreamer.adapters.token_adapters import VGGTHouseGlobalTokenObsAdapter

        global_tokens = (
            np.arange(1374 * 1024, dtype=np.float32).reshape(1374, 1024) / 1000.0
        )

        class FakeExtractor:
            def __init__(self):
                self.calls = 0

            def extract(self, image):
                self.calls += 1
                import jax.numpy as jnp

                return {"aggregator_features": jnp.asarray(global_tokens)}

        extractor = FakeExtractor()
        adapter = VGGTHouseGlobalTokenObsAdapter(extractor)
        image = np.random.default_rng(2).integers(
            0, 256, size=(3, 518, 518), dtype=np.uint8
        )

        replay, agent_obs = adapter.transform(
            ObservationFrame(image=image, is_first=True)
        )

        assert set(replay) == {HYBRID_IMAGE_KEY, GLOBAL_TOKENS_KEY}
        assert replay[HYBRID_IMAGE_KEY].shape == (3, 64, 64)
        assert replay[HYBRID_IMAGE_KEY].dtype == np.uint8
        assert replay[GLOBAL_TOKENS_KEY].shape == (1374, 1024)
        assert replay[GLOBAL_TOKENS_KEY].dtype == np.float16
        assert set(agent_obs) == {HYBRID_IMAGE_KEY, GLOBAL_TOKENS_KEY, "is_first"}
        assert agent_obs[GLOBAL_TOKENS_KEY].shape == (1374, 1024)
        assert extractor.calls == 1

        batch = ReplayBatch(
            obs={
                HYBRID_IMAGE_KEY: np.zeros((2, 3, 3, 64, 64), dtype=np.uint8),
                GLOBAL_TOKENS_KEY: np.zeros((2, 3, 1374, 1024), dtype=np.float16),
            },
            actions=np.zeros((2, 3), dtype=np.int32),
            rewards=np.zeros((2, 3), dtype=np.float32),
            is_episode_end=np.zeros((2, 3), dtype=bool),
            is_first=np.zeros((2, 3), dtype=np.float32),
        )
        augmented = adapter.augment_replay_batch(batch)

        assert extractor.calls == 1, "replay augmentation must not run VGGT"
        assert set(augmented.obs) == {HYBRID_IMAGE_KEY, GLOBAL_TOKENS_KEY}
        assert augmented.obs[HYBRID_IMAGE_KEY].shape == (2, 3, 3, 64, 64)
        assert augmented.obs[GLOBAL_TOKENS_KEY].shape == (2, 3, 1374, 1024)


class TestVGGTHouseGlobalEmbeddingEncoder:
    def test_global_embedding_encoder_exposes_split_token_replay_spec(
        self, patch_vggt
    ):
        patch_vggt()

        enc = VGGTHouseGlobalEmbeddingEncoder()
        adapter = enc.make_adapter()
        spec = enc.spec()

        assert isinstance(adapter, VGGTHouseGlobalEmbeddingObsAdapter)
        assert spec.encoder_type == "vggt_house_global_embedding"
        assert enc.vggt_reset_mode is ResetMode.PERSIST_SCENE
        assert enc.vggt_compute_heads is False
        expected_shape = {
            HYBRID_IMAGE_KEY: (3, 64, 64),
            CAMERA_TOKEN_GLOBAL_KEY: (1, 1024),
            GLOBAL_PATCH_TOKENS_KEY: (1369, 1024),
        }
        assert adapter.buffer_shape == expected_shape
        assert adapter.buffer_dtype == {
            HYBRID_IMAGE_KEY: "uint8",
            CAMERA_TOKEN_GLOBAL_KEY: "float16",
            GLOBAL_PATCH_TOKENS_KEY: "float16",
        }
        assert spec.obs_shape == expected_shape
        assert spec.module_cls is ModelHouseGlobalEmbeddingEncoder
        assert spec.env_render_resolution == 518
        assert spec.agent_overrides["buffer_capacity"] == 5_000
        # Scene-aware reset fires during prefill (prefill-orphaning fix).
        assert adapter.on_episode_reset is not None
        adapter.on_episode_reset("house-7")
        assert adapter._extractor.last_scene_id == "house-7"
        # Point head is off by default (PERSIST_SCENE + heads off); dumps disabled.
        assert adapter._dump_enabled is False

    def test_global_embedding_adapter_dumps_when_knob_set(self, monkeypatch, tmp_path):
        class FakeExtractor(_StubExtractor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.ply_dumps = []

            def extract(self, obs):
                import jax.numpy as jnp

                return {
                    "aggregator_features": jnp.zeros((1374, 1024), dtype=jnp.float32)
                }

            def write_point_cloud_ply(self, path):
                self.ply_dumps.append(path)

        monkeypatch.setattr(
            "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor", FakeExtractor
        )

        enc = VGGTHouseGlobalEmbeddingEncoder(
            pointcloud_dump_every=3, pointcloud_dump_dir=str(tmp_path)
        )
        adapter = enc.make_adapter()
        assert adapter._dump_enabled is True
        image = np.random.default_rng(0).integers(
            0, 256, size=(3, 8, 8), dtype=np.uint8
        )
        for s in range(6):
            adapter.transform(
                ObservationFrame(image=image, is_first=(s == 0), scene_id="houseA")
            )
        # Dumps at env steps 3 and 6.
        assert len(adapter._extractor.ply_dumps) == 2


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
        camera_pose = outputs_data["camera_pose"]  # (10, 9)

        # Reset KV-cache before the sequence
        adapter._extractor.reset()

        for i in range(len(frames)):
            obs = ObservationFrame(image=frames[i], is_first=i == 0)
            features, agent_obs = adapter.transform(obs)

            expected_wp = world_points[i].transpose(2, 0, 1)
            expected_cp = camera_pose[i]

            assert set(features) == {WORLD_POINTS_KEY, CAMERA_POSE_KEY}
            assert features[WORLD_POINTS_KEY].shape == (3, 37, 37)
            assert features[WORLD_POINTS_KEY].dtype == np.float16
            assert features[CAMERA_POSE_KEY].shape == (9,)
            assert features[CAMERA_POSE_KEY].dtype == np.float16

            np.testing.assert_allclose(
                features[WORLD_POINTS_KEY],
                expected_wp.astype(np.float16),
                atol=2e-2,
                rtol=1e-2,
                err_msg=f"World-points mismatch at frame {i}",
            )
            np.testing.assert_allclose(
                features[CAMERA_POSE_KEY],
                expected_cp.astype(np.float16),
                atol=2e-2,
                rtol=1e-2,
                err_msg=f"Camera-pose mismatch at frame {i}",
            )

            assert agent_obs[WORLD_POINTS_KEY].shape == (3, 37, 37)
            assert agent_obs[WORLD_POINTS_KEY].dtype.name == "float16"
            assert agent_obs[CAMERA_POSE_KEY].shape == (9,)
            assert agent_obs[CAMERA_POSE_KEY].dtype.name == "float16"
            assert agent_obs["is_first"] is (i == 0)
