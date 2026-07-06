"""Observation Preparation contract tests."""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import json

import numpy as np

from src.configs.config import (
    ObservationDims,
    ObservationRunConfig,
    ReplayObservationConfig,
)
from src.environments.observation import ObservationFrame
from src.r2dreamer.encoders.cnn import ConvEncoder
from src.r2dreamer.encoders.mlp import (
    HybridEncoder,
    MLPEncoder,
    VGGTAggregatorMLPEncoder,
    WP64CNNCPMLPEncoder,
)
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
    WORLD_POINTS_KEY,
)
from src.r2dreamer.observation_preparation import (
    HYBRID_FEATURE_DIM,
    HYBRID_IMAGE_SHAPE,
    VGGT_DREAMER_SPECS,
    CNNObservationPreparation,
    PreparedObservation,
    VGGTFeatureKind,
    build_hybrid_contract,
    build_vggt_contract,
    recover_encoder_input_contract,
)


class TestObservationRunConfig:
    def test_replay_shapes_are_derived_from_dimension_knobs(self):
        config = ObservationRunConfig(
            encoder="hybrid",
            dims=ObservationDims(wp_side=64),
            replay=ReplayObservationConfig(components=("image", "wp_cp")),
        )

        assert config.replay_field_shapes() == {
            "image": (3, 64, 64),
            "wp_cp": (64 * 64 * 3 + 9,),
        }
        assert config.replay_field_dtypes() == {
            "image": "uint8",
            "wp_cp": "float32",
        }
        assert config.replay_field_normalize() == {
            "image": True,
            "wp_cp": False,
        }


class TestCNNObservationPreparation:
    def test_contract_declares_cnn_replay_and_encoder_input_forms(self):
        prep = CNNObservationPreparation()
        contract = prep.contract

        assert contract.observation_preparation_type == "cnn"
        assert contract.encoder_type == "cnn"
        assert contract.env_render_resolution == 64
        assert contract.encoder_module_cls is ConvEncoder
        assert contract.encoder_input.shape == (3, 64, 64)
        assert contract.env_observation.fields["image"].shape == (3, 64, 64)
        assert contract.env_observation.fields["image"].dtype == "uint8"
        assert contract.env_observation.fields["is_first"].dtype == "bool"

        replay = contract.replay_observation
        assert replay.shape == (3, 64, 64)
        assert replay.dtype == "uint8"
        assert replay.normalize_on_sample is True

        decoder = contract.decoder_target
        assert decoder is not None
        assert decoder.shape == (3, 64, 64)
        assert decoder.dtype == "float32"

    def test_contract_snapshot_is_json_serializable_and_recoverable(self):
        contract = CNNObservationPreparation().contract

        snapshot = contract.to_snapshot()
        encoded = json.loads(json.dumps(snapshot))
        recovered = recover_encoder_input_contract(encoded)

        assert encoded["encoder_module"] == "src.r2dreamer.encoders.cnn.ConvEncoder"
        assert "encoder_module_cls" not in encoded
        assert encoded["encoder_module_kwargs"] == {}
        assert encoded["replay_observation"] == {
            "kind": "single",
            "field": {
                "shape": [3, 64, 64],
                "dtype": "uint8",
                "normalize_on_sample": True,
            },
        }
        assert recovered == contract

    def test_prepare_env_step_returns_replay_and_encoder_observation(self):
        prep = CNNObservationPreparation()
        image = np.arange(3 * 64 * 64, dtype=np.uint8).reshape(3, 64, 64)
        prepared = prep.prepare_env_step(ObservationFrame(image=image, is_first=True))

        assert isinstance(prepared, PreparedObservation)
        np.testing.assert_array_equal(prepared.replay_obs, image)
        np.testing.assert_array_equal(prepared.encoder_obs["image"], image)
        assert prepared.is_first is True

    def test_legacy_transform_routes_through_prepared_observation(self):
        prep = CNNObservationPreparation()
        image = np.zeros((3, 64, 64), dtype=np.uint8)

        replay_obs, agent_obs = prep.transform(
            ObservationFrame(image=image, is_first=False)
        )

        np.testing.assert_array_equal(replay_obs, image)
        assert agent_obs["image"] is image
        assert agent_obs["is_first"] is False


class TestVGGTObservationPreparationContracts:
    class _Extractor:
        aggregator_feature_shape = (86, 128)
        image_size = 518
        wp_pool_size = 37

    def test_storage_axis_declares_replay_vs_live_readout(self):
        assert VGGT_DREAMER_SPECS["vggt"].storage.replay_rgb is False
        assert VGGT_DREAMER_SPECS["vggt"].storage.replay_readout is True
        assert VGGT_DREAMER_SPECS["hybrid"].storage.replay_rgb is True
        assert VGGT_DREAMER_SPECS["hybrid"].storage.replay_readout is True
        assert (
            VGGT_DREAMER_SPECS["vggt_house_global_tokens_nogate"].storage.replay_readout
            is True
        )
        assert VGGT_DREAMER_SPECS["vggt_agg_raw"].readout.token_source == "flattened"
        assert (
            VGGT_DREAMER_SPECS["vggt_agg_token_transformer"].readout.token_source
            == "global"
        )

    def test_wp_cp_contract_declares_raw_env_replay_and_encoder_forms(self):
        contract = build_vggt_contract(self._Extractor(), feature_kind="wp_cp")

        assert contract.observation_preparation_type == "vggt"
        assert contract.encoder_type == "vggt"
        assert contract.env_render_resolution == 518
        assert contract.encoder_module_cls is MLPEncoder
        assert contract.env_observation.fields["image"].shape == (3, 518, 518)
        assert contract.env_observation.fields["image"].dtype == "uint8"
        assert contract.env_observation.fields["is_first"].dtype == "bool"

        replay = contract.replay_observation
        assert replay.fields[WORLD_POINTS_KEY].shape == (3, 37, 37)
        assert replay.fields[WORLD_POINTS_KEY].dtype == "float16"
        assert replay.fields[CAMERA_POSE_KEY].shape == (9,)
        assert replay.fields[CAMERA_POSE_KEY].dtype == "float16"
        assert contract.agent_observation.fields[WORLD_POINTS_KEY].shape == (3, 37, 37)
        assert contract.encoder_input.shape == (37 * 37 * 3 + 9,)
        assert contract.decoder_target is None
        assert contract.agent_overrides == {"buffer_capacity": 1_000_000}

    def test_variants_derive_contract_shapes_from_extractor_metadata(self):
        cases: list[tuple[VGGTFeatureKind, str, tuple[int, ...], str, type]] = [
            (
                "aggregator",
                "vggt_aggregator_mlp",
                (3 * 128,),
                "float32",
                VGGTAggregatorMLPEncoder,
            ),
        ]

        for feature_kind, encoder_type, shape, dtype, module_cls in cases:
            contract = build_vggt_contract(self._Extractor(), feature_kind=feature_kind)

            assert contract.observation_preparation_type == encoder_type
            assert contract.encoder_type == encoder_type
            assert contract.encoder_module_cls is module_cls
            assert contract.replay_observation.shape == shape
            assert contract.replay_observation.dtype == dtype
            assert contract.encoder_input.shape == shape
            assert contract.decoder_target is None

    def test_wp_cp_64_contract_is_resolution_ablation(self):
        class Extractor64(self._Extractor):
            wp_pool_size = 64

        contract = build_vggt_contract(Extractor64(), feature_kind="wp_cp")

        assert contract.observation_preparation_type == "vggt_wp_cp_64"
        assert contract.encoder_type == "vggt_wp_cp_64"
        assert contract.replay_observation.fields[WORLD_POINTS_KEY].shape == (3, 64, 64)
        assert contract.replay_observation.fields[CAMERA_POSE_KEY].shape == (9,)
        assert contract.encoder_input.shape == (64 * 64 * 3 + 9,)
        assert contract.encoder_module_cls is MLPEncoder

    def test_wp_dense_contract_stores_structured_wp_cp_but_encodes_world_points(self):
        contract = build_vggt_contract(self._Extractor(), feature_kind="wp_dense")

        assert contract.observation_preparation_type == "vggt_wp_dense_cnn"
        assert contract.replay_observation.fields[WORLD_POINTS_KEY].shape == (
            3,
            518,
            518,
        )
        assert contract.replay_observation.fields[CAMERA_POSE_KEY].shape == (9,)
        assert contract.encoder_input.shape == (3, 518, 518)
        assert contract.encoder_module_cls is ConvEncoder

    def test_wp64_cnn_cp_mlp_contract_uses_structured_float16_replay(self):
        class Extractor64(self._Extractor):
            wp_pool_size = 64

        contract = build_vggt_contract(Extractor64(), feature_kind="wp64_cp")

        assert contract.observation_preparation_type == "vggt_wp64_cnn_cp_mlp"
        assert contract.encoder_type == "vggt_wp64_cnn_cp_mlp"
        assert contract.encoder_module_cls is WP64CNNCPMLPEncoder
        assert contract.replay_observation.fields[WORLD_POINTS_KEY].shape == (3, 64, 64)
        assert contract.replay_observation.fields[WORLD_POINTS_KEY].dtype == "float16"
        assert contract.replay_observation.fields[CAMERA_POSE_KEY].shape == (9,)
        assert contract.replay_observation.fields[CAMERA_POSE_KEY].dtype == "float16"
        assert contract.encoder_input.fields[WORLD_POINTS_KEY].dtype == "float32"
        assert contract.encoder_input.fields[CAMERA_POSE_KEY].dtype == "float32"
        assert contract.decoder_target is None
        assert contract.agent_overrides == {"buffer_capacity": 1_000_000}

    def test_hybrid_contract_declares_structured_replay_and_decoder_target(self):
        contract = build_hybrid_contract(self._Extractor())

        assert contract.observation_preparation_type == "hybrid"
        assert contract.encoder_type == "hybrid"
        assert contract.env_render_resolution == 518
        assert contract.encoder_module_cls is HybridEncoder
        assert contract.env_observation.fields["image"].shape == (3, 518, 518)
        assert (
            contract.replay_observation.fields[HYBRID_IMAGE_KEY].shape
            == HYBRID_IMAGE_SHAPE
        )
        assert contract.replay_observation.fields[HYBRID_IMAGE_KEY].dtype == "uint8"
        assert contract.replay_observation.fields[HYBRID_WP_CP_KEY].shape == (
            37 * 37 * 3 + 9,
        )
        assert contract.replay_observation.fields[HYBRID_WP_CP_KEY].dtype == "float32"
        assert (
            contract.agent_observation.fields[HYBRID_IMAGE_KEY].shape
            == HYBRID_IMAGE_SHAPE
        )
        assert contract.agent_observation.fields[HYBRID_WP_CP_KEY].shape == (
            37 * 37 * 3 + 9,
        )
        assert contract.encoder_input.shape == (HYBRID_FEATURE_DIM,)
        assert contract.decoder_target is not None
        assert contract.decoder_target.shape == HYBRID_IMAGE_SHAPE
