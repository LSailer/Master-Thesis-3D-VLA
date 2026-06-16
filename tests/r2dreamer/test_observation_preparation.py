"""Observation Preparation contract tests."""

import json

import numpy as np

from src.r2dreamer.observation_preparation import (
    CNNObservationPreparation,
    HYBRID_FEATURE_DIM,
    HYBRID_IMAGE_SHAPE,
    VGGTFeatureKind,
    build_hybrid_contract,
    build_vggt_contract,
    PreparedObservation,
    recover_encoder_input_contract,
)
from src.r2dreamer.obs_batch import HYBRID_IMAGE_KEY, HYBRID_WP_CP_KEY
from src.r2dreamer.world_model import encoders as wm_encoders


class TestCNNObservationPreparation:
    def test_contract_declares_cnn_replay_and_encoder_input_forms(self):
        prep = CNNObservationPreparation()
        contract = prep.contract

        assert contract.observation_preparation_type == "cnn"
        assert contract.encoder_type == "cnn"
        assert contract.env_render_resolution == 64
        assert contract.encoder_module_cls is wm_encoders.ConvEncoder
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

        assert encoded["encoder_module"] == "src.r2dreamer.world_model.encoders.ConvEncoder"
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

    def test_prepare_env_step_returns_replay_and_agent_observations(self):
        prep = CNNObservationPreparation()
        image = np.arange(3 * 64 * 64, dtype=np.uint8).reshape(3, 64, 64)

        prepared = prep.prepare_env_step({"image": image, "is_first": True})

        assert isinstance(prepared, PreparedObservation)
        np.testing.assert_array_equal(prepared.replay_obs, image)
        assert prepared.agent_obs["image"] is image
        assert prepared.agent_obs["is_first"] is True

    def test_legacy_transform_routes_through_prepared_observation(self):
        prep = CNNObservationPreparation()
        image = np.zeros((3, 64, 64), dtype=np.uint8)

        replay_obs, agent_obs = prep.transform({"image": image})

        np.testing.assert_array_equal(replay_obs, image)
        assert agent_obs["image"] is image
        assert agent_obs["is_first"] is False


class TestVGGTObservationPreparationContracts:
    class _Extractor:
        aggregator_feature_shape = (86, 128)
        image_size = 518
        wp_pool_size = 37

    def test_wp_cp_contract_declares_raw_env_replay_and_encoder_forms(self):
        contract = build_vggt_contract(self._Extractor(), feature_kind="wp_cp")

        assert contract.observation_preparation_type == "vggt"
        assert contract.encoder_type == "vggt"
        assert contract.env_render_resolution == 518
        assert contract.encoder_module_cls is wm_encoders.VGGTEncoder
        assert contract.env_observation.fields["image"].shape == (3, 518, 518)
        assert contract.env_observation.fields["image"].dtype == "uint8"
        assert contract.env_observation.fields["is_first"].dtype == "bool"

        replay = contract.replay_observation
        assert replay.shape == (37 * 37 * 3 + 9,)
        assert replay.dtype == "float32"
        assert replay.normalize_on_sample is False
        assert contract.agent_observation.fields["features"].shape == replay.shape
        assert contract.encoder_input.shape == replay.shape
        assert contract.decoder_target is None
        assert contract.agent_overrides == {"buffer_capacity": 1_000_000}

    def test_variants_derive_contract_shapes_from_extractor_metadata(self):
        cases: list[tuple[VGGTFeatureKind, str, tuple[int, ...], str, type]] = [
            ("aggregator", "vggt_aggregator_mlp", (3 * 128,), "float32", wm_encoders.VGGTAggregatorMLPEncoder),
            ("wp_dense", "vggt_wp_dense_cnn", (3, 518, 518), "float16", wm_encoders.WPConvEncoder),
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
        assert contract.replay_observation.shape == (64 * 64 * 3 + 9,)
        assert contract.encoder_input.shape == (64 * 64 * 3 + 9,)
        assert contract.encoder_module_cls is wm_encoders.VGGTEncoder

    def test_hybrid_contract_declares_structured_replay_and_decoder_target(self):
        contract = build_hybrid_contract(self._Extractor())

        assert contract.observation_preparation_type == "hybrid"
        assert contract.encoder_type == "hybrid"
        assert contract.env_render_resolution == 518
        assert contract.encoder_module_cls is wm_encoders.HybridEncoder
        assert contract.env_observation.fields["image"].shape == (3, 518, 518)
        assert contract.replay_observation.fields[HYBRID_IMAGE_KEY].shape == HYBRID_IMAGE_SHAPE
        assert contract.replay_observation.fields[HYBRID_IMAGE_KEY].dtype == "uint8"
        assert contract.replay_observation.fields[HYBRID_WP_CP_KEY].shape == (37 * 37 * 3 + 9,)
        assert contract.replay_observation.fields[HYBRID_WP_CP_KEY].dtype == "float32"
        assert contract.agent_observation.fields[HYBRID_IMAGE_KEY].shape == HYBRID_IMAGE_SHAPE
        assert contract.agent_observation.fields[HYBRID_WP_CP_KEY].shape == (37 * 37 * 3 + 9,)
        assert contract.encoder_input.shape == (HYBRID_FEATURE_DIM,)
        assert contract.decoder_target is not None
        assert contract.decoder_target.shape == HYBRID_IMAGE_SHAPE
