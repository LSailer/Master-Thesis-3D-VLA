"""Observation Preparation contract tests."""

import numpy as np

from src.r2dreamer.observation_preparation import (
    CNNObservationPreparation,
    PreparedObservation,
)
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
