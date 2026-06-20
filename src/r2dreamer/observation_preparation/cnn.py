"""CNN Observation Preparation."""

from __future__ import annotations

import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.observation_preparation.contracts import (
    EncoderInputContract,
    ObservationField,
    ObservationFormContract,
    PreparedObservation,
)
from src.r2dreamer.world_model import encoders as wm_encoders


CNN_IMAGE_SHAPE = (3, 64, 64)


def _cnn_contract() -> EncoderInputContract:
    image_uint8 = ObservationField(CNN_IMAGE_SHAPE, "uint8")
    return EncoderInputContract(
        observation_preparation_type="cnn",
        encoder_type="cnn",
        env_render_resolution=64,
        encoder_module_cls=wm_encoders.ConvEncoder,
        env_observation=ObservationFormContract(
            {
                "image": image_uint8,
                "is_first": ObservationField((), "bool"),
            }
        ),
        replay_observation=ObservationFormContract(
            ObservationField(CNN_IMAGE_SHAPE, "uint8", normalize_on_sample=True)
        ),
        agent_observation=ObservationFormContract(
            {
                "image": image_uint8,
                "is_first": ObservationField((), "bool"),
            }
        ),
        encoder_input=ObservationFormContract(
            ObservationField(CNN_IMAGE_SHAPE, "float32")
        ),
        decoder_target=ObservationFormContract(
            ObservationField(CNN_IMAGE_SHAPE, "float32")
        ),
    )


class CNNObservationPreparation(ObsAdapter):
    """Prepare 64x64 CHW RGB observations for the CNN Encoder Module."""

    def __init__(self, contract: EncoderInputContract | None = None):
        self.contract = contract or _cnn_contract()
        super().__init__(
            buffer_dtype=self.contract.replay_observation.buffer_dtype(),
            buffer_shape=self.contract.replay_observation.buffer_shape(),
            normalize_on_sample=self.contract.replay_observation.buffer_normalize(),
            agent_obs_shape=self.contract.encoder_input.shape,
        )

    def prepare_env_step(self, env_obs: ObservationFrame) -> PreparedObservation:
        image = np.asarray(env_obs.image)
        return PreparedObservation(
            replay_obs=image,
            agent_obs={
                "image": image,
                "is_first": env_obs.is_first,
            },
        )

    def transform(self, env_obs: ObservationFrame):
        """Compatibility wrapper for ObsAdapter call sites."""
        prepared = self.prepare_env_step(env_obs)
        return prepared.replay_obs, prepared.agent_obs
