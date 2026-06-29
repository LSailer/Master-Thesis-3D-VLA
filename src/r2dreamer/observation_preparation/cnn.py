"""CNN Observation Preparation."""

from __future__ import annotations

import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.configs.config import (
    ObservationDims,
    ObservationRunConfig,
    ReplayObservationConfig,
)
from src.r2dreamer.observation_preparation.contracts import (
    EncoderInputContract,
    ObservationField,
    ObservationFormContract,
    PreparedObservation,
    replay_observation_form,
)
from src.r2dreamer.world_model import encoders as wm_encoders


CNN_OBSERVATION_CONFIG = ObservationRunConfig(
    encoder="cnn",
    dims=ObservationDims(render_size=64, replay_image_size=64),
    replay=ReplayObservationConfig(components=("image",), normalize_image=True),
)
CNN_IMAGE_SHAPE = CNN_OBSERVATION_CONFIG.dims.image_shape


def _cnn_contract(
    config: ObservationRunConfig = CNN_OBSERVATION_CONFIG,
) -> EncoderInputContract:
    image_uint8 = ObservationField(config.dims.image_shape, config.replay.image_dtype)
    return EncoderInputContract(
        observation_preparation_type="cnn",
        encoder_type="cnn",
        env_render_resolution=config.dims.render_size,
        encoder_module_cls=wm_encoders.ConvEncoder,
        env_observation=ObservationFormContract(
            {
                "image": image_uint8,
                "is_first": ObservationField((), "bool"),
            }
        ),
        replay_observation=replay_observation_form(config),
        agent_observation=ObservationFormContract(
            {
                "image": image_uint8,
                "is_first": ObservationField((), "bool"),
            }
        ),
        encoder_input=ObservationFormContract(
            ObservationField(config.dims.image_shape, "float32")
        ),
        decoder_target=ObservationFormContract(
            ObservationField(config.dims.image_shape, "float32")
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

    def prepare_env_step(self, env_obs: ObservationFrame, packer) -> PreparedObservation:
        image = np.asarray(env_obs.image)
        step_obs = {
            "image": image,
            "is_first": env_obs.is_first,
        }
        return PreparedObservation(
            replay_obs=image,
            encoder_obs=packer.from_step(step_obs),
            is_first=bool(env_obs.is_first),
        )

    def transform(self, env_obs: ObservationFrame):
        """Compatibility wrapper for ObsAdapter call sites."""
        image = np.asarray(env_obs.image)
        return image, {"image": image, "is_first": env_obs.is_first}
