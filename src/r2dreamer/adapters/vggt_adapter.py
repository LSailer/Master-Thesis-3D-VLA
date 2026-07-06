"""VGGTObsAdapter: wraps VGGTFeatureExtractor for RL acting."""

from __future__ import annotations

import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.observation_preparation.vggt import (
    VGGTFeatureKind,
    build_vggt_contract,
    wp_cp_dim,
)
from src.r2dreamer.observation_preparation.vggt_readouts import make_vggt_readout
from src.vggt.jax.feature_extractor import (
    JAXVGGTFeatureExtractor as VGGTFeatureExtractor,
)


VGGT_FEATURE_DIM = wp_cp_dim()  # 37*37*3 + 9

class VGGTObsAdapter(ObsAdapter):
    """Runs VGGT extraction, returns features for both buffer and agent."""

    def __init__(
        self,
        extractor: VGGTFeatureExtractor,
        feature_kind: VGGTFeatureKind = "wp_cp",
        *,
        env_render_resolution: int | None = None,
        encoder_type: str | None = None,
        encoder_module_cls=None,
        agent_overrides=None,
        design_notes: str = "",
    ):
        self.contract = build_vggt_contract(
            extractor,
            feature_kind=feature_kind,
            env_render_resolution=env_render_resolution,
            encoder_type=encoder_type,
            encoder_module_cls=encoder_module_cls,
            agent_overrides=agent_overrides,
            design_notes=design_notes,
        )
        super().__init__(
            buffer_dtype=self.contract.replay_observation.buffer_dtype(),
            buffer_shape=self.contract.replay_observation.buffer_shape(),
            normalize_on_sample=self.contract.replay_observation.buffer_normalize(),
            agent_obs_shape=self.contract.encoder_input.buffer_shape(),
            on_episode_reset=lambda scene_id="scene": extractor.reset_for_scene(scene_id),
        )
        self._extractor = extractor
        self._readout = make_vggt_readout(
            feature_kind=feature_kind,
            extractor=extractor,
            contract=self.contract,
        )

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        return self._readout.prepare(self._extractor, env_obs)
