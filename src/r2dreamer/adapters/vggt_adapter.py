"""VGGTObsAdapter: wraps VGGTFeatureExtractor for RL acting."""

from __future__ import annotations

import numpy as np

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.observation_preparation.vggt import (
    VGGT_AGGREGATOR_TOKEN_COUNT,
    VGGT_DEFAULT_AGGREGATOR_SHAPE,
    VGGT_FULL_TOKEN_EMBED_DIM,
    VGGTFeatureKind,
    aggregator_raw_dim,
    aggregator_token_dim,
    build_vggt_contract,
    wp_cp_dim,
)
from src.r2dreamer.observation_preparation.vggt_readouts import make_vggt_readout
from src.vggt.jax.feature_extractor import (
    JAXVGGTFeatureExtractor as VGGTFeatureExtractor,
)


VGGT_FEATURE_DIM = wp_cp_dim()  # 37*37*3 + 9

# Backwards-compatible aliases. Dimensions are derived from the shared VGGT
# observation-preparation constants instead of being hand-typed here.
AGG_RAW_DIM = aggregator_raw_dim(VGGT_DEFAULT_AGGREGATOR_SHAPE)
AGG_RAW_TOKENS = AGG_RAW_DIM // VGGT_DEFAULT_AGGREGATOR_SHAPE[-1]
AGG_TOKEN_TOKENS = VGGT_AGGREGATOR_TOKEN_COUNT
AGG_TOKEN_DIM = aggregator_token_dim(VGGT_DEFAULT_AGGREGATOR_SHAPE)
FULL_TOKEN_DIM = AGG_TOKEN_TOKENS * VGGT_FULL_TOKEN_EMBED_DIM


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
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor
        self._readout = make_vggt_readout(
            feature_kind=feature_kind,
            extractor=extractor,
            contract=self.contract,
        )

    def transform(
        self, obs_dict: dict
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        return self._readout.prepare(
            self._extractor,
            obs_dict["image"],
            is_first=obs_dict.get("is_first", False),
        )
