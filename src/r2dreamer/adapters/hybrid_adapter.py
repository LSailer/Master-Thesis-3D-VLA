"""HybridObsAdapter: wraps a VGGT extractor for the CNN+WP/CP hybrid encoder."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.observation_keys import (
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_IMAGE_SIZE as IMAGE_SIZE,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_RGB_DIM,
    build_hybrid_contract,
    wp_cp_dim,
)
from src.r2dreamer.observation_preparation.vggt_readouts import (
    flatten_world_points_camera_pose,
)
from src.shared.video_utils import resize_chw_uint8

HYBRID_FEATURE_DIM = HYBRID_RGB_DIM + wp_cp_dim()


class HybridObsAdapter(ObsAdapter):
    """Builds the hybrid replay fields ``{"image": rgb64, "wp_cp": wp_cp}``.

    The env renders 518x518 CHW uint8 (for VGGT). Each step we run VGGT once to
    obtain world_points + camera_pose, downsample the same frame to 64x64 for the
    CNN branch, and store both modalities under explicit replay keys. The agent
    still packs them into the legacy flat encoder input at the JAX boundary.
    """

    def __init__(
        self,
        extractor,
        *,
        env_render_resolution: int | None = None,
        encoder_module_cls=None,
        agent_overrides=None,
        design_notes: str = "",
    ):
        self.contract = build_hybrid_contract(
            extractor,
            env_render_resolution=env_render_resolution,
            encoder_module_cls=encoder_module_cls,
            agent_overrides=agent_overrides,
            design_notes=design_notes,
        )
        super().__init__(
            buffer_dtype=self.contract.replay_observation.buffer_dtype(),
            buffer_shape=self.contract.replay_observation.buffer_shape(),
            normalize_on_sample=self.contract.replay_observation.buffer_normalize(),
            agent_obs_shape=self.contract.encoder_input.shape,
            on_episode_reset=lambda scene_id="scene": extractor.reset_for_scene(scene_id),
        )
        self._extractor = extractor

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[dict[str, np.ndarray], dict]:
        out = self._extractor.extract(env_obs)  # image is 518 CHW uint8
        wp_cp = flatten_world_points_camera_pose(out)  # jnp (4116,)
        img64 = resize_chw_uint8(env_obs.image, IMAGE_SIZE)
        replay = {
            HYBRID_IMAGE_KEY: img64,
            HYBRID_WP_CP_KEY: np.asarray(wp_cp, dtype=np.float32),
        }
        agent_obs = {
            HYBRID_IMAGE_KEY: img64,
            HYBRID_WP_CP_KEY: jnp.asarray(replay[HYBRID_WP_CP_KEY]),
            "is_first": env_obs.is_first,
        }
        return replay, agent_obs
