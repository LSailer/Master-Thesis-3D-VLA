"""HybridObsAdapter: wraps a VGGT extractor for the CNN+WP/CP hybrid encoder."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.vggt_adapter import (
    VGGT_FEATURE_DIM,
    flatten_world_points_camera_pose,
)
from src.shared.video_utils import resize_chw_uint8


# Derived, not hand-typed: RGB branch (3*64*64, flattened) + the VGGT WP/CP vector.
# VGGT_FEATURE_DIM is the single source of truth for the 4116 term (see vggt_adapter),
# so a grid-size ablation that changes it stays consistent here automatically.
HYBRID_FEATURE_DIM = 3 * 64 * 64 + VGGT_FEATURE_DIM  # 12288 RGB + 4116 WP/CP = 16404


class HybridObsAdapter(ObsAdapter):
    """Builds the flat hybrid buffer vector ``[ rgb (12288) | wp_cp (4116) ]``.

    The env renders 518x518 CHW uint8 (for VGGT). Each step we run VGGT once to
    obtain world_points + camera_pose, downsample the same frame to 64x64 for the
    CNN branch, and concatenate the normalised RGB ([0,1]) with the wp/cp vector.
    RGB is already divided by 255 here, so ``normalize_on_sample=False``.
    """

    def __init__(self, extractor):
        super().__init__(
            buffer_dtype="float32",
            buffer_shape=(HYBRID_FEATURE_DIM,),
            normalize_on_sample=False,
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        out = self._extractor.extract(obs_dict["image"])  # image is 518 CHW uint8
        wp_cp = flatten_world_points_camera_pose(out)  # jnp (4116,)
        img64 = resize_chw_uint8(obs_dict["image"], 64)  # (3,64,64) uint8
        rgb = (img64.astype(np.float32) / 255.0).reshape(-1)  # (12288,) [0,1]
        replay = np.concatenate(
            [rgb, np.asarray(wp_cp, dtype=np.float32)]
        ).astype(np.float32)  # (16404,)
        agent_obs = {
            "hybrid": jnp.asarray(replay),
            "image": img64,
            "is_first": obs_dict.get("is_first", False),
        }
        return replay, agent_obs
