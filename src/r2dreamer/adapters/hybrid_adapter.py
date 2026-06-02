"""HybridObsAdapter: wraps a VGGT extractor for CNN+VGGT hybrid encoders."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.vggt_adapter import (
    VGGT_FEATURE_DIM,
    VGGTFeatureKind,
    flatten_world_points_camera_pose,
    flatten_raw_aggregator,
    pool_aggregator_tokens,
)
from src.shared.video_utils import resize_chw_uint8


# Derived, not hand-typed: RGB branch (3*64*64, flattened) + the VGGT WP/CP vector.
# VGGT_FEATURE_DIM is the single source of truth for the 4116 term (see vggt_adapter),
# so a grid-size ablation that changes it stays consistent here automatically.
HYBRID_FEATURE_DIM = 3 * 64 * 64 + VGGT_FEATURE_DIM  # 12288 RGB + 4116 WP/CP = 16404


class HybridObsAdapter(ObsAdapter):
    """Builds a flat hybrid buffer vector ``[ rgb (12288) | vggt_features ]``.

    The env renders 518x518 CHW uint8 (for VGGT). Each step we run VGGT once to
    obtain the configured VGGT readout, downsample the same frame to 64x64 for
    the CNN branch, and concatenate the normalised RGB ([0,1]) with the readout.
    RGB is already divided by 255 here, so ``normalize_on_sample=False``. The
    default remains the original WP/CP hybrid.
    """

    def __init__(self, extractor, feature_kind: VGGTFeatureKind = "wp_cp"):
        if feature_kind == "wp_cp":
            k = int(getattr(extractor, "wp_pool_size", 37))
            vggt_dim = k * k * 3 + 9
            buffer_dtype = "float32"
        elif feature_kind == "aggregator":
            vggt_dim = 3 * int(extractor.aggregator_feature_shape[-1])
            buffer_dtype = "float32"
        elif feature_kind == "agg_raw":
            shp = extractor.aggregator_feature_shape
            n_tokens = 1 + (int(shp[0]) - 5)
            vggt_dim = n_tokens * int(shp[-1])
            buffer_dtype = "float16"
        else:
            raise ValueError(f"unknown hybrid feature_kind {feature_kind!r}")

        super().__init__(
            buffer_dtype=buffer_dtype,
            buffer_shape=(3 * 64 * 64 + vggt_dim,),
            normalize_on_sample=False,
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor
        self._feature_kind: VGGTFeatureKind = feature_kind
        self._aggregator_feature_shape = tuple(getattr(extractor, "aggregator_feature_shape", ()))

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        out = self._extractor.extract(obs_dict["image"])  # image is 518 CHW uint8
        if self._feature_kind == "aggregator":
            vggt_features = pool_aggregator_tokens(out, self._aggregator_feature_shape)
        elif self._feature_kind == "agg_raw":
            vggt_features = flatten_raw_aggregator(out, self._aggregator_feature_shape)
        else:
            vggt_features = flatten_world_points_camera_pose(out)
        img64 = resize_chw_uint8(obs_dict["image"], 64)  # (3,64,64) uint8
        rgb = (img64.astype(np.float32) / 255.0).reshape(-1)  # (12288,) [0,1]
        replay_vggt = np.asarray(vggt_features, dtype=np.float32)
        replay = np.concatenate([rgb, replay_vggt])
        agent_hybrid = jnp.concatenate(
            [jnp.asarray(rgb, dtype=jnp.float32), vggt_features.astype(jnp.float32)]
        )
        if self._feature_kind == "agg_raw":
            replay = replay.astype(np.float16)
        else:
            replay = replay.astype(np.float32)
        agent_obs = {
            "hybrid": agent_hybrid,
            "image": img64,
            "is_first": obs_dict.get("is_first", False),
        }
        return replay, agent_obs
