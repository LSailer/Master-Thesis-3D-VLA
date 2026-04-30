"""VGGTObsAdapter: wraps VGGTFeatureExtractor for RL acting."""

from __future__ import annotations

import numpy as np

from modules.r2dreamer.adapters.obs_adapter import ObsAdapter
from modules.vggt.feature_extractor import VGGTFeatureExtractor


VGGT_FEATURE_DIM = 4116  # 37*37*3 + 9


def _flatten_vggt(out: dict) -> np.ndarray:
    """Flatten VGGT outputs into a single feature vector."""
    wp = out["world_points"].reshape(-1)  # (4107,)
    cp = out["camera_pose"]               # (9,)
    return np.concatenate([wp, cp]).astype(np.float32)


class VGGTObsAdapter(ObsAdapter):
    """Runs VGGT extraction, returns features for both buffer and agent."""

    def __init__(self, extractor: VGGTFeatureExtractor):
        super().__init__(
            buffer_dtype="float32",
            buffer_shape=(VGGT_FEATURE_DIM,),
            normalize_on_sample=False,
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        features = _flatten_vggt(self._extractor.extract(obs_dict["image"]))
        agent_obs = {"features": features, "is_first": obs_dict.get("is_first", False)}
        return features, agent_obs
