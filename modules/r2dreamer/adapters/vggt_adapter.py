"""VGGTObsAdapter: wraps VGGTFeatureExtractor for RL acting."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from modules.r2dreamer.adapters.obs_adapter import ObsAdapter
from modules.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


VGGT_FEATURE_DIM = 4116  # 37*37*3 + 9


def _flatten_vggt(out: dict) -> jnp.ndarray:
    """Flatten VGGT outputs into a single feature vector (JAX)."""
    wp = out["world_points"].reshape(-1)  # (4107,)
    cp = out["camera_pose"]              # (9,)
    return jnp.concatenate([wp, cp]).astype(jnp.float32)


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
        features_jax = _flatten_vggt(self._extractor.extract(obs_dict["image"]))
        # Replay buffer stores numpy; agent consumes JAX.
        features_np = np.asarray(features_jax)
        agent_obs = {"features": features_jax, "is_first": obs_dict.get("is_first", False)}
        return features_np, agent_obs
