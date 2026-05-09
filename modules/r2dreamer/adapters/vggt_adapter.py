"""VGGTObsAdapter: wraps VGGTFeatureExtractor for RL acting."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from modules.r2dreamer.adapters.obs_adapter import ObsAdapter
from modules.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


VGGT_FEATURE_DIM = 4116  # 37*37*3 + 9
VGGT_AGGREGATOR_FEATURE_SHAPE = (37, 37, 1024)


def _flatten_vggt(out: dict) -> jnp.ndarray:
    """Flatten VGGT outputs into a single feature vector (JAX)."""
    wp = out["world_points"].reshape(-1)  # (4107,)
    cp = out["camera_pose"]              # (9,)
    return jnp.concatenate([wp, cp]).astype(jnp.float32)


def _vggt_aggregator_features(out: dict) -> jnp.ndarray:
    """Return final pre-head VGGT aggregator patch features (37, 37, 1024)."""
    features = out["aggregator_features"]
    if features.shape != VGGT_AGGREGATOR_FEATURE_SHAPE:
        raise ValueError(
            f"expected aggregator_features shape {VGGT_AGGREGATOR_FEATURE_SHAPE}, "
            f"got {features.shape}"
        )
    return features.astype(jnp.float32)


class VGGTObsAdapter(ObsAdapter):
    """Runs VGGT extraction, returns features for both buffer and agent."""

    def __init__(self, extractor: VGGTFeatureExtractor, feature_kind: str = "wp_cp"):
        if feature_kind == "wp_cp":
            buffer_shape = (VGGT_FEATURE_DIM,)
        elif feature_kind == "aggregator":
            buffer_shape = VGGT_AGGREGATOR_FEATURE_SHAPE
        else:
            raise ValueError(f"unknown VGGT feature_kind {feature_kind!r}")
        buffer_dtype = "float16" if feature_kind == "aggregator" else "float32"
        super().__init__(
            buffer_dtype=buffer_dtype,
            buffer_shape=buffer_shape,
            normalize_on_sample=False,
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor
        self._feature_kind = feature_kind

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        out = self._extractor.extract(obs_dict["image"])
        if self._feature_kind == "aggregator":
            features_jax = _vggt_aggregator_features(out)
        else:
            features_jax = _flatten_vggt(out)
        # Replay buffer stores numpy; agent consumes JAX.
        features_np = np.asarray(features_jax)
        if self._feature_kind == "aggregator":
            features_np = features_np.astype(np.float16)
        agent_obs = {"features": features_jax, "is_first": obs_dict.get("is_first", False)}
        return features_np, agent_obs
