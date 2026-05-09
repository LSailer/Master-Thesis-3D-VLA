"""VGGTObsAdapter: wraps VGGTFeatureExtractor for RL acting."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

from modules.r2dreamer.adapters.obs_adapter import ObsAdapter
from modules.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


VGGT_FEATURE_DIM = 4116  # 37*37*3 + 9
VGGTFeatureKind = Literal["wp_cp", "aggregator"]


def _flatten_vggt(out: dict) -> jnp.ndarray:
    """Flatten VGGT outputs into a single feature vector (JAX)."""
    wp = out["world_points"].reshape(-1)  # (4107,)
    cp = out["camera_pose"]              # (9,)
    return jnp.concatenate([wp, cp]).astype(jnp.float32)


def _vggt_aggregator_features(out: dict, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Return final pre-head VGGT aggregator patch features."""
    features = out["aggregator_features"]
    if features.shape != expected_shape:
        raise ValueError(
            f"expected aggregator_features shape {expected_shape}, "
            f"got {features.shape}"
        )
    return features.astype(jnp.float32)


class VGGTObsAdapter(ObsAdapter):
    """Runs VGGT extraction, returns features for both buffer and agent."""

    def __init__(self, extractor: VGGTFeatureExtractor, feature_kind: VGGTFeatureKind = "wp_cp"):
        if feature_kind == "wp_cp":
            buffer_shape = (VGGT_FEATURE_DIM,)
            buffer_dtype = "float32"
        elif feature_kind == "aggregator":
            buffer_shape = tuple(extractor.aggregator_feature_shape)
            buffer_dtype = "float16"
        else:
            raise ValueError(f"unknown VGGT feature_kind {feature_kind!r}")
        super().__init__(
            buffer_dtype=buffer_dtype,
            buffer_shape=buffer_shape,
            normalize_on_sample=False,
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor
        self._feature_kind: VGGTFeatureKind = feature_kind

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        out = self._extractor.extract(obs_dict["image"])
        if self._feature_kind == "aggregator":
            features_jax = _vggt_aggregator_features(out, self.buffer_shape)
        else:
            features_jax = _flatten_vggt(out)

        # The replay buffer is CPU/NumPy storage. The acting path keeps JAX
        # float32 features so it can feed the JIT-compiled agent directly.
        replay_features = np.asarray(features_jax)
        if self._feature_kind == "aggregator":
            replay_features = replay_features.astype(np.float16)

        agent_features = features_jax.astype(jnp.float32)
        agent_obs = {"features": agent_features, "is_first": obs_dict.get("is_first", False)}
        return replay_features, agent_obs
