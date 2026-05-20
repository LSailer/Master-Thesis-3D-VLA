"""VGGTObsAdapter: wraps VGGTFeatureExtractor for RL acting."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


VGGT_FEATURE_DIM = 4116  # 37*37*3 + 9
VGGTFeatureKind = Literal["wp_cp", "aggregator"]


def flatten_world_points_camera_pose(out: dict) -> jnp.ndarray:
    """Flatten VGGT outputs into a single feature vector (JAX)."""
    wp = out["world_points"].reshape(-1)  # (4107,)
    cp = out["camera_pose"]              # (9,)
    return jnp.concatenate([wp, cp]).astype(jnp.float32)


_PATCH_START_IDX = 5  # 1 camera token + 4 register tokens, then patches


def pool_aggregator_tokens(out: dict, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Return three pre-head pools concatenated as a single (3*D,) vector.

    Layout matches the aggregator's ``[camera, register, patches]`` ordering
    (see ``src/vggt/jax/aggregator.py``): the camera token at index 0 is
    the embedding VGGT's own ``camera_head`` reads to predict pose, so we keep
    it unmixed. Register tokens (1:5) are attention sinks and dropped. Patches
    (5:) are reduced with both mean (smooth global) and max (salient features)
    so the encoder sees signals at different scales.
    """
    features = out["aggregator_features"]
    if features.shape != expected_shape:
        raise ValueError(
            f"expected aggregator_features shape {expected_shape}, "
            f"got {features.shape}"
        )
    features = features.astype(jnp.float32)
    cam = features[0]
    patches = features[_PATCH_START_IDX:]
    mean_p = patches.mean(axis=0)
    max_p = patches.max(axis=0)
    return jnp.concatenate([cam, mean_p, max_p], axis=0)


class VGGTObsAdapter(ObsAdapter):
    """Runs VGGT extraction, returns features for both buffer and agent."""

    def __init__(self, extractor: VGGTFeatureExtractor, feature_kind: VGGTFeatureKind = "wp_cp"):
        if feature_kind == "wp_cp":
            buffer_shape = (VGGT_FEATURE_DIM,)
            buffer_dtype = "float32"
        elif feature_kind == "aggregator":
            embed_dim = int(extractor.aggregator_feature_shape[-1])
            buffer_shape = (3 * embed_dim,)  # [cam | mean_patches | max_patches]
            buffer_dtype = "float32"
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
        self._aggregator_feature_shape = tuple(getattr(extractor, "aggregator_feature_shape", ()))

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        out = self._extractor.extract(obs_dict["image"])
        if self._feature_kind == "aggregator":
            features_jax = pool_aggregator_tokens(out, self._aggregator_feature_shape)
        else:
            features_jax = flatten_world_points_camera_pose(out)

        # The replay buffer is CPU/NumPy storage. The acting path keeps JAX
        # float32 features so it can feed the JIT-compiled agent directly.
        replay_features = np.asarray(features_jax)
        if self._feature_kind == "aggregator":
            replay_features = replay_features.astype(np.float32)

        agent_features = features_jax.astype(jnp.float32)
        agent_obs = {"features": agent_features, "is_first": obs_dict.get("is_first", False)}
        return replay_features, agent_obs
