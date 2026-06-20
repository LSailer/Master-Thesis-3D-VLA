"""Internal VGGT readout implementations.

This module groups the VGGT-specific extraction/readout rules behind one small
``prepare(extractor, image, is_first=...)`` interface. The public adapter only
wires a readout and delegates to it; shape validation, token pooling, flattening,
and replay dtype conversion stay local here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from src.r2dreamer.obs_batch import CAMERA_POSE_KEY, WORLD_POINTS_KEY
from src.r2dreamer.observation_preparation.vggt import (
    VGGT_AGGREGATOR_PATCH_START_IDX,
    VGGTFeatureKind,
    contract_world_points_hwc_shape,
    head_readout_spec,
)


def flatten_world_points_camera_pose(out: dict) -> jnp.ndarray:
    """Flatten structured VGGT world-points + camera-pose for MLP encoders."""
    wp = out[WORLD_POINTS_KEY].reshape(-1)
    cp = out[CAMERA_POSE_KEY].reshape(-1)
    return jnp.concatenate([wp, cp]).astype(jnp.float32)


def _structured_world_points_camera_pose(
    out: dict,
    *,
    expected_hwc_shape: tuple[int, int, int],
    world_points_key: str = "world_points",
) -> dict[str, jnp.ndarray]:
    """Return VGGT world points and camera pose as explicit fields."""
    world_points = out[world_points_key]
    if tuple(world_points.shape) != expected_hwc_shape:
        raise ValueError(
            f"expected {world_points_key} shape {expected_hwc_shape}, "
            f"got {tuple(world_points.shape)}"
        )
    return {
        WORLD_POINTS_KEY: jnp.transpose(world_points, (2, 0, 1)).astype(jnp.float32),
        CAMERA_POSE_KEY: out["camera_pose"].astype(jnp.float32),
    }


def _checked_feature(
    out: dict, key: str, expected_shape: tuple[int, ...]
) -> jnp.ndarray:
    """Return a float32 VGGT readout after validating its contract shape."""
    features = out[key]
    if tuple(features.shape) != expected_shape:
        raise ValueError(f"expected {key} shape {expected_shape}, got {features.shape}")
    return features.astype(jnp.float32)


def _pool_aggregator_tokens(features: jnp.ndarray) -> jnp.ndarray:
    """Pool camera + patch summaries from pre-head aggregator tokens.

    Layout matches the aggregator's ``[camera, register, patches]`` ordering
    (see ``src/vggt/jax/aggregator.py``): keep the camera token unmixed, drop
    register tokens (1:5), and reduce patches with mean + max.
    """
    cam = features[0]
    patches = features[VGGT_AGGREGATOR_PATCH_START_IDX:]
    mean_p = patches.mean(axis=0)
    max_p = patches.max(axis=0)
    return jnp.concatenate([cam, mean_p, max_p], axis=0)


def _flatten_raw_aggregator(features: jnp.ndarray) -> jnp.ndarray:
    """Flatten camera + patch tokens, dropping the 4 register tokens."""
    cam = features[0:1]
    patches = features[VGGT_AGGREGATOR_PATCH_START_IDX:]
    kept = jnp.concatenate([cam, patches], axis=0)
    return kept.reshape(-1)


def _flatten_full_aggregator_tokens(features: jnp.ndarray) -> jnp.ndarray:
    """Flatten camera, register, and patch tokens for token-Transformer replay."""
    return features.reshape(-1)


def full_aggregator_tokens(out: dict, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Return full-width VGGT aggregator tokens for live context paths."""
    return _checked_feature(out, "aggregator_full_tokens", expected_shape)


AggregatorProjection = Callable[[jnp.ndarray], jnp.ndarray]


AGGREGATOR_PROJECTIONS: dict[str, AggregatorProjection] = {
    "aggregator": _pool_aggregator_tokens,
    "agg_raw": _flatten_raw_aggregator,
    "agg_tokens": _flatten_full_aggregator_tokens,
}


@dataclass(frozen=True)
class VGGTHeadReadout:
    """Readout family for VGGT heads: world-points plus camera-pose."""

    expected_hwc_shape: tuple[int, int, int]
    replay_dtype: dict[str, str]
    world_points_key: str = "world_points"
    return_dense: bool = False

    def prepare(
        self,
        extractor: Any,
        image: np.ndarray,
        *,
        is_first: bool,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Extract and format head outputs for replay and acting."""
        out = (
            extractor.extract(image, return_dense=True)
            if self.return_dense
            else extractor.extract(image)
        )
        features = _structured_world_points_camera_pose(
            out,
            expected_hwc_shape=self.expected_hwc_shape,
            world_points_key=self.world_points_key,
        )
        replay = {
            key: np.asarray(features[key], dtype=np.dtype(self.replay_dtype[key]))
            for key in (WORLD_POINTS_KEY, CAMERA_POSE_KEY)
        }
        agent_obs = {
            WORLD_POINTS_KEY: features[WORLD_POINTS_KEY].astype(jnp.float16),
            CAMERA_POSE_KEY: features[CAMERA_POSE_KEY].astype(jnp.float16),
            "is_first": is_first,
        }
        return replay, agent_obs


@dataclass(frozen=True)
class VGGTAggregatorReadout:
    """Readout family for VGGT pre-head aggregator tokens."""

    expected_shape: tuple[int, ...]
    replay_dtype: str
    project: AggregatorProjection

    def prepare(
        self,
        extractor: Any,
        image: np.ndarray,
        *,
        is_first: bool,
    ) -> tuple[np.ndarray, dict]:
        """Extract and format aggregator outputs for replay and acting."""
        out = extractor.extract(image)
        features = _checked_feature(out, "aggregator_features", self.expected_shape)
        readout = self.project(features).astype(jnp.float32)
        replay = np.asarray(readout, dtype=np.dtype(self.replay_dtype))
        agent_obs = {
            "features": readout.astype(jnp.float32),
            "is_first": is_first,
        }
        return replay, agent_obs


def make_vggt_readout(
    *,
    feature_kind: VGGTFeatureKind,
    extractor: Any,
    contract,
) -> VGGTHeadReadout | VGGTAggregatorReadout:
    """Build the internal readout adapter for one VGGT feature kind."""
    replay_dtype = contract.replay_observation.buffer_dtype()
    if spec := head_readout_spec(feature_kind):
        if not isinstance(replay_dtype, dict):
            raise ValueError("VGGT head readouts require per-field replay dtypes")
        return VGGTHeadReadout(
            expected_hwc_shape=contract_world_points_hwc_shape(contract),
            replay_dtype=replay_dtype,
            world_points_key=(
                "dense_world_points" if spec.wp_side == "dense" else "world_points"
            ),
            return_dense=spec.wp_side == "dense",
        )
    if feature_kind in AGGREGATOR_PROJECTIONS:
        if isinstance(replay_dtype, dict):
            raise ValueError("VGGT aggregator readouts require a scalar replay dtype")
        return VGGTAggregatorReadout(
            expected_shape=tuple(getattr(extractor, "aggregator_feature_shape", ())),
            replay_dtype=replay_dtype,
            project=AGGREGATOR_PROJECTIONS[feature_kind],
        )
    raise ValueError(f"unknown VGGT feature_kind {feature_kind!r}")
