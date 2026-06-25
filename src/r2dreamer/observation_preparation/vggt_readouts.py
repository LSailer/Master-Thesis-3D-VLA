"""Internal VGGT readout implementations.

This module groups the VGGT-specific extraction/readout rules behind one small
``prepare(extractor, obs)`` interface. The public adapter only wires a readout
and delegates to it; shape validation, token pooling, flattening,
and replay dtype conversion stay local here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.obs_batch import CAMERA_POSE_KEY, WORLD_POINTS_KEY
from src.r2dreamer.observation_preparation.vggt import (
    VGGT_AGGREGATOR_PATCH_START_IDX,
    VGGT_DEFAULT_WP_POOL_SIZE,
    VGGT_IMAGE_SIZE,
    VGGTFeatureKind,
    contract_world_points_hwc_shape,
    head_readout_spec,
)


class VGGTOutputLike(Protocol):
    """Raw structured VGGT output fields consumed by readout adapters."""

    world_points: jnp.ndarray | None
    camera_pose: jnp.ndarray | None
    frame_tokens: jnp.ndarray
    global_tokens: jnp.ndarray


def _require_output_field(
    out: VGGTOutputLike,
    field_name: str,
    value: jnp.ndarray | None,
) -> jnp.ndarray:
    """Return a required raw extractor field or fail with a contract error."""
    if value is None:
        raise ValueError(f"VGGT output field {field_name!r} is required")
    return value


def flatten_world_points_camera_pose(out: VGGTOutputLike) -> jnp.ndarray:
    """Flatten default 37x37 world-points + camera-pose for MLP encoders."""
    features = _structured_world_points_camera_pose(
        out,
        expected_hwc_shape=(VGGT_DEFAULT_WP_POOL_SIZE, VGGT_DEFAULT_WP_POOL_SIZE, 3),
        include_camera_pose=True,
    )
    wp = features[WORLD_POINTS_KEY].reshape(-1)
    cp = features[CAMERA_POSE_KEY].reshape(-1)
    return jnp.concatenate([wp, cp]).astype(jnp.float32)


def _pool_world_points_hwc(
    world_points: jnp.ndarray,
    expected_hwc_shape: tuple[int, int, int],
) -> jnp.ndarray:
    """Project raw full-resolution VGGT points to the readout HWC contract."""
    if tuple(world_points.shape) == expected_hwc_shape:
        return world_points
    if tuple(world_points.shape) != (VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE, 3):
        raise ValueError(
            f"expected raw world_points shape {(VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE, 3)} "
            f"or readout shape {expected_hwc_shape}, got {tuple(world_points.shape)}"
        )
    height, width, channels = expected_hwc_shape
    if height != width or channels != 3:
        raise ValueError(f"expected HWC point-map shape, got {expected_hwc_shape}")
    batched = world_points[None]
    if VGGT_IMAGE_SIZE % height == 0:
        factor = VGGT_IMAGE_SIZE // height
        return batched.reshape(1, height, factor, width, factor, channels).mean(
            axis=(2, 4)
        )[0]
    return jax.image.resize(
        batched,
        (1, height, width, channels),
        method="linear",
        antialias=True,
    )[0]


def _structured_world_points_camera_pose(
    out: VGGTOutputLike,
    *,
    expected_hwc_shape: tuple[int, int, int],
    include_camera_pose: bool,
) -> dict[str, jnp.ndarray]:
    """Transform raw extractor points into legacy structured readout fields."""
    raw_world_points = _require_output_field(
        out, "world_points", out.world_points
    )
    world_points = _pool_world_points_hwc(raw_world_points, expected_hwc_shape)
    features = {
        WORLD_POINTS_KEY: jnp.transpose(world_points, (2, 0, 1)).astype(jnp.float32),
    }
    if include_camera_pose:
        camera_pose = _require_output_field(out, "camera_pose", out.camera_pose)
        features[CAMERA_POSE_KEY] = camera_pose.astype(jnp.float32)
    return features


def _checked_feature(
    out: VGGTOutputLike, key: str, expected_shape: tuple[int, ...]
) -> jnp.ndarray:
    """Return a float32 VGGT readout after validating its contract shape."""
    features = getattr(out, key)
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


def full_aggregator_tokens(
    out: VGGTOutputLike, expected_shape: tuple[int, ...]
) -> jnp.ndarray:
    """Return full-width VGGT aggregator tokens for live context paths."""
    frame_tokens = out.frame_tokens
    global_tokens = out.global_tokens
    features = jnp.concatenate([frame_tokens, global_tokens], axis=-1)
    if tuple(features.shape) != expected_shape:
        raise ValueError(
            f"expected full tokens shape {expected_shape}, got {features.shape}"
        )
    return features.astype(jnp.float32)


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
    include_camera_pose: bool

    def prepare(
        self,
        extractor: Any,
        obs: ObservationFrame,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Extract and format head outputs for replay and acting."""
        out = extractor.extract(obs)
        features = _structured_world_points_camera_pose(
            out,
            expected_hwc_shape=self.expected_hwc_shape,
            include_camera_pose=self.include_camera_pose,
        )
        replay_keys = tuple(features.keys())
        replay = {
            key: np.asarray(features[key], dtype=np.dtype(self.replay_dtype[key]))
            for key in replay_keys
        }
        agent_obs = {
            key: features[key].astype(jnp.float16)
            for key in replay_keys
        }
        agent_obs["is_first"] = obs.is_first
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
        obs: ObservationFrame,
    ) -> tuple[np.ndarray, dict]:
        """Extract and format aggregator outputs for replay and acting."""
        out = extractor.extract(obs)
        features = _checked_feature(out, "global_tokens", self.expected_shape)
        readout = self.project(features).astype(jnp.float32)
        replay = np.asarray(readout, dtype=np.dtype(self.replay_dtype))
        agent_obs = {
            "features": readout.astype(jnp.float32),
            "is_first": obs.is_first,
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
            include_camera_pose=spec.include_camera_pose,
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
