"""Helpers for modality-aware replay observations.

Replay storage may keep modalities under explicit keys, while the current
Flax encoders still consume a single tensor. This module is the narrow bridge
between those two contracts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

import jax.numpy as jnp
from jax.typing import ArrayLike

from src.r2dreamer.encoder_types import FLAT_VGGT_ENCODER_TYPES


HYBRID_IMAGE_KEY = "image"
HYBRID_WP_CP_KEY = "wp_cp"
HOUSE_CONTEXT_KEY = "house_context"
FULL_TOKENS_KEY = "full_tokens"
GLOBAL_TOKENS_KEY = "global_tokens"
WORLD_POINTS_KEY = "world_points"
CAMERA_POSE_KEY = "camera_pose"

_RGB_CONTEXT_KEYS = {
    "hybrid": HYBRID_WP_CP_KEY,
    "vggt_house_context": HOUSE_CONTEXT_KEY,
}

ObsShape: TypeAlias = tuple[int, ...] | Mapping[str, tuple[int, ...]]
EncoderObs: TypeAlias = jnp.ndarray | dict[str, jnp.ndarray]
ObsMapping: TypeAlias = Mapping[str, ArrayLike]
ObsInput: TypeAlias = ArrayLike | ObsMapping
DTypeLike: TypeAlias = Any


@dataclass(frozen=True)
class ReplayBatchShape:
    """Replay observation prefix dimensions."""

    batch_size: int
    seq_len: int

    @property
    def steps(self) -> int:
        """Flattened replay-step count consumed by encoder modules."""
        return self.batch_size * self.seq_len


class ObsBatchConfig(Protocol):
    """Config fields needed by observation batch packing helpers.

    Keep this protocol narrower than ``R2DreamerConfig`` so tests and launch-time
    shims can pass config-like objects without importing the concrete dataclass.
    """

    @property
    def encoder_type(self) -> str:
        """Observation preparation / Encoder Module type key."""

    @property
    def obs_shape(self) -> ObsShape:
        """Shape consumed by the Encoder Module."""

    @property
    def compute_dtype(self) -> str:
        """JAX compute dtype name for prepared feature tensors."""

    @property
    def vggt_feature_dim(self) -> int:
        """Flat VGGT feature dimension used by hybrid decoder targets."""

    @property
    def vggt_token_count(self) -> int:
        """Number of VGGT tokens expected by token encoders."""

    @property
    def vggt_token_dim(self) -> int:
        """Width of each VGGT token expected by token encoders."""


def compute_jnp_dtype(dtype: str):
    """Return the JAX dtype named by ``R2DreamerConfig.compute_dtype``."""
    if dtype == "float32":
        return jnp.float32
    if dtype in ("bfloat16", "bf16"):
        return jnp.bfloat16
    if dtype in ("float16", "fp16"):
        return jnp.float16
    raise ValueError(f"Unsupported compute_dtype={dtype!r}")


def obs_leading_shape(obs: ObsInput) -> ReplayBatchShape:
    """Return the ``(B, T)`` prefix of a replay observation batch."""
    if isinstance(obs, Mapping):
        if HYBRID_IMAGE_KEY in obs:
            first = obs[HYBRID_IMAGE_KEY]
        elif CAMERA_POSE_KEY in obs:
            first = obs[CAMERA_POSE_KEY]
        else:
            first = next(iter(obs.values()))
        prefix = jnp.shape(first)
        return ReplayBatchShape(batch_size=prefix[0], seq_len=prefix[1])
    prefix = jnp.shape(obs)
    return ReplayBatchShape(batch_size=prefix[0], seq_len=prefix[1])


def normalize_image_obs(image: ArrayLike) -> jnp.ndarray:
    """Return CHW image observations as float32 in ``[0, 1]``.

    Direct unit tests often pass already-normalized float arrays, while replay
    can now pass compact uint8 images. Branch on dtype so both inputs work.
    """
    image = jnp.asarray(image)
    if image.dtype == jnp.uint8:
        return image.astype(jnp.float32) / 255.0
    return image.astype(jnp.float32)


def _features(obs: ObsMapping, context_key: str) -> ArrayLike:
    if context_key in obs:
        return obs[context_key]
    if context_key == HYBRID_WP_CP_KEY and "features" in obs:
        return obs["features"]
    raise KeyError(f"obs must contain {context_key!r}")


def pack_rgb_context_obs(obs: ObsInput, *, context_key: str) -> jnp.ndarray:
    """Pack dict observations into the legacy flat ``[rgb | context]`` tensor."""
    if not isinstance(obs, Mapping):
        return jnp.asarray(obs, dtype=jnp.float32)
    image = normalize_image_obs(obs[HYBRID_IMAGE_KEY])
    features = jnp.asarray(_features(obs, context_key), dtype=jnp.float32)
    prefix = image.shape[:-3]
    image_flat = image.reshape(*prefix, -1)
    features_flat = features.reshape(*features.shape[:-1], -1)
    return jnp.concatenate([image_flat, features_flat], axis=-1)


def pack_hybrid_obs(obs: ObsInput) -> jnp.ndarray:
    """Pack hybrid dict observations into the legacy flat encoder tensor."""
    return pack_rgb_context_obs(obs, context_key=HYBRID_WP_CP_KEY)


def pack_world_points_camera_pose_obs(
    obs: ObsMapping,
    *,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Flatten structured VGGT WP/CP replay into legacy MLP features."""
    world_points = jnp.asarray(obs[WORLD_POINTS_KEY], dtype=dtype)
    camera_pose = jnp.asarray(obs[CAMERA_POSE_KEY], dtype=dtype)
    prefix = world_points.shape[:-3]
    world_points_flat = world_points.reshape(*prefix, -1)
    camera_pose_flat = camera_pose.reshape(*camera_pose.shape[:-1], -1)
    return jnp.concatenate([world_points_flat, camera_pose_flat], axis=-1)


def _flat_obs_shape(cfg: ObsBatchConfig) -> tuple[int, ...]:
    if isinstance(cfg.obs_shape, Mapping):
        raise TypeError(f"{cfg.encoder_type} expects structured obs_shape")
    return cfg.obs_shape


def _mapping_obs(obs: ObsInput, encoder_type: str) -> ObsMapping:
    """Return a structured observation or fail with an encoder-specific error."""
    if not isinstance(obs, Mapping):
        raise TypeError(f"{encoder_type} expects dict obs")
    return cast(ObsMapping, obs)


def _rgb_token_image(
    obs: ObsMapping, replay_shape: ReplayBatchShape
) -> jnp.ndarray:
    """Return RGB replay images flattened from ``(B, T)`` to ``B*T``."""
    return normalize_image_obs(obs[HYBRID_IMAGE_KEY]).reshape(
        replay_shape.steps, 3, 64, 64
    )


def _structured_shape(cfg: ObsBatchConfig, key: str) -> tuple[int, ...]:
    if not isinstance(cfg.obs_shape, Mapping):
        raise TypeError(f"{cfg.encoder_type} expects structured obs_shape")
    return tuple(cfg.obs_shape[key])


def _flatten_replay_field(
    obs: ObsMapping,
    key: str,
    replay_shape: ReplayBatchShape,
    dtype: DTypeLike,
    tail_shape: tuple[int, ...] | None = None,
) -> jnp.ndarray:
    value = jnp.asarray(obs[key], dtype=dtype)
    if tail_shape is None:
        return value.reshape(replay_shape.steps, -1)
    return value.reshape(replay_shape.steps, *tail_shape)


def _singleton_sidecar(value: ArrayLike, dtype: DTypeLike) -> jnp.ndarray:
    sidecar = jnp.asarray(value, dtype=dtype)
    if sidecar.ndim == 2:
        sidecar = sidecar[None]
    return sidecar


def _step_field(obs: ObsMapping, key: str, dtype: DTypeLike) -> jnp.ndarray:
    return jnp.asarray(obs[key], dtype=dtype)[None]


def _full_tokens_batch(
    obs: ObsMapping, cfg: ObsBatchConfig, replay_shape: ReplayBatchShape, compute_dtype
) -> EncoderObs:
    """Pack full-token replay fields for the RGB+token Encoder Module."""
    image = _rgb_token_image(obs, replay_shape)
    tokens = jnp.asarray(obs[FULL_TOKENS_KEY], dtype=compute_dtype).reshape(
        replay_shape.steps, cfg.vggt_token_count, cfg.vggt_token_dim
    )
    return {HYBRID_IMAGE_KEY: image, FULL_TOKENS_KEY: tokens}


def _global_tokens_batch(
    obs: ObsMapping, cfg: ObsBatchConfig, replay_shape: ReplayBatchShape, compute_dtype
) -> EncoderObs:
    """Pack singleton global-token replay fields for the RGB+token encoder."""
    image = _rgb_token_image(obs, replay_shape)
    tokens = jnp.asarray(obs[GLOBAL_TOKENS_KEY], dtype=compute_dtype)
    expected_tokens = (1, cfg.vggt_token_count, cfg.vggt_token_dim)
    if tokens.shape != expected_tokens:
        raise ValueError(
            "vggt_house_global_tokens_nogate expects singleton "
            f"global tokens with shape {expected_tokens}, got {tokens.shape}"
        )
    return {HYBRID_IMAGE_KEY: image, GLOBAL_TOKENS_KEY: tokens}


def _wp64_cnn_cp_mlp_batch(
    obs: ObsMapping, cfg: ObsBatchConfig, replay_shape: ReplayBatchShape, compute_dtype
) -> EncoderObs:
    """Pack structured WP64+CP replay fields for the hybrid Encoder Module."""
    world_points = _flatten_replay_field(
        obs,
        WORLD_POINTS_KEY,
        replay_shape,
        compute_dtype,
        _structured_shape(cfg, WORLD_POINTS_KEY),
    )
    camera_pose = _flatten_replay_field(
        obs,
        CAMERA_POSE_KEY,
        replay_shape,
        compute_dtype,
        _structured_shape(cfg, CAMERA_POSE_KEY),
    )
    return {WORLD_POINTS_KEY: world_points, CAMERA_POSE_KEY: camera_pose}


def _house_points_pose_batch(
    obs: ObsMapping, _cfg: ObsBatchConfig, replay_shape: ReplayBatchShape, compute_dtype
) -> EncoderObs:
    """Pack current camera poses plus one static house point cloud."""
    camera_pose = _flatten_replay_field(
        obs,
        CAMERA_POSE_KEY,
        replay_shape,
        compute_dtype,
    )
    house_context = _singleton_sidecar(obs[HOUSE_CONTEXT_KEY], compute_dtype)
    return {CAMERA_POSE_KEY: camera_pose, HOUSE_CONTEXT_KEY: house_context}


def _full_tokens_step(
    obs: ObsMapping, _cfg: ObsBatchConfig, compute_dtype
) -> EncoderObs:
    return {
        HYBRID_IMAGE_KEY: normalize_image_obs(obs[HYBRID_IMAGE_KEY])[None],
        FULL_TOKENS_KEY: _step_field(obs, FULL_TOKENS_KEY, compute_dtype),
    }


def _global_tokens_step(
    obs: ObsMapping, _cfg: ObsBatchConfig, compute_dtype
) -> EncoderObs:
    return {
        HYBRID_IMAGE_KEY: normalize_image_obs(obs[HYBRID_IMAGE_KEY])[None],
        GLOBAL_TOKENS_KEY: _singleton_sidecar(obs[GLOBAL_TOKENS_KEY], compute_dtype),
    }


def _wp64_cnn_cp_mlp_step(
    obs: ObsMapping, _cfg: ObsBatchConfig, compute_dtype
) -> EncoderObs:
    return {
        WORLD_POINTS_KEY: _step_field(obs, WORLD_POINTS_KEY, compute_dtype),
        CAMERA_POSE_KEY: _step_field(obs, CAMERA_POSE_KEY, compute_dtype),
    }


def _house_points_pose_step(
    obs: ObsMapping, _cfg: ObsBatchConfig, compute_dtype
) -> EncoderObs:
    return {
        CAMERA_POSE_KEY: _step_field(obs, CAMERA_POSE_KEY, compute_dtype),
        HOUSE_CONTEXT_KEY: _singleton_sidecar(obs[HOUSE_CONTEXT_KEY], compute_dtype),
    }


BatchPacker: TypeAlias = Callable[
    [ObsMapping, ObsBatchConfig, ReplayBatchShape, DTypeLike], EncoderObs
]
StepPacker: TypeAlias = Callable[[ObsMapping, ObsBatchConfig, DTypeLike], EncoderObs]


_STRUCTURED_BATCH_PACKERS: Mapping[str, BatchPacker] = {
    "vggt_house_full_tokens_nogate": _full_tokens_batch,
    "vggt_house_global_tokens_nogate": _global_tokens_batch,
    "vggt_wp64_cnn_cp_mlp": _wp64_cnn_cp_mlp_batch,
    "vggt_house_points_pose": _house_points_pose_batch,
}


_STRUCTURED_STEP_PACKERS: Mapping[str, StepPacker] = {
    "vggt_house_full_tokens_nogate": _full_tokens_step,
    "vggt_house_global_tokens_nogate": _global_tokens_step,
    "vggt_wp64_cnn_cp_mlp": _wp64_cnn_cp_mlp_step,
    "vggt_house_points_pose": _house_points_pose_step,
}


def _structured_encoder_batch(
    obs: ObsInput, cfg: ObsBatchConfig, replay_shape: ReplayBatchShape, compute_dtype
) -> EncoderObs | None:
    packer = _STRUCTURED_BATCH_PACKERS.get(cfg.encoder_type)
    if packer is None:
        return None
    return packer(_mapping_obs(obs, cfg.encoder_type), cfg, replay_shape, compute_dtype)


def _structured_encoder_step(
    obs: ObsMapping, cfg: ObsBatchConfig, compute_dtype
) -> EncoderObs | None:
    packer = _STRUCTURED_STEP_PACKERS.get(cfg.encoder_type)
    if packer is None:
        return None
    return packer(obs, cfg, compute_dtype)


class ObservationPacker:
    """Pack prepared observations into Encoder Module inputs.

    The same modality rules are used for live one-step acting and sampled replay
    windows. ``from_step`` adds a single-env batch dimension; ``from_batch``
    flattens replay's ``(B, T)`` prefix to the encoder batch axis.
    """

    def __init__(self, cfg: ObsBatchConfig):
        self.cfg = cfg

    def from_batch(self, obs: ObsInput) -> EncoderObs:
        """Return Encoder Module input for a sampled replay observation batch."""
        cfg = self.cfg
        replay_shape = obs_leading_shape(obs)
        compute_dtype = compute_jnp_dtype(cfg.compute_dtype)
        if (context_key := _RGB_CONTEXT_KEYS.get(cfg.encoder_type)) is not None:
            obs = pack_rgb_context_obs(obs, context_key=context_key)
        elif (
            structured_obs := _structured_encoder_batch(
                obs, cfg, replay_shape, compute_dtype
            )
        ) is not None:
            return structured_obs
        elif cfg.encoder_type in ("vggt", "vggt_wp_cp_64") and isinstance(obs, Mapping):
            obs = pack_world_points_camera_pose_obs(obs, dtype=compute_dtype)
        elif cfg.encoder_type == "vggt_wp_dense_cnn" and isinstance(obs, Mapping):
            obs = jnp.asarray(obs[WORLD_POINTS_KEY], dtype=compute_dtype)
        elif cfg.encoder_type == "cnn":
            obs = normalize_image_obs(obs)
        else:
            obs = jnp.asarray(obs, dtype=compute_dtype)
        return obs.reshape(replay_shape.steps, *_flat_obs_shape(cfg))

    def from_step(self, obs: ObsMapping) -> EncoderObs:
        """Return batched Encoder Module input for one live environment step."""
        cfg = self.cfg
        compute_dtype = compute_jnp_dtype(cfg.compute_dtype)
        if cfg.encoder_type == "hybrid":
            if "hybrid" in obs:
                encoder_obs = jnp.asarray(obs["hybrid"], dtype=jnp.float32)
            else:
                encoder_obs = pack_rgb_context_obs(obs, context_key=HYBRID_WP_CP_KEY)
        elif (context_key := _RGB_CONTEXT_KEYS.get(cfg.encoder_type)) is not None:
            encoder_obs = pack_rgb_context_obs(obs, context_key=context_key)
        elif (
            structured_obs := _structured_encoder_step(obs, cfg, compute_dtype)
        ) is not None:
            return structured_obs
        elif cfg.encoder_type == "vggt_wp_dense_cnn" and WORLD_POINTS_KEY in obs:
            encoder_obs = jnp.asarray(obs[WORLD_POINTS_KEY], dtype=compute_dtype)
        elif cfg.encoder_type in ("vggt", "vggt_wp_cp_64") and WORLD_POINTS_KEY in obs:
            encoder_obs = pack_world_points_camera_pose_obs(obs, dtype=compute_dtype)
        elif cfg.encoder_type in FLAT_VGGT_ENCODER_TYPES:
            encoder_obs = jnp.asarray(obs["features"], dtype=compute_dtype)
        else:
            encoder_obs = normalize_image_obs(obs["image"])
        return encoder_obs[None]


def encoder_obs_from_batch(batch: dict[str, Any], cfg: ObsBatchConfig) -> EncoderObs:
    """Return flattened per-step observations consumed by ``agent.encoder_mod``."""
    return ObservationPacker(cfg).from_batch(batch["obs"])


def decoder_rgb_target(batch: dict[str, Any], cfg: ObsBatchConfig) -> jnp.ndarray:
    """Return decoder RGB targets as ``(B*T, 3, 64, 64)`` in ``[0, 1]``."""
    obs = batch["obs"]
    replay_shape = obs_leading_shape(obs)
    if cfg.encoder_type in (
        "hybrid",
        "vggt_house_context",
        "vggt_house_full_tokens_nogate",
        "vggt_house_global_tokens_nogate",
    ):
        if isinstance(obs, Mapping):
            image = normalize_image_obs(obs[HYBRID_IMAGE_KEY])
            return image.reshape(replay_shape.steps, 3, 64, 64)
        obs_shape = _flat_obs_shape(cfg)
        rgb_dim = obs_shape[0] - cfg.vggt_feature_dim
        return (
            jnp.asarray(obs, dtype=jnp.float32)
            .reshape(replay_shape.steps, -1)[:, :rgb_dim]
            .reshape(replay_shape.steps, 3, 64, 64)
        )
    image = normalize_image_obs(obs)
    return image.reshape(replay_shape.steps, 3, 64, 64)
