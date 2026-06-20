"""Helpers for modality-aware replay observations.

Replay storage may keep modalities under explicit keys, while the current
Flax encoders still consume a single tensor. This module is the narrow bridge
between those two contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias

import jax.numpy as jnp


HYBRID_IMAGE_KEY = "image"
HYBRID_WP_CP_KEY = "wp_cp"
HOUSE_CONTEXT_KEY = "house_context"
FULL_TOKENS_KEY = "full_tokens"
GLOBAL_TOKENS_KEY = "global_tokens"
WORLD_POINTS_KEY = "world_points"
CAMERA_POSE_KEY = "camera_pose"

ObsShape: TypeAlias = tuple[int, ...] | Mapping[str, tuple[int, ...]]
EncoderObs: TypeAlias = jnp.ndarray | dict[str, jnp.ndarray]


class ObsBatchConfig(Protocol):
    """Config fields needed by observation batch packing helpers.

    Keep this protocol narrower than ``R2DreamerConfig`` so tests and launch-time
    shims can pass config-like objects without importing the concrete dataclass.
    """

    encoder_type: str
    obs_shape: ObsShape
    compute_dtype: str
    vggt_feature_dim: int
    vggt_token_count: int
    vggt_token_dim: int


def compute_jnp_dtype(dtype: str):
    """Return the JAX dtype named by ``R2DreamerConfig.compute_dtype``."""
    if dtype == "float32":
        return jnp.float32
    if dtype in ("bfloat16", "bf16"):
        return jnp.bfloat16
    if dtype in ("float16", "fp16"):
        return jnp.float16
    raise ValueError(f"Unsupported compute_dtype={dtype!r}")


def obs_leading_shape(obs: Any) -> tuple[int, int]:
    """Return the ``(B, T)`` prefix of a replay observation batch."""
    if isinstance(obs, Mapping):
        first = obs[HYBRID_IMAGE_KEY] if HYBRID_IMAGE_KEY in obs else next(iter(obs.values()))
        return first.shape[0], first.shape[1]
    return obs.shape[0], obs.shape[1]


def normalize_image_obs(image: Any) -> jnp.ndarray:
    """Return CHW image observations as float32 in ``[0, 1]``.

    Direct unit tests often pass already-normalized float arrays, while replay
    can now pass compact uint8 images. Branch on dtype so both inputs work.
    """
    image = jnp.asarray(image)
    if image.dtype == jnp.uint8:
        return image.astype(jnp.float32) / 255.0
    return image.astype(jnp.float32)


def _features(obs: Mapping[str, Any], context_key: str) -> Any:
    if context_key in obs:
        return obs[context_key]
    if context_key == HYBRID_WP_CP_KEY and "features" in obs:
        return obs["features"]
    raise KeyError(f"obs must contain {context_key!r}")


def pack_rgb_context_obs(obs: Any, *, context_key: str) -> jnp.ndarray:
    """Pack dict observations into the legacy flat ``[rgb | context]`` tensor."""
    if not isinstance(obs, Mapping):
        return jnp.asarray(obs, dtype=jnp.float32)
    image = normalize_image_obs(obs[HYBRID_IMAGE_KEY])
    features = jnp.asarray(_features(obs, context_key), dtype=jnp.float32)
    prefix = image.shape[:-3]
    image_flat = image.reshape(*prefix, -1)
    features_flat = features.reshape(*features.shape[:-1], -1)
    return jnp.concatenate([image_flat, features_flat], axis=-1)


def pack_hybrid_obs(obs: Any) -> jnp.ndarray:
    """Pack hybrid dict observations into the legacy flat encoder tensor."""
    return pack_rgb_context_obs(obs, context_key=HYBRID_WP_CP_KEY)


def pack_world_points_camera_pose_obs(
    obs: Mapping[str, Any],
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


def encoder_obs_from_batch(batch: dict[str, Any], cfg: ObsBatchConfig) -> EncoderObs:
    """Return flattened per-step observations consumed by ``agent.encoder_mod``."""
    obs = batch["obs"]
    B, T = obs_leading_shape(obs)
    compute_dtype = compute_jnp_dtype(cfg.compute_dtype)
    if cfg.encoder_type == "hybrid":
        obs = pack_hybrid_obs(obs)
    elif cfg.encoder_type == "vggt_house_context":
        obs = pack_rgb_context_obs(obs, context_key=HOUSE_CONTEXT_KEY)
    elif cfg.encoder_type == "vggt_house_full_tokens_nogate":
        if not isinstance(obs, Mapping):
            raise TypeError("vggt_house_full_tokens_nogate expects dict obs")
        image = normalize_image_obs(obs[HYBRID_IMAGE_KEY]).reshape(B * T, 3, 64, 64)
        tokens = jnp.asarray(
            obs[FULL_TOKENS_KEY], dtype=compute_jnp_dtype(cfg.compute_dtype)
        ).reshape(B * T, cfg.vggt_token_count, cfg.vggt_token_dim)
        return {HYBRID_IMAGE_KEY: image, FULL_TOKENS_KEY: tokens}
    elif cfg.encoder_type == "vggt_house_global_tokens_nogate":
        if not isinstance(obs, Mapping):
            raise TypeError("vggt_house_global_tokens_nogate expects dict obs")
        image = normalize_image_obs(obs[HYBRID_IMAGE_KEY]).reshape(B * T, 3, 64, 64)
        tokens = jnp.asarray(
            obs[GLOBAL_TOKENS_KEY], dtype=compute_jnp_dtype(cfg.compute_dtype)
        )
        expected_shape = (1, cfg.vggt_token_count, cfg.vggt_token_dim)
        if tokens.shape != expected_shape:
            raise ValueError(
                "vggt_house_global_tokens_nogate expects singleton live global "
                f"tokens with shape {expected_shape}, got {tokens.shape}"
            )
        return {HYBRID_IMAGE_KEY: image, GLOBAL_TOKENS_KEY: tokens}
    elif cfg.encoder_type == "vggt_wp64_cnn_cp_mlp":
        if not isinstance(obs, Mapping):
            raise TypeError("vggt_wp64_cnn_cp_mlp expects dict obs")
        world_points_shape = tuple(cfg.obs_shape[WORLD_POINTS_KEY])  # type: ignore[index]
        camera_pose_shape = tuple(cfg.obs_shape[CAMERA_POSE_KEY])  # type: ignore[index]
        world_points = jnp.asarray(obs[WORLD_POINTS_KEY], dtype=compute_dtype).reshape(
            B * T, *world_points_shape
        )
        camera_pose = jnp.asarray(obs[CAMERA_POSE_KEY], dtype=compute_dtype).reshape(
            B * T, *camera_pose_shape
        )
        return {WORLD_POINTS_KEY: world_points, CAMERA_POSE_KEY: camera_pose}
    elif cfg.encoder_type in ("vggt", "vggt_wp_cp_64") and isinstance(obs, Mapping):
        obs = pack_world_points_camera_pose_obs(obs, dtype=compute_dtype)
    elif cfg.encoder_type == "vggt_wp_dense_cnn" and isinstance(obs, Mapping):
        obs = jnp.asarray(obs[WORLD_POINTS_KEY], dtype=compute_dtype)
    elif cfg.encoder_type == "cnn":
        obs = normalize_image_obs(obs)
    else:
        obs = jnp.asarray(obs, dtype=compute_dtype)
    return obs.reshape(B * T, *_flat_obs_shape(cfg))


def encoder_obs_from_agent_obs(
    obs_dict: Mapping[str, Any], cfg: ObsBatchConfig
) -> EncoderObs:
    """Return one-step encoder input for acting."""
    compute_dtype = compute_jnp_dtype(cfg.compute_dtype)
    if cfg.encoder_type == "hybrid":
        if "hybrid" in obs_dict:
            obs = jnp.asarray(obs_dict["hybrid"], dtype=jnp.float32)
        else:
            obs = pack_hybrid_obs(obs_dict)
    elif cfg.encoder_type == "vggt_house_context":
        obs = pack_rgb_context_obs(obs_dict, context_key=HOUSE_CONTEXT_KEY)
    elif cfg.encoder_type == "vggt_house_full_tokens_nogate":
        obs = {
            HYBRID_IMAGE_KEY: normalize_image_obs(obs_dict[HYBRID_IMAGE_KEY])[None],
            FULL_TOKENS_KEY: jnp.asarray(
                obs_dict[FULL_TOKENS_KEY], dtype=compute_jnp_dtype(cfg.compute_dtype)
            )[None],
        }
        return obs
    elif cfg.encoder_type == "vggt_house_global_tokens_nogate":
        tokens = jnp.asarray(
            obs_dict[GLOBAL_TOKENS_KEY], dtype=compute_jnp_dtype(cfg.compute_dtype)
        )
        if tokens.ndim == 2:
            tokens = tokens[None]
        obs = {
            HYBRID_IMAGE_KEY: normalize_image_obs(obs_dict[HYBRID_IMAGE_KEY])[None],
            GLOBAL_TOKENS_KEY: tokens,
        }
        return obs
    elif cfg.encoder_type == "vggt_wp64_cnn_cp_mlp":
        return {
            WORLD_POINTS_KEY: jnp.asarray(
                obs_dict[WORLD_POINTS_KEY], dtype=compute_dtype
            )[None],
            CAMERA_POSE_KEY: jnp.asarray(
                obs_dict[CAMERA_POSE_KEY], dtype=compute_dtype
            )[None],
        }
    elif cfg.encoder_type == "vggt_wp_dense_cnn" and WORLD_POINTS_KEY in obs_dict:
        obs = jnp.asarray(obs_dict[WORLD_POINTS_KEY], dtype=compute_dtype)
    elif cfg.encoder_type in ("vggt", "vggt_wp_cp_64") and WORLD_POINTS_KEY in obs_dict:
        obs = pack_world_points_camera_pose_obs(obs_dict, dtype=compute_dtype)
    elif cfg.encoder_type in (
        "vggt",
        "vggt_aggregator_mlp",
        "vggt_agg_token_transformer",
        "vggt_wp_dense_cnn",
        "vggt_wp_cp_64",
    ):
        obs = jnp.asarray(obs_dict["features"], dtype=compute_dtype)
    else:
        obs = normalize_image_obs(obs_dict["image"])
    return obs[None]


def decoder_rgb_target(batch: dict[str, Any], cfg: ObsBatchConfig) -> jnp.ndarray:
    """Return decoder RGB targets as ``(B*T, 3, 64, 64)`` in ``[0, 1]``."""
    obs = batch["obs"]
    B, T = obs_leading_shape(obs)
    if cfg.encoder_type in (
        "hybrid",
        "vggt_house_context",
        "vggt_house_full_tokens_nogate",
        "vggt_house_global_tokens_nogate",
    ):
        if isinstance(obs, Mapping):
            image = normalize_image_obs(obs[HYBRID_IMAGE_KEY])
            return image.reshape(B * T, 3, 64, 64)
        obs_shape = _flat_obs_shape(cfg)
        rgb_dim = obs_shape[0] - cfg.vggt_feature_dim
        return (
            jnp.asarray(obs, dtype=jnp.float32)
            .reshape(B * T, -1)[:, :rgb_dim]
            .reshape(B * T, 3, 64, 64)
        )
    image = normalize_image_obs(obs)
    return image.reshape(B * T, 3, 64, 64)
