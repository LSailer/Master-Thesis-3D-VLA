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


def _mapping_obs(obs: Any, encoder_type: str) -> Mapping[str, Any]:
    """Return a structured observation or fail with an encoder-specific error."""
    if not isinstance(obs, Mapping):
        raise TypeError(f"{encoder_type} expects dict obs")
    return obs


def _rgb_token_image(
    obs: Mapping[str, Any], batch_size: int, seq_len: int
) -> jnp.ndarray:
    """Return RGB replay images flattened from ``(B, T)`` to ``B*T``."""
    return normalize_image_obs(obs[HYBRID_IMAGE_KEY]).reshape(
        batch_size * seq_len, 3, 64, 64
    )


def _full_tokens_batch(
    obs: Any, cfg: ObsBatchConfig, batch_size: int, seq_len: int, compute_dtype
) -> EncoderObs:
    """Pack full-token replay fields for the RGB+token Encoder Module."""
    mapped = _mapping_obs(obs, "vggt_house_full_tokens_nogate")
    image = _rgb_token_image(mapped, batch_size, seq_len)
    tokens = jnp.asarray(mapped[FULL_TOKENS_KEY], dtype=compute_dtype).reshape(
        batch_size * seq_len, cfg.vggt_token_count, cfg.vggt_token_dim
    )
    return {HYBRID_IMAGE_KEY: image, FULL_TOKENS_KEY: tokens}


def _global_tokens_batch(
    obs: Any, cfg: ObsBatchConfig, batch_size: int, seq_len: int, compute_dtype
) -> EncoderObs:
    """Pack singleton global-token replay fields for the RGB+token encoder."""
    mapped = _mapping_obs(obs, "vggt_house_global_tokens_nogate")
    image = _rgb_token_image(mapped, batch_size, seq_len)
    tokens = jnp.asarray(mapped[GLOBAL_TOKENS_KEY], dtype=compute_dtype)
    expected_tokens = (1, cfg.vggt_token_count, cfg.vggt_token_dim)
    if tokens.shape != expected_tokens:
        raise ValueError(
            "vggt_house_global_tokens_nogate expects singleton "
            f"global tokens with shape {expected_tokens}, got {tokens.shape}"
        )
    return {HYBRID_IMAGE_KEY: image, GLOBAL_TOKENS_KEY: tokens}


def _wp64_cnn_cp_mlp_batch(
    obs: Any, cfg: ObsBatchConfig, batch_size: int, seq_len: int, compute_dtype
) -> EncoderObs:
    """Pack structured WP64+CP replay fields for the hybrid Encoder Module."""
    mapped = _mapping_obs(obs, "vggt_wp64_cnn_cp_mlp")
    world_points_shape = tuple(cfg.obs_shape[WORLD_POINTS_KEY])  # type: ignore[index]
    camera_pose_shape = tuple(cfg.obs_shape[CAMERA_POSE_KEY])  # type: ignore[index]
    world_points = jnp.asarray(mapped[WORLD_POINTS_KEY], dtype=compute_dtype).reshape(
        batch_size * seq_len, *world_points_shape
    )
    camera_pose = jnp.asarray(mapped[CAMERA_POSE_KEY], dtype=compute_dtype).reshape(
        batch_size * seq_len, *camera_pose_shape
    )
    return {WORLD_POINTS_KEY: world_points, CAMERA_POSE_KEY: camera_pose}


class ObservationPacker:
    """Pack prepared observations into Encoder Module inputs.

    The same modality rules are used for live one-step acting and sampled replay
    windows. ``from_step`` adds a single-env batch dimension; ``from_batch``
    flattens replay's ``(B, T)`` prefix to the encoder batch axis.
    """

    def __init__(self, cfg: ObsBatchConfig):
        self.cfg = cfg

    def from_batch(self, obs: Any) -> EncoderObs:
        """Return Encoder Module input for a sampled replay observation batch."""
        cfg = self.cfg
        batch_size, seq_len = obs_leading_shape(obs)
        compute_dtype = compute_jnp_dtype(cfg.compute_dtype)
        if cfg.encoder_type == "hybrid":
            obs = pack_hybrid_obs(obs)
        elif cfg.encoder_type == "vggt_house_context":
            obs = pack_rgb_context_obs(obs, context_key=HOUSE_CONTEXT_KEY)
        elif cfg.encoder_type == "vggt_house_full_tokens_nogate":
            return _full_tokens_batch(obs, cfg, batch_size, seq_len, compute_dtype)
        elif cfg.encoder_type == "vggt_house_global_tokens_nogate":
            return _global_tokens_batch(obs, cfg, batch_size, seq_len, compute_dtype)
        elif cfg.encoder_type == "vggt_wp64_cnn_cp_mlp":
            return _wp64_cnn_cp_mlp_batch(obs, cfg, batch_size, seq_len, compute_dtype)
        elif cfg.encoder_type in ("vggt", "vggt_wp_cp_64") and isinstance(
            obs, Mapping
        ):
            obs = pack_world_points_camera_pose_obs(obs, dtype=compute_dtype)
        elif cfg.encoder_type == "vggt_wp_dense_cnn" and isinstance(obs, Mapping):
            obs = jnp.asarray(obs[WORLD_POINTS_KEY], dtype=compute_dtype)
        elif cfg.encoder_type == "cnn":
            obs = normalize_image_obs(obs)
        else:
            obs = jnp.asarray(obs, dtype=compute_dtype)
        return obs.reshape(batch_size * seq_len, *_flat_obs_shape(cfg))

    def from_step(self, obs: Mapping[str, Any]) -> EncoderObs:
        """Return batched Encoder Module input for one live environment step."""
        cfg = self.cfg
        compute_dtype = compute_jnp_dtype(cfg.compute_dtype)
        if cfg.encoder_type == "hybrid":
            if "hybrid" in obs:
                encoder_obs = jnp.asarray(obs["hybrid"], dtype=jnp.float32)
            else:
                encoder_obs = pack_hybrid_obs(obs)
        elif cfg.encoder_type == "vggt_house_context":
            encoder_obs = pack_rgb_context_obs(obs, context_key=HOUSE_CONTEXT_KEY)
        elif cfg.encoder_type == "vggt_house_full_tokens_nogate":
            return {
                HYBRID_IMAGE_KEY: normalize_image_obs(obs[HYBRID_IMAGE_KEY])[None],
                FULL_TOKENS_KEY: jnp.asarray(
                    obs[FULL_TOKENS_KEY], dtype=compute_dtype
                )[None],
            }
        elif cfg.encoder_type == "vggt_house_global_tokens_nogate":
            tokens = jnp.asarray(obs[GLOBAL_TOKENS_KEY], dtype=compute_dtype)
            if tokens.ndim == 2:
                tokens = tokens[None]
            return {
                HYBRID_IMAGE_KEY: normalize_image_obs(obs[HYBRID_IMAGE_KEY])[None],
                GLOBAL_TOKENS_KEY: tokens,
            }
        elif cfg.encoder_type == "vggt_wp64_cnn_cp_mlp":
            return {
                WORLD_POINTS_KEY: jnp.asarray(
                    obs[WORLD_POINTS_KEY], dtype=compute_dtype
                )[None],
                CAMERA_POSE_KEY: jnp.asarray(
                    obs[CAMERA_POSE_KEY], dtype=compute_dtype
                )[None],
            }
        elif cfg.encoder_type == "vggt_wp_dense_cnn" and WORLD_POINTS_KEY in obs:
            encoder_obs = jnp.asarray(obs[WORLD_POINTS_KEY], dtype=compute_dtype)
        elif cfg.encoder_type in ("vggt", "vggt_wp_cp_64") and WORLD_POINTS_KEY in obs:
            encoder_obs = pack_world_points_camera_pose_obs(obs, dtype=compute_dtype)
        elif cfg.encoder_type in (
            "vggt",
            "vggt_aggregator_mlp",
            "vggt_agg_token_transformer",
            "vggt_wp_dense_cnn",
            "vggt_wp_cp_64",
        ):
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
    batch_size, seq_len = obs_leading_shape(obs)
    if cfg.encoder_type in (
        "hybrid",
        "vggt_house_context",
        "vggt_house_full_tokens_nogate",
        "vggt_house_global_tokens_nogate",
    ):
        if isinstance(obs, Mapping):
            image = normalize_image_obs(obs[HYBRID_IMAGE_KEY])
            return image.reshape(batch_size * seq_len, 3, 64, 64)
        obs_shape = _flat_obs_shape(cfg)
        rgb_dim = obs_shape[0] - cfg.vggt_feature_dim
        return (
            jnp.asarray(obs, dtype=jnp.float32)
            .reshape(batch_size * seq_len, -1)[:, :rgb_dim]
            .reshape(batch_size * seq_len, 3, 64, 64)
        )
    image = normalize_image_obs(obs)
    return image.reshape(batch_size * seq_len, 3, 64, 64)
