"""Helpers for modality-aware replay observations.

Replay storage may keep modalities under explicit keys, while the current
Flax encoders still consume a single tensor. This module is the narrow bridge
between those two contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp


HYBRID_IMAGE_KEY = "image"
HYBRID_WP_CP_KEY = "wp_cp"


def obs_leading_shape(obs: Any) -> tuple[int, int]:
    """Return the ``(B, T)`` prefix of a replay observation batch."""
    if isinstance(obs, Mapping):
        first = next(iter(obs.values()))
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


def _hybrid_features(obs: Mapping[str, Any]) -> Any:
    if HYBRID_WP_CP_KEY in obs:
        return obs[HYBRID_WP_CP_KEY]
    if "features" in obs:
        return obs["features"]
    raise KeyError(
        f"hybrid obs must contain {HYBRID_WP_CP_KEY!r} or 'features'"
    )


def pack_hybrid_obs(obs: Any) -> jnp.ndarray:
    """Pack hybrid dict observations into the legacy flat encoder tensor."""
    if not isinstance(obs, Mapping):
        return jnp.asarray(obs, dtype=jnp.float32)
    image = normalize_image_obs(obs[HYBRID_IMAGE_KEY])
    features = jnp.asarray(_hybrid_features(obs), dtype=jnp.float32)
    prefix = image.shape[:-3]
    image_flat = image.reshape(*prefix, -1)
    features_flat = features.reshape(*features.shape[:-1], -1)
    return jnp.concatenate([image_flat, features_flat], axis=-1)


def encoder_obs_from_batch(batch: dict[str, Any], cfg: Any) -> jnp.ndarray:
    """Return flattened per-step observations consumed by ``agent.encoder_mod``."""
    obs = batch["obs"]
    B, T = obs_leading_shape(obs)
    if cfg.encoder_type == "hybrid":
        obs = pack_hybrid_obs(obs)
    elif cfg.encoder_type == "cnn":
        obs = normalize_image_obs(obs)
    else:
        obs = jnp.asarray(obs, dtype=jnp.float32)
    return obs.reshape(B * T, *cfg.obs_shape)


def encoder_obs_from_agent_obs(obs_dict: Mapping[str, Any], cfg: Any) -> jnp.ndarray:
    """Return one-step encoder input for acting."""
    if cfg.encoder_type == "hybrid":
        if "hybrid" in obs_dict:
            obs = jnp.asarray(obs_dict["hybrid"], dtype=jnp.float32)
        else:
            obs = pack_hybrid_obs(obs_dict)
    elif cfg.encoder_type in (
        "vggt", "vggt_aggregator_mlp", "vggt_wp_dense_cnn", "vggt_wp_cp_64",
    ):
        obs = jnp.asarray(obs_dict["features"], dtype=jnp.float32)
    else:
        obs = normalize_image_obs(obs_dict["image"])
    return obs[None]


def decoder_rgb_target(batch: dict[str, Any], cfg: Any) -> jnp.ndarray:
    """Return decoder RGB targets as ``(B*T, 3, 64, 64)`` in ``[0, 1]``."""
    obs = batch["obs"]
    B, T = obs_leading_shape(obs)
    if cfg.encoder_type == "hybrid":
        if isinstance(obs, Mapping):
            image = normalize_image_obs(obs[HYBRID_IMAGE_KEY])
            return image.reshape(B * T, 3, 64, 64)
        rgb_dim = cfg.obs_shape[0] - cfg.vggt_feature_dim
        return jnp.asarray(obs, dtype=jnp.float32).reshape(B * T, -1)[
            :, :rgb_dim
        ].reshape(B * T, 3, 64, 64)
    image = normalize_image_obs(obs)
    return image.reshape(B * T, 3, 64, 64)
