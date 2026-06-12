"""Helpers for observation batches that may carry multiple modalities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np


HYBRID_IMAGE_KEY = "image"
HYBRID_WP_CP_KEY = "wp_cp"
HOUSE_CONTEXT_KEY = "house_context"
HYBRID_RGB_DIM = 3 * 64 * 64


def obs_leading_shape(obs: Any) -> tuple[int, ...]:
    """Return the shared leading shape of an array or observation mapping."""
    if isinstance(obs, Mapping):
        first = next(iter(obs.values()))
        return tuple(first.shape[:2])
    return tuple(obs.shape[:2])


def _normalize_image(image: Any) -> jnp.ndarray:
    image = jnp.asarray(image)
    if image.dtype == jnp.uint8:
        return image.astype(jnp.float32) / 255.0
    return image.astype(jnp.float32)


def pack_rgb_context_obs(obs: Mapping[str, Any], *, context_key: str) -> jnp.ndarray:
    """Pack ``{"image": ..., context_key: ...}`` as flat ``[rgb | context]``."""
    if HYBRID_IMAGE_KEY not in obs:
        raise KeyError(f"missing observation field {HYBRID_IMAGE_KEY!r}")
    if context_key not in obs:
        raise KeyError(f"missing observation field {context_key!r}")

    image = _normalize_image(obs[HYBRID_IMAGE_KEY])
    context = jnp.asarray(obs[context_key], dtype=jnp.float32)
    rgb = image.reshape(*image.shape[:-3], HYBRID_RGB_DIM)
    return jnp.concatenate([rgb, context], axis=-1)


def encoder_obs_from_batch(batch: Mapping[str, Any], cfg: Any) -> jnp.ndarray:
    """Return the array consumed by the configured encoder from a train batch."""
    obs = batch["obs"]
    if cfg.encoder_type == "vggt_house_context":
        return pack_rgb_context_obs(obs, context_key=HOUSE_CONTEXT_KEY)
    if cfg.encoder_type == "hybrid" and isinstance(obs, Mapping):
        return pack_rgb_context_obs(obs, context_key=HYBRID_WP_CP_KEY)
    return obs


def encoder_obs_from_agent_obs(obs_dict: Mapping[str, Any], cfg: Any) -> jnp.ndarray:
    """Return a single-step encoder observation from an env adapter output."""
    if cfg.encoder_type == "vggt_house_context":
        return pack_rgb_context_obs(obs_dict, context_key=HOUSE_CONTEXT_KEY)
    if cfg.encoder_type == "hybrid" and HYBRID_WP_CP_KEY in obs_dict:
        return pack_rgb_context_obs(obs_dict, context_key=HYBRID_WP_CP_KEY)
    if cfg.encoder_type == "hybrid":
        return jnp.asarray(obs_dict["hybrid"])
    if cfg.encoder_type in (
        "vggt",
        "vggt_aggregator_mlp",
        "vggt_agg_token_transformer",
        "vggt_wp_dense_cnn",
        "vggt_wp_cp_64",
    ):
        return jnp.asarray(obs_dict["features"])
    image = np.asarray(obs_dict["image"]).astype(np.float32) / 255.0
    return jnp.asarray(image)


def decoder_rgb_target(batch: Mapping[str, Any], cfg: Any) -> jnp.ndarray:
    """Return ``(B*T, 3, 64, 64)`` decoder targets from a train batch."""
    obs = batch["obs"]
    if isinstance(obs, Mapping):
        image = _normalize_image(obs[HYBRID_IMAGE_KEY])
        return image.reshape(-1, 3, 64, 64)
    if cfg.encoder_type in ("hybrid", "vggt_house_context"):
        rgb_dim = cfg.obs_shape[0] - cfg.vggt_feature_dim
        return obs.reshape(-1, obs.shape[-1])[:, :rgb_dim].reshape(-1, 3, 64, 64)
    return obs.reshape(-1, 3, 64, 64)
