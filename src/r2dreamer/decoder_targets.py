"""Decoder target preparation for auxiliary reconstruction losses."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from src.buffer.replay_buffer import ReplayBatch
from src.r2dreamer.observation_keys import HYBRID_IMAGE_KEY


def _normalize_image_obs(image: Any) -> jnp.ndarray:
    """Return CHW image observations as float32 in ``[0, 1]``."""
    image = jnp.asarray(image)
    if image.dtype == jnp.uint8:
        return image.astype(jnp.float32) / 255.0
    return image.astype(jnp.float32)


def replay_batch_shape(batch: ReplayBatch) -> tuple[int, int]:
    """Return ``(B, T)`` from a sampled replay batch."""
    actions = batch.actions
    return int(actions.shape[0]), int(actions.shape[1])


def decoder_rgb_target(batch: ReplayBatch, encoder_type: str) -> jnp.ndarray:
    """Return decoder RGB targets as ``(B*T, 3, 64, 64)`` in ``[0, 1]``."""
    obs = batch.obs
    B, T = replay_batch_shape(batch)
    if encoder_type in (
        "hybrid",
        "vggt_house_context",
        "vggt_house_full_tokens_nogate",
        "vggt_house_global_tokens_nogate",
    ):
        if isinstance(obs, Mapping):
            image = _normalize_image_obs(obs[HYBRID_IMAGE_KEY])
            return image.reshape(B * T, 3, 64, 64)
        rgb_dim = 3 * 64 * 64
        return (
            jnp.asarray(obs, dtype=jnp.float32)
            .reshape(B * T, -1)[:, :rgb_dim]
            .reshape(B * T, 3, 64, 64)
        )
    image = obs[HYBRID_IMAGE_KEY] if isinstance(obs, Mapping) else obs
    return _normalize_image_obs(image).reshape(B * T, 3, 64, 64)
