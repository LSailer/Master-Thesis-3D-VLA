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
    # Imported lazily to avoid a module-level import cycle: registry.py pulls in
    # encoders.cnn -> world_model -> world_model.loss -> back to this module.
    from src.r2dreamer.encoders.registry import RGB_BEARING_ENCODER_TYPES

    obs = batch.obs
    B, T = replay_batch_shape(batch)
    # "cnn" is RGB-bearing too but takes the flat else-branch reshape below
    # (single flat obs, not a Mapping / RGB-prefix split), so it is excluded
    # here — preserving the historical CNN fall-through exactly.
    if encoder_type in RGB_BEARING_ENCODER_TYPES and encoder_type != "cnn":
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
