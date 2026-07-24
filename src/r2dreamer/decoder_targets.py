"""Decoder target preparation for auxiliary reconstruction losses."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from src.buffer.replay_buffer import ReplayBatch
from src.r2dreamer.observation_keys import HYBRID_IMAGE_KEY


def _composite_rgb_encoder_types() -> frozenset[str]:
    """Encoder types whose observation packs RGB alongside other modalities.

    Derived from the recipe registry's ``rgb_key`` fields (DELETIONS.md: the
    global name lists are gone). ``cnn`` is excluded: its observation *is*
    the RGB image, so it needs no extraction. Resolved lazily to keep this
    module import-light for the learner.
    """
    from src.r2dreamer.encoders.recipes import RECIPES

    return frozenset(
        name
        for name, recipe in RECIPES.items()
        if recipe.rgb_key is not None and name != "cnn"
    )


def _normalize_image_obs(image: Any) -> jnp.ndarray:
    """Return HWC image observations as float32 in ``[0, 1]``."""
    image = jnp.asarray(image)
    if image.dtype == jnp.uint8:
        return image.astype(jnp.float32) / 255.0
    return image.astype(jnp.float32)


def replay_batch_shape(batch: ReplayBatch) -> tuple[int, int]:
    """Return ``(B, T)`` from a sampled replay batch."""
    actions = batch.actions
    return int(actions.shape[0]), int(actions.shape[1])


def decoder_rgb_target(batch: ReplayBatch, encoder_type: str) -> jnp.ndarray:
    """Return decoder RGB targets as ``(B*T, 64, 64, 3)`` in ``[0, 1]``."""
    obs = batch.obs
    batch_size, time_steps = replay_batch_shape(batch)
    if encoder_type in _composite_rgb_encoder_types():
        if isinstance(obs, Mapping):
            image = _normalize_image_obs(obs[HYBRID_IMAGE_KEY])
            return image.reshape(batch_size * time_steps, 64, 64, 3)
        rgb_dim = 64 * 64 * 3
        return (
            jnp.asarray(obs, dtype=jnp.float32)
            .reshape(batch_size * time_steps, -1)[:, :rgb_dim]
            .reshape(batch_size * time_steps, 64, 64, 3)
        )
    image = obs[HYBRID_IMAGE_KEY] if isinstance(obs, Mapping) else obs
    return _normalize_image_obs(image).reshape(batch_size * time_steps, 64, 64, 3)
