"""Decoder target preparation for auxiliary reconstruction losses."""

from typing import Any

import jax.numpy as jnp

from src.buffer.replay_buffer import ReplayBatch

RGB_SHAPE = (64, 64, 3)


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


def decoder_rgb_target(batch: ReplayBatch, rgb_key: str) -> jnp.ndarray:
    """Return decoder RGB targets as ``(B*T, 64, 64, 3)`` in ``[0, 1]``.

    Args:
        batch: Sampled replay batch. Routed adapters always store their fields
            under explicit keys, so ``batch.obs`` is a mapping.
        rgb_key: Replay key of the field the adapter flagged as the decoder
            target.

    Returns:
        Normalized RGB targets flattened over batch and time.
    """
    batch_size, time_steps = replay_batch_shape(batch)
    flat = batch_size * time_steps
    return _normalize_image_obs(batch.obs[rgb_key]).reshape(flat, *RGB_SHAPE)
