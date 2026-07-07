"""Shape helpers used inside encoder implementations."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp


def normalize_image_obs(image: Any, dtype: Any = jnp.float32) -> jnp.ndarray:
    """Return CHW image observations as ``dtype`` in ``[0, 1]``.

    Accepts either uint8 images or already-normalized floating arrays and
    preserves all leading dimensions before the final ``(3, H, W)`` axes.

    Args:
        image: uint8 or floating image array with trailing ``(3, H, W)`` axes.
        dtype: Floating dtype of the returned array (default float32; encoders
            pass their compute dtype under the full_bf16 gate).

    Returns:
        Image array in ``[0, 1]`` with dtype ``dtype`` and unchanged shape.
    """
    image = jnp.asarray(image)
    if image.dtype == jnp.uint8:
        return image.astype(dtype) / 255.0
    return image.astype(dtype)


def flatten_event(x: Any, event_ndims: int) -> tuple[jnp.ndarray, tuple[int, ...]]:
    """Flatten all leading dims before ``event_ndims`` event axes into one batch.

    Returns ``(flat, leading_shape)``. A tensor with only event axes is treated as
    one unbatched item and gets a singleton flat batch dimension.
    """
    array = jnp.asarray(x)
    if event_ndims <= 0:
        raise ValueError(f"event_ndims must be positive, got {event_ndims}")
    if array.ndim < event_ndims:
        raise ValueError(
            f"expected at least {event_ndims} dims, got shape {array.shape}"
        )
    leading_shape = array.shape[:-event_ndims]
    event_shape = array.shape[-event_ndims:]
    if not leading_shape:
        return array.reshape(1, *event_shape), ()
    return array.reshape(-1, *event_shape), leading_shape


def restore_leading(x: jnp.ndarray, leading_shape: tuple[int, ...]) -> jnp.ndarray:
    """Restore flattened encoder outputs to their original leading dimensions."""
    if not leading_shape:
        return x[0]
    return x.reshape(*leading_shape, *x.shape[1:])


def batch_live_observation(obs: Any) -> Any:
    """Add a single-env batch dimension to an unbatched live observation tree."""
    if isinstance(obs, Mapping):
        return {
            key: jnp.asarray(value)[None]
            for key, value in obs.items()
            if key != "is_first"
        }
    return jnp.asarray(obs)[None]
