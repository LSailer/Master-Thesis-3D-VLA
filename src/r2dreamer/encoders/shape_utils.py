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


def validate_house_points(house_points: jnp.ndarray, house_point_dim: int) -> jnp.ndarray:
    """Promote an unbatched house cloud to ``(S, N, D)`` and check its shape.

    Args:
        house_points: House point cloud shaped ``(N, D)`` or ``(S, N, D)``.
        house_point_dim: Expected trailing channel count ``D``.

    Returns:
        ``house_points`` with a leading snapshot axis, shape ``(S, N, D)``.

    Raises:
        ValueError: If the promoted array is not 3-D, or its trailing axis is
            not ``house_point_dim``.
    """
    if house_points.ndim == 2:
        house_points = house_points[None]
    if house_points.ndim != 3 or house_points.shape[-1] != house_point_dim:
        raise ValueError(
            "house_points must have shape (N, 6) or (S, N, 6), "
            f"got {house_points.shape}"
        )
    return house_points


def singleton_house_cloud(
    house_points: jnp.ndarray,
    house_size: jnp.ndarray | int | None,
    branch_name: str,
) -> tuple[jnp.ndarray, int, jnp.ndarray]:
    """Unwrap a singleton ``(1, N, D)`` house cloud and resolve its valid size.

    Args:
        house_points: House point cloud shaped ``(S, N, D)``, ``S`` must be 1.
        house_size: Count of valid leading rows, or ``None`` to treat every row
            as valid (legacy paths without the size key).
        branch_name: House-branch name used in the ``S != 1`` error message.

    Returns:
        Tuple ``(points, n_points, size)`` of the float32 ``(N, D)`` cloud, its
        static row count, and the valid-row count as an int32 scalar.

    Raises:
        ValueError: If the snapshot axis is not a singleton.
    """
    if house_points.shape[0] != 1:
        raise ValueError(
            f"{branch_name} house branch expects a singleton house cloud (S=1), "
            f"got S={house_points.shape[0]}"
        )
    points = house_points[0].astype(jnp.float32)
    n_points = points.shape[0]
    if house_size is None:
        house_size = n_points
    size = jnp.asarray(house_size, dtype=jnp.int32).reshape(-1)[0]
    return points, n_points, size


def batch_live_observation(obs: Any) -> Any:
    """Add a single-env batch dimension to an unbatched live observation tree."""
    if isinstance(obs, Mapping):
        return {
            key: jnp.asarray(value)[None]
            for key, value in obs.items()
            if key != "is_first"
        }
    return jnp.asarray(obs)[None]
