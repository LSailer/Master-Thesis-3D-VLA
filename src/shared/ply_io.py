"""Shared ASCII PLY point-cloud I/O (XYZRGB vertex lists)."""

import os
from pathlib import Path

import jax.numpy as jnp

# Host-side file writing uses NumPy (np.savetxt); everything up to the file
# boundary stays in jax.numpy.
import numpy as np


def write_world_points_ply(
    path: str | Path, xyz: jnp.ndarray, rgb: jnp.ndarray
) -> None:
    """Write an ASCII XYZRGB PLY round-trip-compatible with the reader.

    The reader (``static_house_context.load_ascii_ply_xyzrgb``) requires the
    ``x, y, z, red, green, blue`` vertex properties. Inputs may have any
    leading shape (``(N, 3)``, ``(H, W, 3)``, ``(1, H, W, 3)``, ...); they
    are flattened to one vertex per row. The trailing axis must be 3 — a
    CHW image must be transposed to HWC by the caller, since only the
    caller knows its channel layout.

    Args:
      path: Output ``.ply`` path; parent directories are created.
      xyz: ``(..., 3)`` float vertex coordinates.
      rgb: ``(..., 3)`` uint8 or float ``[0, 1]`` vertex colours, flattening
        to the same vertex count as ``xyz``.

    Raises:
      ValueError: If a trailing axis is not 3, or the flattened vertex
        counts of ``xyz`` and ``rgb`` differ.
    """
    if xyz.shape[-1] != 3 or rgb.shape[-1] != 3:
        raise ValueError(
            f"xyz/rgb must have a trailing axis of 3, got {xyz.shape} / "
            f"{rgb.shape} (CHW image? transpose to HWC first)"
        )
    xyz_flat = jnp.reshape(xyz, (-1, 3))
    rgb_flat = jnp.reshape(rgb, (-1, 3))
    if xyz_flat.shape[0] != rgb_flat.shape[0]:
        raise ValueError(
            f"xyz and rgb disagree on vertex count: {xyz.shape} flattens to "
            f"{xyz_flat.shape[0]} points, {rgb.shape} to {rgb_flat.shape[0]}"
        )

    n = int(xyz_flat.shape[0])
    rgb_u8 = (
        rgb_flat
        if rgb_flat.dtype == jnp.uint8
        else jnp.clip(rgb_flat, 0.0, 1.0).astype(jnp.float32) * 255.0
    ).astype(jnp.uint8)
    rows = jnp.concatenate([xyz_flat.astype(jnp.float32), rgb_u8], axis=1)  # (N, 6)

    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header"
    )
    np.savetxt(
        path,
        np.asarray(rows),  # explicit device -> host transfer at the file boundary
        fmt=("%.6f", "%.6f", "%.6f", "%d", "%d", "%d"),
        header=header,
        comments="",
    )
