"""Shared PLY point-cloud I/O (XYZRGB vertex lists, Open3D-backed).

Files are written binary by default; read them back with
``open3d.io.read_point_cloud``. The legacy ASCII reader
(``static_house_context.load_ascii_ply_xyzrgb``) cannot parse this
format — pass ``write_ascii=True`` only if a text PLY is required.
"""

import os
from pathlib import Path

import jax.numpy as jnp

# Open3D consumes concrete host arrays; NumPy marks the device -> host
# file boundary (project convention: NumPy for host-only I/O).
import numpy as np


def _flattened_xyz_rgb(
    xyz: jnp.ndarray, rgb: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Validate and flatten point/colour inputs to ``(N, 3)`` each.

    Args:
      xyz: ``(..., 3)`` float vertex coordinates.
      rgb: ``(..., 3)`` uint8 or float ``[0, 1]`` vertex colours.

    Returns:
      Tuple ``(xyz_flat, rgb_flat)`` of ``(N, 3)`` arrays.

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
    return xyz_flat, rgb_flat


def write_world_points_ply(
    path: str | Path,
    xyz: jnp.ndarray,
    rgb: jnp.ndarray,
    write_ascii: bool = False,
) -> None:
    """Write an XYZRGB PLY via Open3D (binary by default).

    Inputs may have any leading shape (``(N, 3)``, ``(H, W, 3)``,
    ``(1, H, W, 3)``, ...); they are flattened to one vertex per row. The
    trailing axis must be 3 — a CHW image must be transposed to HWC by the
    caller, since only the caller knows its channel layout. Read the file
    back with ``open3d.io.read_point_cloud``.

    Args:
      path: Output ``.ply`` path; parent directories are created.
      xyz: ``(..., 3)`` float vertex coordinates.
      rgb: ``(..., 3)`` uint8 or float ``[0, 1]`` vertex colours, flattening
        to the same vertex count as ``xyz``.
      write_ascii: Write ASCII instead of binary (larger, slower).

    Raises:
      ValueError: If a trailing axis is not 3, or the flattened vertex
        counts of ``xyz`` and ``rgb`` differ.
    """
    # Local import: Open3D is a heavy host-side dependency only this writer needs.
    import open3d as o3d

    xyz_flat, rgb_flat = _flattened_xyz_rgb(xyz, rgb)
    rgb01 = (
        jnp.asarray(rgb_flat, dtype=jnp.float32) / 255.0
        if rgb_flat.dtype == jnp.uint8
        else jnp.clip(rgb_flat, 0.0, 1.0)
    )

    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    # Open3D's Vector3dVector wraps Eigen float64 vectors and requires host
    # memory: np.asarray performs the explicit device -> host transfer.
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz_flat, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb01, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=write_ascii)
