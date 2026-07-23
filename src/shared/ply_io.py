"""Shared PLY point-cloud I/O (XYZRGB vertex lists, binary little-endian).

Vertex records are packed on device by a jitted JAX function (float32
coordinates bitcast to bytes, colours quantized to uint8) and written in
one host write — no Open3D dependency. Files use 15-byte records
(``float x/y/z`` + ``uchar red/green/blue``); read them back with
``open3d.io.read_point_cloud``. The legacy ASCII reader
(``static_house_context.load_ascii_ply_xyzrgb``) cannot parse this format.

Benchmarks vs the previous Open3D writer (prototyp/fast_ply_write): ~6x
faster on CPU, ~7-9x on an H100 node at 100k-5M points, ~1.8x smaller
files.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp


def flattened_xyz_rgb(
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
    rgb_flat = (
        jnp.asarray(rgb_flat, dtype=jnp.float32) / 255.0
        if rgb_flat.dtype == jnp.uint8
        else jnp.clip(rgb_flat, 0.0, 1.0)
    )
    return xyz_flat, rgb_flat


def _header(num_vertices: int) -> bytes:
    """Build the binary-little-endian PLY header for an XYZRGB vertex list.

    Args:
      num_vertices: Number of 15-byte vertex records following the header.

    Returns:
      The header as ASCII bytes, terminated by ``end_header\\n``.
    """
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_vertices}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")


@jax.jit
def _pack_vertices_device(xyz: jnp.ndarray, rgb01: jnp.ndarray) -> jnp.ndarray:
    """Pack XYZRGB vertices into raw PLY record bytes on device.

    Bitcasts float32 coordinates to their little-endian byte quadruples and
    interleaves them with quantized uint8 colours, yielding the exact byte
    stream of a ``binary_little_endian`` PLY vertex list.

    Args:
      xyz: ``(N, 3)`` float coordinates (cast to float32).
      rgb01: ``(N, 3)`` float colours in ``[0, 1]``.

    Returns:
      ``(N * 15,)`` uint8 array — 12 coordinate bytes + 3 colour bytes per
      vertex, in vertex order.
    """
    xyz_bytes = jax.lax.bitcast_convert_type(
        xyz.astype(jnp.float32), jnp.uint8
    ).reshape(xyz.shape[0], 12)
    rgb_u8 = jnp.round(rgb01 * 255.0).astype(jnp.uint8)
    return jnp.concatenate([xyz_bytes, rgb_u8], axis=1).reshape(-1)


def write_world_points_ply(
    path: str | Path,
    xyz: jnp.ndarray,
    rgb: jnp.ndarray,
) -> None:
    """Write an XYZRGB PLY (binary little-endian, packed on device in JAX).

    Inputs may have any leading shape (``(N, 3)``, ``(H, W, 3)``,
    ``(1, H, W, 3)``, ...); they are flattened to one vertex per row. The
    trailing axis must be 3 — a CHW image must be transposed to HWC by the
    caller, since only the caller knows its channel layout. Coordinates are
    stored as float32, colours as uint8. Read the file back with
    ``open3d.io.read_point_cloud``.

    Args:
      path: Output ``.ply`` path; parent directories are created.
      xyz: ``(..., 3)`` float vertex coordinates.
      rgb: ``(..., 3)`` uint8 or float ``[0, 1]`` vertex colours, flattening
        to the same vertex count as ``xyz``.

    Raises:
      ValueError: If a trailing axis is not 3, or the flattened vertex
        counts of ``xyz`` and ``rgb`` differ.
    """
    xyz_flat, rgb_flat = flattened_xyz_rgb(xyz, rgb)
    body = jax.device_get(_pack_vertices_device(xyz_flat, rgb_flat))

    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_header(xyz_flat.shape[0]))
        f.write(body.tobytes())
