"""ASCII PLY point-cloud I/O for prototype experiments.

Host-side NumPy is used for text parsing/writing (file I/O boundary); all
returned arrays are JAX arrays.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REQUIRED_PROPERTIES = ("x", "y", "z", "red", "green", "blue")


def load_ply_xyzrgb(path: str | PathLike[str]) -> tuple[jax.Array, jax.Array]:
    """Load an ASCII PLY into ``(N, 3)`` float32 xyz and ``(N, 3)`` uint8 rgb.

    Extra vertex properties (e.g. ``point_id``/``status_id``/``added_step`` in
    buffer snapshots) are tolerated and ignored. Raises ``ValueError`` for
    binary PLY files or missing x/y/z/red/green/blue properties.
    """
    ply_path = Path(path)
    with ply_path.open("r", encoding="utf-8", errors="replace") as ply_file:
        vertex_count, property_names = _parse_header(ply_file)
        columns = {name: index for index, name in enumerate(property_names)}
        missing = [name for name in _REQUIRED_PROPERTIES if name not in columns]
        if missing:
            raise ValueError(f"PLY {ply_path} missing vertex properties {missing}")
        usecols = tuple(columns[name] for name in _REQUIRED_PROPERTIES)
        rows = np.loadtxt(
            ply_file,
            dtype=np.float64,
            usecols=usecols,
            max_rows=vertex_count,
            ndmin=2,
        )
    if rows.shape[0] != vertex_count:
        raise ValueError(
            f"PLY {ply_path} declared {vertex_count} vertices, "
            f"parsed {rows.shape[0]}"
        )
    xyz = jnp.asarray(rows[:, :3], dtype=jnp.float32)
    rgb = jnp.asarray(np.clip(rows[:, 3:], 0, 255), dtype=jnp.uint8)
    return xyz, rgb


def save_ply_xyzrgb(
    path: str | PathLike[str], xyz: jax.Array, rgb: jax.Array
) -> Path:
    """Write ``(N, 3)`` xyz (any float dtype) and ``(N, 3)`` uint8 rgb as ASCII PLY."""
    points = np.asarray(jax.device_get(xyz), dtype=np.float32)
    colors = np.asarray(jax.device_get(rgb), dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected xyz shape (N, 3), got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"expected rgb shape {points.shape}, got {colors.shape}")

    ply_path = Path(path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with ply_path.open("w", encoding="utf-8") as ply_file:
        ply_file.write("\n".join(header))
        ply_file.write("\n")
        for point, color in zip(points, colors, strict=True):
            ply_file.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return ply_path


def _parse_header(ply_file) -> tuple[int, list[str]]:
    """Read the PLY header, returning the vertex count and property names."""
    magic = ply_file.readline().strip()
    if magic != "ply":
        raise ValueError(f"not a PLY file (magic line {magic!r})")

    vertex_count: int | None = None
    property_names: list[str] = []
    in_vertex_element = False
    for line in ply_file:
        tokens = line.split()
        if not tokens:
            continue
        keyword = tokens[0]
        if keyword == "format":
            if tokens[1] != "ascii":
                raise ValueError(f"only ASCII PLY supported, got format {tokens[1]}")
        elif keyword == "element":
            in_vertex_element = tokens[1] == "vertex"
            if in_vertex_element:
                vertex_count = int(tokens[2])
        elif keyword == "property" and in_vertex_element:
            property_names.append(tokens[-1])
        elif keyword == "end_header":
            break
    if vertex_count is None:
        raise ValueError("PLY header has no vertex element")
    return vertex_count, property_names
