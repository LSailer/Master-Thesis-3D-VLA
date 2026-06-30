"""Deterministic static house point-cloud context for R2Dreamer.

The first prototype compresses a fixed RGB point cloud into the existing
``HOUSE_CONTEXT_DIM`` vector consumed by ``vggt_house_context``. The layout is a
fixed ``8 x 8 x 4`` XYZ grid. Each voxel contributes normalized occupancy and
mean RGB, yielding ``8 * 8 * 4 * 4 == 1024`` float16 values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

# Host-side PLY IO and voxel accumulation use NumPy; JAX conversion happens in
# the observation packer before the Encoder Module runs.
import numpy as np

from src.r2dreamer.encoders.constants import HOUSE_CONTEXT_DIM


DEFAULT_STATIC_HOUSE_GRID = (8, 8, 4)
_STATS_PER_VOXEL = 4
_REQUIRED_VERTEX_FIELDS = ("x", "y", "z", "red", "green", "blue")


class PlyFormatError(ValueError):
    """Raised when a PLY file cannot be read as an ASCII XYZRGB point cloud."""


@dataclass(frozen=True)
class _PlyVertexLayout:
    vertex_count: int
    usecols: tuple[int, ...]


def _validated_grid_shape(grid_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    shape = tuple(int(axis) for axis in grid_shape)
    if len(shape) != 3 or any(axis <= 0 for axis in shape):
        raise ValueError(f"grid_shape must contain three positive axes, got {shape}")
    encoded_dim = int(np.prod(shape)) * _STATS_PER_VOXEL
    if encoded_dim != HOUSE_CONTEXT_DIM:
        raise ValueError(
            f"grid_shape {shape} encodes {encoded_dim} values, "
            f"expected {HOUSE_CONTEXT_DIM}"
        )
    return shape


def _read_ply_layout(handle: TextIO) -> _PlyVertexLayout:
    if handle.readline().strip() != "ply":
        raise PlyFormatError("expected PLY header to start with 'ply'")

    vertex_count: int | None = None
    vertex_properties: list[str] = []
    current_element: str | None = None
    saw_ascii_format = False

    for raw_line in handle:
        line = raw_line.strip()
        if line == "end_header":
            break
        parts = line.split()
        if not parts or parts[0] == "comment":
            continue
        if parts[0] == "format":
            saw_ascii_format = len(parts) >= 2 and parts[1] == "ascii"
            continue
        if parts[0] == "element" and len(parts) >= 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            continue
        if parts[0] == "property" and current_element == "vertex":
            vertex_properties.append(parts[-1])
    else:
        raise PlyFormatError("PLY header is missing 'end_header'")

    if not saw_ascii_format:
        raise PlyFormatError("only ascii PLY files are supported")
    if vertex_count is None:
        raise PlyFormatError("PLY header is missing a vertex element")

    try:
        usecols = tuple(vertex_properties.index(name) for name in _REQUIRED_VERTEX_FIELDS)
    except ValueError as exc:
        raise PlyFormatError(
            "PLY vertex properties must include x, y, z, red, green, blue"
        ) from exc

    return _PlyVertexLayout(vertex_count=vertex_count, usecols=usecols)


def load_ascii_ply_xyzrgb(path: str | Path) -> np.ndarray:
    """Load an ASCII PLY point cloud as ``float32`` rows ``[x, y, z, r, g, b]``."""
    ply_path = Path(path)
    with ply_path.open("r", encoding="ascii") as handle:
        layout = _read_ply_layout(handle)
        if layout.vertex_count == 0:
            return np.empty((0, len(_REQUIRED_VERTEX_FIELDS)), dtype=np.float32)
        points = np.loadtxt(
            handle,
            dtype=np.float32,
            max_rows=layout.vertex_count,
            usecols=layout.usecols,
        )

    return np.atleast_2d(points).astype(np.float32, copy=False)


def _finite_xyzrgb(points_xyzrgb: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyzrgb, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < len(_REQUIRED_VERTEX_FIELDS):
        raise ValueError(
            "points_xyzrgb must have shape (N, >=6) with columns x, y, z, r, g, b"
        )
    points = points[:, : len(_REQUIRED_VERTEX_FIELDS)]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] == 0:
        raise ValueError("cannot encode static house context from zero finite points")
    return points


def _normalised_rgb(rgb: np.ndarray) -> np.ndarray:
    colour = np.asarray(rgb, dtype=np.float32)
    if float(np.max(colour)) > 1.0:
        colour = colour / 255.0
    return np.clip(colour, 0.0, 1.0)


def _voxel_indices(xyz: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    xyz_min = np.min(xyz, axis=0)
    xyz_span = np.max(xyz, axis=0) - xyz_min
    xyz_span = np.where(xyz_span > 0.0, xyz_span, 1.0)
    normalised = np.clip((xyz - xyz_min) / xyz_span, 0.0, 1.0)
    normalised = np.minimum(normalised, np.nextafter(np.float32(1.0), np.float32(0.0)))

    grid = np.asarray(grid_shape, dtype=np.int64)
    coords = np.floor(normalised * grid).astype(np.int64)
    coords = np.clip(coords, 0, grid - 1)
    return np.ravel_multi_index((coords[:, 0], coords[:, 1], coords[:, 2]), grid_shape)


def encode_static_house_context(
    points_xyzrgb: np.ndarray,
    *,
    grid_shape: tuple[int, int, int] = DEFAULT_STATIC_HOUSE_GRID,
) -> np.ndarray:
    """Encode XYZRGB points into the fixed ``float16 (1024,)`` house context.

    Args:
        points_xyzrgb: Point rows with at least six columns: ``x, y, z, r, g, b``.
            RGB may be either ``0..255`` uchar-style values or already normalized
            ``0..1`` floats.
        grid_shape: XYZ voxel grid. The default ``(8, 8, 4)`` is the only shape
            that currently matches ``HOUSE_CONTEXT_DIM``.

    Returns:
        Flattened ``float16`` context. Per voxel layout is
        ``[log_count_norm, mean_r, mean_g, mean_b]``.
    """
    grid = _validated_grid_shape(grid_shape)
    points = _finite_xyzrgb(points_xyzrgb)
    xyz = points[:, :3]
    rgb = _normalised_rgb(points[:, 3:6])

    voxel_count = int(np.prod(grid))
    flat_indices = _voxel_indices(xyz, grid)
    counts = np.bincount(flat_indices, minlength=voxel_count).astype(np.float32)

    rgb_sums = np.zeros((voxel_count, 3), dtype=np.float32)
    np.add.at(rgb_sums, flat_indices, rgb)
    mean_rgb = np.divide(
        rgb_sums,
        counts[:, None],
        out=np.zeros_like(rgb_sums),
        where=counts[:, None] > 0.0,
    )

    occupancy = np.log1p(counts)
    occupancy_max = float(np.max(occupancy))
    if occupancy_max > 0.0:
        occupancy = occupancy / occupancy_max

    context = np.concatenate([occupancy[:, None], mean_rgb], axis=1).reshape(-1)
    return context.astype(np.float16)
