"""Typed interface for accumulating VGGT house-context points."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import ClassVar, TextIO

import jax
import jax.numpy as jnp
import numpy as np

from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGTExtractOutput


class HouseContextPoseBuffer:
    """Accumulate and save colored house-context points from VGGT outputs.

    Args:
        confidence_score: Minimum VGGT confidence score required for admitting
            points into the house context.
        scene_id: Scene identifier used for the output subdirectory.
        voxel_size_m: Edge length in metres of the voxel grid used to
            deduplicate accumulated points. Smaller keeps more detail.

    Side effects:
        ``add`` mutates ``points_xyz`` and ``colors_rgb`` when points are admitted.
        ``save`` writes the accumulated colored house context to disk.
    """

    NEW_STATUS_ID: ClassVar[int] = 2
    NEW_STATUS_COMMENT: ClassVar[str] = "newly_added_to_context"
    XYZ_CHANNELS: ClassVar[int] = 3
    DEFAULT_VOXEL_SIZE_M: ClassVar[float] = 0.01

    def __init__(
        self,
        confidence_score: float,
        scene_id: str,
        voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
    ) -> None:
        self.confidence_score = confidence_score
        self.scene_id = scene_id
        self.voxel_size_m = voxel_size_m
        self.points_xyz: jax.Array | None = None
        self.colors_rgb: jax.Array | None = None
        self._voxel_keys: set[tuple[int, int, int]] = set()

    def add(
        self,
        vggt_output: VGGTExtractOutput,
        observation: ObservationFrame,
    ) -> jax.Array:
        """Add one VGGT output and its source observation to the buffer.

        Args:
            vggt_output: Structured VGGT output for the source observation.
            observation: Environment frame that produced ``vggt_output``.

        Returns:
            The accumulated house-context points with shape ``(P, 3)`` and
            bfloat16 dtype.
        """
        flat_points, flat_rgb, confidence_flat = self._flatten_aligned_inputs(
            vggt_output,
            observation,
        )
        active_mask = self._high_confidence_finite_mask(flat_points, confidence_flat)
        if not bool(jnp.any(active_mask)):
            return self._current_points()

        valid_points = flat_points[active_mask]
        valid_rgb = flat_rgb[active_mask]
        unique_voxels, first_indices = self._unique_voxel_representatives(valid_points)
        new_positions = self._new_voxel_positions(unique_voxels)
        if not new_positions:
            return self._current_points()

        representative_indices = first_indices[
            jnp.asarray(new_positions, dtype=jnp.int32)
        ]
        return self._append_points(
            valid_points[representative_indices],
            valid_rgb[representative_indices],
        )

    def _flatten_aligned_inputs(
        self,
        vggt_output: VGGTExtractOutput,
        observation: ObservationFrame,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return aligned flat ``(P, 3)`` points/RGB and ``(P,)`` confidence."""
        points = jnp.asarray(vggt_output.world_points, dtype=jnp.float32)
        confidence_map = jnp.asarray(vggt_output.confidence, dtype=jnp.float32)
        rgb_hwc = np.moveaxis(observation.image, 0, -1)
        flat_points = points.reshape(-1, self.XYZ_CHANNELS)
        flat_rgb = jnp.asarray(rgb_hwc, dtype=jnp.uint8).reshape(-1, self.XYZ_CHANNELS)
        confidence_flat = confidence_map.reshape(-1)
        return flat_points, flat_rgb, confidence_flat

    def _empty_points(self) -> jax.Array:
        """Return the canonical empty house-context point buffer."""
        return jnp.empty((0, self.XYZ_CHANNELS), dtype=jnp.bfloat16)

    def _current_points(self) -> jax.Array:
        """Return the accumulated points, or an empty buffer before the first add."""
        if self.points_xyz is None:
            return self._empty_points()
        return self.points_xyz

    def save(self, output_path: str | PathLike[str]) -> Path:
        """Save accumulated colored house context as a PLY snapshot.

        Args:
            output_path: Destination root. A scene subdirectory is created below it
                containing ``step_00000_context.ply``.

        Returns:
            The scene directory that was written.
        """
        output_dir = Path(output_path) / self._safe_path_name(self.scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_house_context_ply(output_dir)
        return output_dir

    @staticmethod
    def _safe_path_name(name: str) -> str:
        """Return a filesystem-safe scene folder name."""
        safe_chars = [char if char.isalnum() or char in "._-" else "_" for char in name]
        safe_name = "".join(safe_chars).strip("._")
        return safe_name or "scene"

    def _high_confidence_finite_mask(
        self,
        flat_points: jax.Array,
        confidence_flat: jax.Array,
    ) -> jax.Array:
        """Return mask for finite points whose confidence passes admission."""
        finite_mask = jnp.isfinite(flat_points).all(axis=1) & jnp.isfinite(
            confidence_flat
        )
        return finite_mask & (confidence_flat >= self.confidence_score)

    def _unique_voxel_representatives(
        self,
        points_xyz: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return unique voxel keys and first point indices for representatives."""
        scaled = jnp.asarray(points_xyz) / self.voxel_size_m
        quantized = jnp.floor(scaled).astype(jnp.int32)
        unique_voxels, first_indices = jnp.unique(quantized, axis=0, return_index=True)
        return unique_voxels, first_indices

    def _new_voxel_positions(self, unique_voxels: jax.Array) -> list[int]:
        """Return positions whose voxel keys are not yet stored.

        ``_voxel_keys`` is kept in sync here as points are admitted, so it always
        reflects the voxels already in the buffer.
        """
        new_positions: list[int] = []
        # Host-side set membership needs Python ints; pull the small voxel grid
        # to host once rather than syncing per element.
        for unique_pos, voxel in enumerate(unique_voxels.tolist()):
            key = (voxel[0], voxel[1], voxel[2])
            if key not in self._voxel_keys:
                self._voxel_keys.add(key)
                new_positions.append(unique_pos)
        return new_positions

    def _append_points(
        self,
        new_points: jax.Array,
        new_colors: jax.Array,
    ) -> jax.Array:
        """Append new representative points and aligned RGB colors.

        ``points_xyz`` and ``colors_rgb`` are only ever assigned together, so a
        ``None`` buffer marks the first add (the new arrays become the whole
        buffer); otherwise the new arrays are concatenated onto the existing ones.
        """
        points = jnp.asarray(new_points, dtype=jnp.bfloat16)
        colors = jnp.asarray(new_colors, dtype=jnp.uint8)
        if self.points_xyz is not None and self.colors_rgb is not None:
            points = jnp.concatenate([self.points_xyz, points], axis=0)
            colors = jnp.concatenate([self.colors_rgb, colors], axis=0)
        self.points_xyz = points
        self.colors_rgb = colors
        return points

    def _write_house_context_ply(self, output_dir: Path) -> Path:
        """Write a single-step colored PLY snapshot for the buffered context."""
        points_xyz, colors_rgb = self._host_point_color_arrays()
        step_id = 0
        ply_path = output_dir / f"step_{step_id:05d}_context.ply"
        with ply_path.open("w", encoding="utf-8") as ply_file:
            self._write_ply_header(ply_file, points_xyz.shape[0])
            for point_id, (point, color) in enumerate(
                zip(points_xyz, colors_rgb, strict=True)
            ):
                ply_file.write(
                    self._ply_vertex_line(
                        point_id,
                        point,
                        color,
                        self.NEW_STATUS_ID,
                        step_id,
                    )
                )
        return ply_path

    def _host_point_color_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return buffered points/colors as validated host arrays."""
        if self.points_xyz is None or self.colors_rgb is None:
            raise ValueError("cannot save empty house context")

        points_xyz, colors_rgb = jax.device_get((self.points_xyz, self.colors_rgb))
        points = np.asarray(points_xyz, dtype=np.float32)
        colors = np.asarray(colors_rgb, dtype=np.uint8)
        self._validate_point_color_shapes(points, colors)
        return points, colors

    @staticmethod
    def _validate_point_color_shapes(points: np.ndarray, colors: np.ndarray) -> None:
        """Validate point/color shape contract before writing PLY."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected points shape (P, 3), got {points.shape}")
        if colors.shape != points.shape:
            raise ValueError(
                f"expected colors shape {points.shape}, got {colors.shape}"
            )

    @classmethod
    def _write_ply_header(cls, ply_file: TextIO, vertex_count: int) -> None:
        """Write a PLY header compatible with CloudCompare scalar fields."""
        header_lines = [
            "ply",
            "format ascii 1.0",
            "comment color legend: 2=new orange",
            f"comment status_id {cls.NEW_STATUS_ID} {cls.NEW_STATUS_COMMENT}",
            f"element vertex {vertex_count}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int point_id",
            "property int status_id",
            "property int added_step",
            "end_header",
        ]
        ply_file.write("\n".join(header_lines))
        ply_file.write("\n")

    @staticmethod
    def _ply_vertex_line(
        point_id: int,
        point: np.ndarray,
        color: np.ndarray,
        status_id: int,
        added_step: int,
    ) -> str:
        """Format one PLY vertex row."""
        return (
            f"{float(point[0]):.8g} {float(point[1]):.8g} {float(point[2]):.8g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} "
            f"{point_id} {status_id} {added_step}\n"
        )
