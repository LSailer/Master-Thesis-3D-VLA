"""Typed interface for accumulating VGGT house-context points."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import ClassVar, TextIO

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from src.buffer.voxel_hash import VoxelContextConfig as _VoxelContextConfig
from src.buffer.voxel_hash import VoxelContextState as _VoxelContextState
from src.buffer.voxel_hash import add_frame_to_state as _add_frame_to_state
from src.buffer.voxel_hash import empty_state as _empty_state
from src.buffer.voxel_hash import house_context_snapshot as _house_context_snapshot
from src.buffer.voxel_hash import is_power_of_two as _is_power_of_two
from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGTExtractOutput


class _HouseContextBufferParams(BaseModel):
    """Validated construction parameters for ``HouseContextPoseBuffer``.

    Used internally at construction time only; the buffer's public constructor
    keeps its flat-kwarg signature and validates through this model on the host,
    never on a per-step hot path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence_score: float
    voxel_size_m: float = Field(gt=0.0)
    capacity: int = Field(gt=0)
    hash_table_size: int = Field(gt=0)


class HouseContextPoseBuffer:
    """Accumulate and save colored house-context points from VGGT outputs.

    Args:
        confidence_score: Minimum VGGT confidence score required for admitting
            points into the house context.
        scene_id: Scene identifier used for the output subdirectory.
        voxel_size_m: Edge length in metres of the voxel grid used to
            deduplicate accumulated points. Smaller keeps more detail.
        capacity: Maximum number of representative voxels stored in the backing
            point/color arrays. Extra new voxels are counted in ``overflow_count``.
        hash_table_size: Power-of-two number of slots in the exact occupancy
            table. It must be at least ``capacity``.

    Side effects:
        ``add`` mutates fixed-shape JAX device state. The public ``points_xyz``
        and ``colors_rgb`` properties expose the logical stored prefix and
        synchronize ``point_count`` when accessed.
    """

    NEW_STATUS_ID: ClassVar[int] = 2
    NEW_STATUS_COMMENT: ClassVar[str] = "newly_added_to_context"
    XYZ_CHANNELS: ClassVar[int] = 3
    RGB_CHANNELS: ClassVar[int] = 3
    HOUSE_POINT_CHANNELS: ClassVar[int] = 6
    DEFAULT_VOXEL_SIZE_M: ClassVar[float] = 0.01
    DEFAULT_CAPACITY: ClassVar[int] = 1 << 20
    DEFAULT_HASH_TABLE_SIZE: ClassVar[int] = 1 << 21
    DEFAULT_MAX_PROBE_COUNT: ClassVar[int] = 128

    def __init__(
        self,
        confidence_score: float,
        scene_id: str,
        voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
        *,
        capacity: int = DEFAULT_CAPACITY,
        hash_table_size: int = DEFAULT_HASH_TABLE_SIZE,
    ) -> None:
        params = self._validate_config(
            confidence_score, voxel_size_m, capacity, hash_table_size
        )
        self.confidence_score = params.confidence_score
        self.scene_id = scene_id
        self.voxel_size_m = params.voxel_size_m
        self.capacity = params.capacity
        self.hash_table_size = params.hash_table_size
        self._config = _VoxelContextConfig(
            voxel_size_m=self.voxel_size_m,
            confidence_score=self.confidence_score,
            hash_table_size=self.hash_table_size,
            capacity=self.capacity,
            max_probe_count=self.DEFAULT_MAX_PROBE_COUNT,
        )
        self._state = _empty_state(self.hash_table_size, self.capacity)

    @staticmethod
    def _validate_config(
        confidence_score: float,
        voxel_size_m: float,
        capacity: int,
        hash_table_size: int,
    ) -> _HouseContextBufferParams:
        """Validate fixed-state sizing and voxel quantization parameters.

        Args:
            confidence_score: Minimum VGGT confidence score for admission.
            voxel_size_m: Edge length in metres of the dedup voxel grid.
            capacity: Maximum number of representative voxels stored.
            hash_table_size: Power-of-two occupancy table slot count.

        Returns:
            The validated, frozen ``_HouseContextBufferParams`` model.

        Raises:
            ValueError: If any field fails its positivity/type constraint, or
                if ``hash_table_size`` is not a power of two or is smaller
                than ``capacity``.
        """
        try:
            params = _HouseContextBufferParams(
                confidence_score=confidence_score,
                voxel_size_m=voxel_size_m,
                capacity=capacity,
                hash_table_size=hash_table_size,
            )
        except PydanticValidationError as exc:
            raise ValueError(str(exc)) from exc
        if not _is_power_of_two(params.hash_table_size):
            raise ValueError(
                "hash_table_size must be a positive power of two, "
                f"got {params.hash_table_size}"
            )
        if params.hash_table_size < params.capacity:
            raise ValueError(
                "hash_table_size must be at least capacity "
                f"({params.hash_table_size} < {params.capacity})"
            )
        return params

    @property
    def points_xyz(self) -> jax.Array:
        """Logical ``(point_count, 3)`` bfloat16 XYZ prefix.

        Accessing this property synchronizes ``point_count`` to host. The rollout
        hot path should use ``add``/``house_context_array`` instead.
        """
        return self._state.store_xyz[: self.point_count]

    @property
    def colors_rgb(self) -> jax.Array:
        """Logical ``(point_count, 3)`` uint8 RGB prefix.

        Accessing this property synchronizes ``point_count`` to host. The rollout
        hot path should use ``add``/``house_context_array`` instead.
        """
        return self._state.store_rgb[: self.point_count]

    @property
    def point_count(self) -> int:
        """Logical stored point count as a host integer.

        Accessing this property synchronizes the scalar size to host. It is meant
        for diagnostics, saving, and tests, not for the rollout hot path.
        """
        return int(jax.device_get(self._state.size))

    @property
    def overflow_count(self) -> int:
        """Number of new voxels admitted after ``capacity`` was full."""
        return int(jax.device_get(self._state.overflow_count))

    @property
    def failed_insert_count(self) -> int:
        """Number of valid points that could not find a hash-table slot."""
        return int(jax.device_get(self._state.failed_insert_count))

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
            The fixed-capacity backing XYZ array with shape ``(capacity, 3)`` and
            bfloat16 dtype. The logical prefix length is ``point_count``.
        """
        flat_points, flat_rgb, confidence_flat = self._flatten_aligned_inputs(
            vggt_output,
            observation,
        )
        self._state = _add_frame_to_state(
            self._state,
            flat_points,
            flat_rgb,
            confidence_flat,
            self._config,
        )
        return self._state.store_xyz

    def seed_xyzrgb(self, xyzrgb: jax.Array) -> None:
        """Seed the buffer from ``(P, 6)`` XYZRGB rows with RGB in ``[0, 1]``."""
        seed = jnp.asarray(xyzrgb, dtype=jnp.float32)
        if seed.ndim != 2 or seed.shape[1] != self.HOUSE_POINT_CHANNELS:
            raise ValueError(f"expected seed shape (P, 6), got {seed.shape}")
        if seed.shape[0] == 0:
            return
        xyz = seed[:, : self.XYZ_CHANNELS]
        rgb01 = jnp.clip(seed[:, self.XYZ_CHANNELS :], 0.0, 1.0)
        rgb = jnp.rint(rgb01 * 255.0).astype(jnp.uint8)
        confidence = jnp.full((seed.shape[0],), self.confidence_score, dtype=jnp.float32)
        self._state = _add_frame_to_state(self._state, xyz, rgb, confidence, self._config)

    def house_context_array(
        self, max_points: int, dtype: jnp.dtype = jnp.float32
    ) -> tuple[jax.Array, jax.Array]:
        """Return a JIT-stable ``((max_points, 6) dtype, () int32)`` snapshot.

        The second element is the number of valid leading rows; rows beyond it
        are zero padding (see ``_house_context_snapshot``).
        """
        if max_points <= 0:
            raise ValueError(f"max_points must be positive, got {max_points}")
        return _house_context_snapshot(self._state, int(max_points), dtype)

    @staticmethod
    def resample_xyzrgb(xyzrgb: jax.Array, max_points: int) -> jax.Array:
        """Resample ``(P, 6)`` XYZRGB rows to fixed ``(max_points, 6)`` float32."""
        if max_points <= 0:
            raise ValueError(f"max_points must be positive, got {max_points}")
        rows = jnp.asarray(xyzrgb, dtype=jnp.float32)
        expected_channels = HouseContextPoseBuffer.HOUSE_POINT_CHANNELS
        if rows.ndim != 2 or rows.shape[1] != expected_channels:
            raise ValueError(f"expected xyzrgb shape (P, 6), got {rows.shape}")
        point_count = int(rows.shape[0])
        if point_count == 0:
            return jnp.zeros(
                (max_points, expected_channels),
                dtype=jnp.float32,
            )
        # Host int64 avoids the int32 overflow of ``arange * point_count``
        # (reached at point_count * max_points > 2**31); both operands are
        # static here so the exact indices transfer as a constant.
        stride_indices = (
            np.arange(max_points, dtype=np.int64) * point_count
        ) // max_points
        indices = jnp.asarray(
            np.minimum(stride_indices, point_count - 1), dtype=jnp.int32
        )
        return rows[indices].astype(jnp.float32)

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
        flat_rgb = jnp.asarray(rgb_hwc, dtype=jnp.uint8).reshape(-1, self.RGB_CHANNELS)
        confidence_flat = confidence_map.reshape(-1)
        if flat_points.shape[0] != flat_rgb.shape[0]:
            raise ValueError(
                "points/RGB pixel count mismatch: "
                f"{flat_points.shape[0]} != {flat_rgb.shape[0]}"
            )
        if flat_points.shape[0] != confidence_flat.shape[0]:
            raise ValueError(
                "points/confidence pixel count mismatch: "
                f"{flat_points.shape[0]} != {confidence_flat.shape[0]}"
            )
        return flat_points, flat_rgb, confidence_flat

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
        """Return logical buffered points/colors as validated host arrays."""
        size = self.point_count
        if size == 0:
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
