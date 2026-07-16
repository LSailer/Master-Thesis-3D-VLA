"""Typed interface for accumulating VGGT house-context points."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import ClassVar, NamedTuple, TextIO

import jax
import jax.numpy as jnp
import numpy as np

from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGTExtractOutput


@dataclass(frozen=True, slots=True)
class _VoxelContextConfig:
    """Static JAX config for the fixed-shape voxel context state."""

    voxel_size_m: float
    confidence_score: float
    hash_table_size: int
    capacity: int
    max_probe_count: int


# Registered as a pytree so the frame can cross the jit boundary as its
# three arrays (jit unflattening re-runs __post_init__ on tracers, which is
# fine: the row-alignment check only reads static shapes).
@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class _FlattenedFrame:
    """One flattened VGGT frame with row-aligned XYZ, RGB and confidence.

    Attributes:
      xyz: ``(P, 3)`` world points.
      rgb: ``(P, 3)`` colours.
      confidence: ``(P,)`` per-point VGGT confidence.
    """

    xyz: jax.Array
    rgb: jax.Array
    confidence: jax.Array

    def __post_init__(self) -> None:
        """Validates that ``rgb`` shares ``xyz``'s row count.

        ``rgb`` comes from the camera image while ``xyz`` comes from the VGGT
        point map, so nothing upstream guarantees they share a row count; this
        catches a point-map/image resolution mismatch. ``confidence`` always
        matches ``xyz`` by construction (``VGGTExtractOutput`` owns that
        invariant on the VGGT path; ``seed_xyzrgb`` builds both from one seed),
        so it is trusted from upstream rather than re-checked here.

        Raises:
          ValueError: If ``rgb`` has a different row count than ``xyz``.
        """
        if self.rgb.shape[0] != self.xyz.shape[0]:
            raise ValueError(
                "points/RGB pixel count mismatch: "
                f"{self.xyz.shape[0]} != {self.rgb.shape[0]}"
            )


class _VoxelContextState(NamedTuple):
    """Fixed-shape device state for exact voxel occupancy and point storage."""

    key_xyz: jax.Array  # (hash_table_size, 3) int32 voxel keys
    occupied: jax.Array  # (hash_table_size,) bool
    store_xyz: jax.Array  # (capacity, 3) bfloat16 representative points
    store_rgb: jax.Array  # (capacity, 3) uint8 representative colours
    size: jax.Array  # () int32 logical stored row count, capped at capacity
    overflow_count: jax.Array  # () int32 new voxels dropped after capacity fills
    failed_insert_count: jax.Array  # () int32 hash-table insert failures


class _UniqueFrameVoxels(NamedTuple):
    """Static-length unique voxel representatives for one flattened frame."""

    xyz: jax.Array  # (P, 3) float32 representative points, sorted by voxel key
    rgb: jax.Array  # (P, 3) uint8 representative colours
    key_xyz: jax.Array  # (P, 3) int32 voxel keys
    active: jax.Array  # (P,) bool, true only for valid first representatives


class _ProbeLoopState(NamedTuple):
    """Carry for vectorized open-addressing probe rounds."""

    probe_index: jax.Array  # () int32
    voxel_state: _VoxelContextState
    active: jax.Array  # (P,) bool keys not inserted/found yet


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _hash_voxel_keys(keys_xyz: jax.Array, hash_table_size: int) -> jax.Array:
    """Hash int32 ``(..., 3)`` voxel keys into a power-of-two table."""
    keys_u32 = keys_xyz.astype(jnp.uint32)
    hashed = (
        keys_u32[..., 0] * jnp.uint32(73_856_093)
        ^ keys_u32[..., 1] * jnp.uint32(19_349_663)
        ^ keys_u32[..., 2] * jnp.uint32(83_492_791)
    )
    return (hashed & jnp.uint32(hash_table_size - 1)).astype(jnp.int32)


def _quantize_points(
    flat_xyz: jax.Array, voxel_size_m: float
) -> tuple[jax.Array, jax.Array]:
    """Return finite mask and int32 voxel keys for ``(P, 3)`` XYZ points."""
    finite_xyz = jnp.isfinite(flat_xyz).all(axis=1)
    safe_xyz = jnp.where(jnp.isfinite(flat_xyz), flat_xyz, jnp.float32(0.0))
    voxel_keys = jnp.floor(safe_xyz / voxel_size_m).astype(jnp.int32)
    return finite_xyz, voxel_keys


def _unique_frame_voxels(
    flat_xyz: jax.Array,
    flat_rgb: jax.Array,
    valid: jax.Array,
    voxel_keys: jax.Array,
) -> _UniqueFrameVoxels:
    """Sort rows by voxel key and keep the first valid input row per voxel."""
    # Validity is the most-significant sort key, so invalid rows sort last.
    # lexsort is stable, so equal keys keep input order and the first valid
    # input row per voxel wins without an explicit tie-break key.
    sort_order = jnp.lexsort(
        (voxel_keys[:, 2], voxel_keys[:, 1], voxel_keys[:, 0], ~valid)
    )
    sorted_keys = voxel_keys[sort_order]
    sorted_valid = valid[sort_order]
    same_as_previous = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.bool_),
            jnp.all(sorted_keys[1:] == sorted_keys[:-1], axis=1),
        ]
    )
    active = sorted_valid & ~same_as_previous
    return _UniqueFrameVoxels(
        xyz=flat_xyz[sort_order],
        rgb=flat_rgb[sort_order],
        key_xyz=sorted_keys,
        active=active,
    )


def _store_winning_voxels(
    state: _VoxelContextState,
    frame: _UniqueFrameVoxels,
    slots: jax.Array,
    wins: jax.Array,
    config: _VoxelContextConfig,
) -> _VoxelContextState:
    """Commit this probe round's winning new voxels to table and store."""
    table_slots = jnp.where(wins, slots, config.hash_table_size)
    offsets = jnp.cumsum(wins.astype(jnp.int32)) - jnp.int32(1)
    destinations = state.size + offsets
    can_store = wins & (destinations < config.capacity)
    store_slots = jnp.where(can_store, destinations, config.capacity)
    return _VoxelContextState(
        key_xyz=state.key_xyz.at[table_slots].set(frame.key_xyz, mode="drop"),
        occupied=state.occupied.at[table_slots].set(True, mode="drop"),
        store_xyz=state.store_xyz.at[store_slots].set(
            frame.xyz.astype(jnp.bfloat16), mode="drop"
        ),
        store_rgb=state.store_rgb.at[store_slots].set(
            frame.rgb.astype(jnp.uint8), mode="drop"
        ),
        size=jnp.minimum(state.size + jnp.sum(wins), config.capacity).astype(jnp.int32),
        overflow_count=state.overflow_count + jnp.sum(wins & ~can_store),
        failed_insert_count=state.failed_insert_count,
    )


def _probe_round(
    loop_state: _ProbeLoopState,
    frame: _UniqueFrameVoxels,
    first_slot: jax.Array,
    config: _VoxelContextConfig,
) -> _ProbeLoopState:
    """Run one vectorized linear-probing round for all active frame keys."""
    table_mask = jnp.int32(config.hash_table_size - 1)
    slots = (first_slot + loop_state.probe_index) & table_mask
    slot_occupied = loop_state.voxel_state.occupied[slots]
    same_key = (
        loop_state.active
        & slot_occupied
        & jnp.all(loop_state.voxel_state.key_xyz[slots] == frame.key_xyz, axis=1)
    )
    empty_candidate = loop_state.active & ~slot_occupied
    inactive_order = jnp.int32(frame.active.shape[0])
    contender_order = jnp.where(
        empty_candidate,
        jnp.arange(frame.active.shape[0], dtype=jnp.int32),
        inactive_order,
    )
    winner_by_slot = jnp.full(
        (config.hash_table_size,), inactive_order, dtype=jnp.int32
    ).at[slots].min(contender_order)
    wins = empty_candidate & (winner_by_slot[slots] == contender_order)
    voxel_state = _store_winning_voxels(
        loop_state.voxel_state,
        frame,
        slots,
        wins,
        config,
    )
    return _ProbeLoopState(
        probe_index=loop_state.probe_index + jnp.int32(1),
        voxel_state=voxel_state,
        active=loop_state.active & ~(same_key | wins),
    )


def _insert_unique_voxels(
    state: _VoxelContextState,
    frame: _UniqueFrameVoxels,
    config: _VoxelContextConfig,
) -> _VoxelContextState:
    """Insert sorted unique frame voxels with bounded vectorized probing."""
    first_slot = _hash_voxel_keys(frame.key_xyz, config.hash_table_size)
    initial = _ProbeLoopState(
        probe_index=jnp.asarray(0, dtype=jnp.int32),
        voxel_state=state,
        active=frame.active,
    )

    def should_probe(loop_state: _ProbeLoopState) -> jax.Array:
        return (loop_state.probe_index < config.max_probe_count) & jnp.any(
            loop_state.active
        )

    def probe_once(loop_state: _ProbeLoopState) -> _ProbeLoopState:
        return _probe_round(loop_state, frame, first_slot, config)

    result = jax.lax.while_loop(should_probe, probe_once, initial)
    return result.voxel_state._replace(
        failed_insert_count=result.voxel_state.failed_insert_count
        + jnp.sum(result.active.astype(jnp.int32))
    )


@functools.partial(jax.jit, static_argnums=(2,), donate_argnums=(0,))
def _add_frame_to_state(
    state: _VoxelContextState,
    frame: _FlattenedFrame,
    config: _VoxelContextConfig,
) -> _VoxelContextState:
    """Fold one flattened frame into the voxel context state.

    The kernel normalizes dtypes itself, so frames may carry any convertible
    arrays.

    Per-frame representatives are selected by a static lexicographic sort of
    int32 voxel keys. Cross-frame novelty is resolved by exact key comparison
    in a vectorized open-addressed table; hash collisions only add probe rounds.

    Args:
      state: Current voxel context state to extend; donated to the kernel.
      frame: Row-aligned XYZ/RGB/confidence frame to fold in.
      config: Static voxel context configuration.

    Returns:
      The updated voxel context state.
    """
    flat_xyz = jnp.asarray(frame.xyz, dtype=jnp.float32)
    flat_rgb = jnp.asarray(frame.rgb, dtype=jnp.uint8)
    confidence = jnp.asarray(frame.confidence, dtype=jnp.float32)
    finite_xyz, voxel_keys = _quantize_points(flat_xyz, config.voxel_size_m)
    valid = (
        finite_xyz
        & jnp.isfinite(confidence)
        & (confidence >= config.confidence_score)
    )
    unique = _unique_frame_voxels(flat_xyz, flat_rgb, valid, voxel_keys)
    return _insert_unique_voxels(state, unique, config)


@functools.partial(jax.jit, static_argnums=(1, 2))
def _house_context_snapshot(
    state: _VoxelContextState,
    max_points: int,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(max_points, 6)`` XYZRGB rows in ``dtype`` plus the valid count.

    Rows ``[0, count)`` carry stored points and rows beyond are zeros, so
    consumers can mask padding exactly (masked pooling in the encoder). While
    more voxels are stored than ``max_points``, an even stride subsamples them
    and ``count == max_points``; below that the stored prefix is zero-padded.
    """
    safe_size = jnp.maximum(state.size, jnp.int32(1))
    # int32 ``arange * size`` overflows once size exceeds 2**31 / max_points
    # (~524k stored voxels at max_points=4096). float32 keeps the stride math
    # exact enough: size <= capacity <= 2**24 is exactly representable, the
    # per-index error stays below one row, and floor preserves monotonicity.
    stride_ratio = safe_size.astype(jnp.float32) / jnp.float32(max_points)
    strided = jnp.floor(
        jnp.arange(max_points, dtype=jnp.float32) * stride_ratio
    ).astype(jnp.int32)
    rows = jnp.arange(max_points, dtype=jnp.int32)
    indices = jnp.where(state.size > max_points, strided, rows)
    indices = jnp.clip(indices, jnp.int32(0), safe_size - jnp.int32(1))
    xyz = state.store_xyz[indices].astype(dtype)
    rgb = state.store_rgb[indices].astype(dtype) / jnp.asarray(255.0, dtype)
    snapshot = jnp.concatenate([xyz, rgb], axis=1)
    count = jnp.minimum(state.size, jnp.int32(max_points))
    return jnp.where(rows[:, None] < count, snapshot, jnp.asarray(0.0, dtype)), count


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
        self._validate_config(voxel_size_m, capacity, hash_table_size)
        self.confidence_score = float(confidence_score)
        self.scene_id = scene_id
        self.voxel_size_m = float(voxel_size_m)
        self.capacity = int(capacity)
        self.hash_table_size = int(hash_table_size)
        self._config = _VoxelContextConfig(
            voxel_size_m=self.voxel_size_m,
            confidence_score=self.confidence_score,
            hash_table_size=self.hash_table_size,
            capacity=self.capacity,
            max_probe_count=self.DEFAULT_MAX_PROBE_COUNT,
        )
        self._state = _VoxelContextState(
            key_xyz=jnp.zeros((self.hash_table_size, 3), dtype=jnp.int32),
            occupied=jnp.zeros((self.hash_table_size,), dtype=jnp.bool_),
            store_xyz=jnp.zeros((self.capacity, 3), dtype=jnp.bfloat16),
            store_rgb=jnp.zeros((self.capacity, 3), dtype=jnp.uint8),
            size=jnp.asarray(0, dtype=jnp.int32),
            overflow_count=jnp.asarray(0, dtype=jnp.int32),
            failed_insert_count=jnp.asarray(0, dtype=jnp.int32),
        )

    @staticmethod
    def _validate_config(
        voxel_size_m: float,
        capacity: int,
        hash_table_size: int,
    ) -> None:
        """Validate fixed-state sizing and voxel quantization parameters."""
        if voxel_size_m <= 0.0:
            raise ValueError(f"voxel_size_m must be positive, got {voxel_size_m}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not _is_power_of_two(hash_table_size):
            raise ValueError(
                f"hash_table_size must be a positive power of two, got {hash_table_size}"
            )
        if hash_table_size < capacity:
            raise ValueError(
                "hash_table_size must be at least capacity "
                f"({hash_table_size} < {capacity})"
            )

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
        frame = self._flatten_aligned_inputs(vggt_output, observation)
        self._state = _add_frame_to_state(self._state, frame, self._config)
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
        frame = _FlattenedFrame(xyz=xyz, rgb=rgb, confidence=confidence)
        self._state = _add_frame_to_state(self._state, frame, self._config)

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
    ) -> _FlattenedFrame:
        """Return one row-aligned flattened frame from a VGGT output and image.

        ``_FlattenedFrame.__post_init__`` documents and enforces the
        alignment contract.
        """
        rgb_hwc = np.moveaxis(observation.image, 0, -1)
        return _FlattenedFrame(
            xyz=vggt_output.world_points.reshape(-1, self.XYZ_CHANNELS),
            rgb=jnp.asarray(rgb_hwc, dtype=jnp.uint8).reshape(-1, self.RGB_CHANNELS),
            confidence=vggt_output.confidence.reshape(-1),
        )

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
