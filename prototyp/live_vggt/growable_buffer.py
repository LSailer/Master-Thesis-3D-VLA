"""Doubling-capacity wrapper around ``HouseContextPoseBuffer``.

``HouseContextPoseBuffer`` keeps fixed-shape device state, so its jitted
add-kernel compiles once per ``capacity`` value. Picking one huge fixed
capacity up front avoids recompiles but overcommits memory; letting the
store grow per frame would recompile every step. This wrapper takes the
dynamic-array middle ground: start at a small power-of-two capacity and
double it whenever a frame overflows, so a run of N stored voxels pays at
most ``log2(N / initial_capacity)`` extra compiles while memory stays
proportional to what is actually stored.

Growth is loss-free: the old stored prefix is re-seeded into the doubled
buffer and the overflowing frame is re-added. Re-adding is safe because
voxel insertion is first-writer-wins — voxels that already made it in keep
their representatives and only the previously dropped ones retry.

Caveat: the backing store holds bfloat16 representatives, so re-seeding
re-quantizes *rounded* positions. Where bfloat16 spacing approaches the
voxel size (|coordinate| / 128 vs ``voxel_size_m``), neighbouring voxels
can merge on growth and re-added frames can leave near-duplicate
representatives one voxel apart. At house scale (|xyz| ~ 10 m, 1 cm
voxels) the displacement is ~3 cm — irrelevant for visualization, but do
not treat grown buffers as exact voxel sets.
"""

from __future__ import annotations

import jax.numpy as jnp

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGTExtractOutput


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


class GrowableHouseContextBuffer:
    """House-context point store whose capacity doubles on overflow.

    Attributes:
        growth_events: Capacities adopted by past growth steps, in order
            (empty while the initial capacity has never overflowed).
    """

    def __init__(
        self,
        confidence_score: float,
        scene_id: str,
        voxel_size_m: float = HouseContextPoseBuffer.DEFAULT_VOXEL_SIZE_M,
        *,
        initial_capacity: int = 1 << 17,
        max_capacity: int = 1 << 24,
    ) -> None:
        """Creates the initial fixed-shape backing buffer.

        Args:
            confidence_score: Minimum VGGT confidence for admitting points.
            scene_id: Scene identifier used for the save subdirectory.
            voxel_size_m: Dedup voxel edge length in metres.
            initial_capacity: Starting voxel capacity; must be a power of
                two so the doubled hash-table size stays a power of two.
            max_capacity: Hard ceiling for growth; ``add`` raises once a
                frame would need more than this.

        Raises:
            ValueError: If ``initial_capacity`` is not a power of two or
                exceeds ``max_capacity``.
        """
        if not _is_power_of_two(initial_capacity):
            raise ValueError(
                f"initial_capacity must be a power of two, got {initial_capacity}"
            )
        if initial_capacity > max_capacity:
            raise ValueError(
                f"initial_capacity {initial_capacity} exceeds max_capacity {max_capacity}"
            )
        self.max_capacity = int(max_capacity)
        self.growth_events: list[int] = []
        self._buffer = self._make_buffer(
            confidence_score, scene_id, voxel_size_m, int(initial_capacity)
        )

    @staticmethod
    def _make_buffer(
        confidence_score: float,
        scene_id: str,
        voxel_size_m: float,
        capacity: int,
    ) -> HouseContextPoseBuffer:
        """Builds a backing buffer with a 2x-capacity hash table."""
        return HouseContextPoseBuffer(
            confidence_score=confidence_score,
            scene_id=scene_id,
            voxel_size_m=voxel_size_m,
            capacity=capacity,
            hash_table_size=2 * capacity,
        )

    @property
    def capacity(self) -> int:
        """Current backing capacity in voxels."""
        return self._buffer.capacity

    @property
    def point_count(self) -> int:
        """Logical stored point count (host-synchronizing)."""
        return self._buffer.point_count

    @property
    def points_xyz(self) -> jnp.ndarray:
        """Logical ``(point_count, 3)`` bfloat16 XYZ prefix."""
        return self._buffer.points_xyz

    @property
    def colors_rgb(self) -> jnp.ndarray:
        """Logical ``(point_count, 3)`` uint8 RGB prefix."""
        return self._buffer.colors_rgb

    def add(
        self,
        vggt_output: VGGTExtractOutput,
        observation: ObservationFrame,
    ) -> int:
        """Adds one frame, doubling capacity until it fits without loss.

        Args:
            vggt_output: Structured VGGT output for the source observation.
            observation: Environment frame that produced ``vggt_output``.

        Returns:
            The number of capacity doublings this call triggered.

        Raises:
            RuntimeError: If fitting the frame would exceed ``max_capacity``.
        """
        self._buffer.add(vggt_output, observation)
        doublings = 0
        # Reading the counters synchronizes two scalars to host per frame;
        # acceptable next to a ~100 ms VGGT extract.
        while (
            self._buffer.overflow_count > 0 or self._buffer.failed_insert_count > 0
        ):
            self._grow()
            self._buffer.add(vggt_output, observation)
            doublings += 1
        return doublings

    def _grow(self) -> None:
        """Doubles capacity and re-seeds the stored prefix into the new buffer.

        Raises:
            RuntimeError: If the doubled capacity would exceed ``max_capacity``.
        """
        old = self._buffer
        new_capacity = old.capacity * 2
        if new_capacity > self.max_capacity:
            raise RuntimeError(
                f"house context needs more than max_capacity={self.max_capacity} "
                f"voxels (current capacity {old.capacity}, "
                f"overflow {old.overflow_count}, "
                f"failed inserts {old.failed_insert_count})"
            )
        grown = self._make_buffer(
            old.confidence_score, old.scene_id, old.voxel_size_m, new_capacity
        )
        xyz = old.points_xyz.astype(jnp.float32)
        rgb01 = old.colors_rgb.astype(jnp.float32) / 255.0
        grown.seed_xyzrgb(jnp.concatenate([xyz, rgb01], axis=1))
        self._buffer = grown
        self.growth_events.append(new_capacity)

    def save(self, output_path) -> object:
        """Saves the accumulated context as a binary PLY (see backing buffer).

        Args:
            output_path: Destination root directory.

        Returns:
            The scene directory that was written.
        """
        return self._buffer.save(output_path)
