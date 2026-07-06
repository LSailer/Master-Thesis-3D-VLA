"""Per-scene house-context point-buffer lifecycle for the live house adapters.

Hosts the collaborators the ``VGGTHousePointsPoseObsAdapter`` composes to own
its per-``scene_id`` :class:`~src.buffer.house_context_pose_buffer.HouseContextPoseBuffer`
buffers: the structural buffer interface, the default buffer factory, and the
:class:`SceneBufferManager` that creates/seeds/looks-up one buffer per scene.
Split out of ``hybrid_adapter.py`` so the adapter file holds only adapter
classes; ``hybrid_adapter`` re-exports these names for backward compatibility.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import jax.numpy as jnp

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.environments.observation import ObservationFrame


class HouseContextPoseBufferLike(Protocol):
    """Structural interface the scene-buffer manager needs from a point buffer.

    Matches the subset of :class:`~src.buffer.house_context_pose_buffer.HouseContextPoseBuffer`
    that :class:`SceneBufferManager` and ``HouseBufferDiagnostics`` call,
    so a test double or an alternative buffer implementation can stand in for
    it without subclassing.
    """

    capacity: int

    def seed_xyzrgb(self, xyzrgb: jnp.ndarray) -> None:
        """Seed the buffer from ``(P, 6)`` XYZRGB rows with RGB in ``[0, 1]``."""
        ...

    def add(self, vggt_output: Any, observation: ObservationFrame) -> jnp.ndarray:
        """Add one VGGT output and its source observation to the buffer."""
        ...

    def house_context_array(
        self, max_points: int, dtype: jnp.dtype = jnp.float32
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return a JIT-stable ``((max_points, 6) dtype, () int32)`` snapshot."""
        ...

    @property
    def point_count(self) -> int:
        """Logical stored point count as a host integer."""
        ...

    @property
    def overflow_count(self) -> int:
        """Number of new voxels admitted after ``capacity`` was full."""
        ...

    @property
    def failed_insert_count(self) -> int:
        """Number of valid points that could not find a hash-table slot."""
        ...


BufferFactory = Callable[[str], HouseContextPoseBufferLike]


def default_house_context_pose_buffer_factory(
    *,
    confidence_score: float,
    voxel_size_m: float,
    capacity: int,
    hash_table_size: int,
) -> BufferFactory:
    """Build the default ``HouseContextPoseBuffer`` factory for one adapter.

    Args:
      confidence_score: Minimum VGGT confidence required to admit a point.
      voxel_size_m: Voxel edge length in metres used for deduplication.
      capacity: Maximum number of representative voxels stored per scene.
      hash_table_size: Power-of-two occupancy-table size (>= ``capacity``).

    Returns:
      A callable ``scene_id -> HouseContextPoseBuffer`` that constructs one
      fresh buffer per call, configured identically for every scene.
    """

    def factory(scene_id: str) -> HouseContextPoseBufferLike:
        return HouseContextPoseBuffer(
            confidence_score=confidence_score,
            scene_id=scene_id,
            voxel_size_m=voxel_size_m,
            capacity=capacity,
            hash_table_size=hash_table_size,
        )

    return factory


class SceneBufferManager:
    """Owns the per-scene :class:`HouseContextPoseBuffer` lifecycle.

    Creates one buffer per ``scene_id`` on first use (optionally warm-started
    from a shared static-PLY seed), and hands back the same buffer on later
    lookups for that scene — mirroring
    ``ScenePointCloudTracker.point_clouds`` in the ``live_vggt`` prototype.

    Args:
        buffer_factory: Callable ``scene_id -> HouseContextPoseBufferLike``
            used to construct a fresh buffer the first time a scene is seen.
            Defaults to :func:`default_house_context_pose_buffer_factory` with
            the module's fixed capacity/hash-table-size constants (injected by
            the adapter), so callers only need to override this for tests or
            alternative buffer backends.
        seed_xyzrgb: Optional ``(M, 6)`` XYZRGB warm-start seed (RGB in
            ``[0, 1]``) applied to every newly created scene buffer.
    """

    def __init__(
        self,
        buffer_factory: BufferFactory,
        *,
        seed_xyzrgb: jnp.ndarray | None = None,
    ):
        self._buffer_factory = buffer_factory
        self._seed_xyzrgb = seed_xyzrgb
        self._buffers: dict[str, HouseContextPoseBufferLike] = {}

    @property
    def buffers(self) -> Mapping[str, HouseContextPoseBufferLike]:
        """Read-only view of the live per-scene buffers, keyed by scene id."""
        return self._buffers

    def get_or_create(self, scene_id: str) -> HouseContextPoseBufferLike:
        """Return the buffer for ``scene_id``, creating and seeding it once.

        Args:
          scene_id: Scene identifier from the observation frame; falls back to
            ``"scene"`` when falsy.

        Returns:
          The (possibly newly created) buffer for this scene.
        """
        key = scene_id or "scene"
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = self._buffer_factory(key)
            if self._seed_xyzrgb is not None:
                buffer.seed_xyzrgb(self._seed_xyzrgb)
            self._buffers[key] = buffer
        return buffer

    def total_point_count(self) -> int:
        """Sum ``point_count`` across all live scene buffers (host sync)."""
        return sum(buffer.point_count for buffer in self._buffers.values())
