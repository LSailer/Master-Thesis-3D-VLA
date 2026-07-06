"""Growth-history and end-of-run stats for live house-context scene buffers.

Hosts :class:`HouseBufferDiagnostics`, the diagnostics collaborator the
``VGGTHousePointsPoseObsAdapter`` composes over its
:class:`~src.r2dreamer.adapters.scene_buffer.SceneBufferManager`. Split out of
``hybrid_adapter.py``; ``hybrid_adapter`` re-exports this name for backward
compatibility.
"""

from __future__ import annotations

from src.r2dreamer.adapters.scene_buffer import SceneBufferManager


class HouseBufferDiagnostics:
    """Collects growth-history samples and end-of-run stats for scene buffers.

    Args:
        scene_buffers: The :class:`SceneBufferManager` whose buffers are
            summarized.
    """

    def __init__(self, scene_buffers: SceneBufferManager):
        self._scene_buffers = scene_buffers
        self._env_steps = 0
        self._growth_history: list[tuple[int, int]] = []

    def record_step(self) -> None:
        """Sample total stored points at log-spaced (doubling) env steps.

        The host sync happens at steps 1, 2, 4, 8, ... only, so over a 2M-step
        run the growth curve costs ~21 scalar reads in total.
        """
        self._env_steps += 1
        if self._env_steps & (self._env_steps - 1):
            return  # sample only at powers of two
        total = self._scene_buffers.total_point_count()
        self._growth_history.append((self._env_steps, total))

    @property
    def growth_history(self) -> list[tuple[int, int]]:
        """``(env_step, total_points)`` samples at doubling env steps."""
        return list(self._growth_history)

    def diagnostics(self) -> dict[str, float]:
        """Per-scene house-buffer usage; syncs one scalar per buffer to host."""
        stats: dict[str, float] = {}
        buffers = self._scene_buffers.buffers
        total_points = 0
        total_overflow = 0
        total_failed = 0
        max_fill = 0.0
        for buffer in buffers.values():
            points = buffer.point_count
            total_points += points
            total_overflow += buffer.overflow_count
            total_failed += buffer.failed_insert_count
            max_fill = max(max_fill, points / buffer.capacity)
        if buffers:
            stats["house_buffer/scenes"] = float(len(buffers))
            stats["house_buffer/total_points"] = float(total_points)
            stats["house_buffer/max_fill_fraction"] = max_fill
            stats["house_buffer/overflow_count"] = float(total_overflow)
            stats["house_buffer/failed_insert_count"] = float(total_failed)
        return stats
