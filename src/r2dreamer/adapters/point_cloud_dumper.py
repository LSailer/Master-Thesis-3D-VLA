"""Debug PLY point-cloud dump scheduling for a live VGGT extractor.

Hosts :class:`PointCloudDumper` and its structural extractor interface
:class:`PointCloudDumpingExtractor`, the diagnostics-only collaborator the
``VGGTHouseGlobalEmbeddingObsAdapter`` composes. Split out of
``hybrid_adapter.py``; ``hybrid_adapter`` re-exports these names for backward
compatibility.
"""

from __future__ import annotations

import os
from typing import Protocol


class PointCloudDumpingExtractor(Protocol):
    """Structural interface the dumper needs from a VGGT extractor."""

    def write_point_cloud_ply(self, path: str) -> None:
        """Write the extractor's current point cloud to ``path`` as PLY."""
        ...


class PointCloudDumper:
    """Schedules debug PLY point-cloud snapshots for a live VGGT extractor.

    Pure diagnostics concern extracted from
    ``VGGTHouseGlobalEmbeddingObsAdapter``: it decides *when* to dump
    (every N env steps, plus once at the end of the first episode) and calls
    ``extractor.write_point_cloud_ply`` when due. It never affects replay or
    agent-obs content.

    Args:
        extractor: Object exposing ``write_point_cloud_ply(path)``. Dumping is
            enabled only if this also exposes that attribute (duck-typed).
        dump_every: Dump every N env steps; ``0`` disables periodic dumps.
        dump_dir: Destination directory for PLY files; ``None`` disables
            dumping regardless of ``dump_every``.
    """

    def __init__(
        self,
        extractor: PointCloudDumpingExtractor,
        *,
        dump_every: int = 0,
        dump_dir: str | None = None,
    ):
        self._extractor = extractor
        self._dump_every = int(dump_every)
        self._dump_dir = dump_dir
        self._enabled = (
            self._dump_every > 0
            and self._dump_dir is not None
            and hasattr(extractor, "write_point_cloud_ply")
        )
        self._env_steps = 0
        self._episode_starts_seen = 0
        self._first_episode_dumped = False
        self._dump_count = 0

    @property
    def enabled(self) -> bool:
        """Whether periodic/end-of-episode dumping is active for this run."""
        return self._enabled

    @property
    def dump_count(self) -> int:
        """Number of PLY snapshots written so far."""
        return self._dump_count

    @property
    def env_steps(self) -> int:
        """Number of ``on_step`` calls observed so far."""
        return self._env_steps

    def _dump(self, label: str) -> None:
        """Write a PLY snapshot via the extractor if dumping is enabled.

        Args:
          label: Filename label inserted into ``pointcloud_<label>.ply``.
        """
        if not self._enabled:
            return
        os.makedirs(self._dump_dir, exist_ok=True)
        path = os.path.join(self._dump_dir, f"pointcloud_{label}.ply")
        self._extractor.write_point_cloud_ply(path)
        self._dump_count += 1

    def on_episode_start(self, is_first: bool) -> None:
        """Fire the one-time end-of-first-episode dump if ``is_first`` starts it.

        Args:
          is_first: Whether the current frame starts a new episode.
        """
        if not is_first:
            return
        self._episode_starts_seen += 1
        if self._episode_starts_seen == 2 and not self._first_episode_dumped:
            self._dump("end_of_first_episode")
            self._first_episode_dumped = True

    def on_step(self) -> None:
        """Advance the step counter and dump if this step is due."""
        self._env_steps += 1
        if self._enabled and self._env_steps % self._dump_every == 0:
            self._dump(f"step{self._env_steps}")
