"""``house_voxels``: live per-scene voxel house map, pose, RGB frame."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp

from src.adapters.contract import (
    AdapterField,
    AdapterOutput,
    Encoder,
    FeatureExtractor,
)
from src.adapters.replay_image import replay_image
from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGT_IMAGE_SIZE, ResetMode

# Subdirectory of the run directory the PLY snapshots land in. Fixed rather than
# configurable: the map belongs to one run, so its snapshots belong next to that
# run's checkpoints and metrics, and the visualization guide can name one path.
DUMP_SUBDIR = "pointcloud_dumps"


def _dump_steps(steps: str | Sequence[int] | None) -> frozenset[int]:
    """Normalize a dump schedule to a set of adapter-step numbers.

    Args:
        steps: Either the CLI form (comma-separated step numbers, because the
            SLURM launcher renders scalars only) or an iterable of ints.

    Returns:
        The steps at which to snapshot; empty means dumping is off.
    """
    if steps is None:
        return frozenset()
    if isinstance(steps, str):
        return frozenset(int(part) for part in steps.split(",") if part.strip())
    return frozenset(int(step) for step in steps)


class HouseVoxelsAdapter:
    """Accumulates VGGT world points per scene and emits a fixed-size cloud.

    Three routed fields: the 64x64 frame to a conv branch, the camera pose to an
    MLP branch, and the accumulated house map to a cloud branch. The cloud is the
    one live field - a global context, not a per-step observation - so replay
    stores only its latest value.

    Points are deduplicated on device by an exact voxel hash
    (:class:`HouseContextPoseBuffer`), one buffer per ``scene_id``, so episodes
    of the same house keep extending one map. The full 518x518 point map is fed
    in every step; voxel dedup is a fixed-shape JIT graph (~2 ms/frame on H100
    regardless of stored size), so no input subsampling is needed.

    What the map looks like is only checkable by eye, so an optional schedule
    (``pointcloud_dump_steps``) writes it to disk as PLY at chosen steps and at
    the end of the first episode - the drift and ghost-copy failures this
    pipeline has had were all visible in a viewer long before any metric moved.

    Single-scene contract: the emitted cloud is the *current* scene's, while a
    sampled replay batch may hold camera poses from earlier scenes. Exact for
    the single-house L1 curriculum, an approximation above it - the same one the
    model's global-house-context design has always made.
    """

    RENDER_RESOLUTION = VGGT_IMAGE_SIZE
    NEEDS_FEATURES = True
    # PERSIST_SCENE keeps the VGGT attention stream per scene instead of
    # re-anchoring every episode, so all episodes of one house share a world
    # frame - the same frame the voxel buffer below accumulates in.
    EXTRACTOR_KWARGS: dict[str, object] = {"reset_mode": ResetMode.PERSIST_SCENE}
    ENCODER_OVERRIDES: dict[str, object] = {}
    # The accumulated map is this variant's own artifact, so its snapshot
    # schedule is a knob only this variant can consume.
    RUN_FLAGS: tuple[str, ...] = ("pointcloud_dump_steps", "output_dir")

    # Which branch the cloud goes to; the GNN arm below overrides it.
    CLOUD_ENCODER = Encoder.POINTNET

    # Cloud rows handed to the encoder. Fixed, so ``train_step``/``act`` never
    # recompile as the map grows, and every row is a real point: below this many
    # stored voxels the valid rows repeat rather than pad with zeros, which is
    # what removes the need for a separate valid-count field and masked pooling.
    HOUSE_POINTS = 16384
    VOXEL_SIZE_M = 0.01
    CONFIDENCE_SCORE = 1.5
    # 1 cm voxels on full-res input reach ~210k points in 50 steps on one L1
    # scene; a whole house is bounded by surface area (~500 m^2 -> ~5M voxels),
    # so 2^23 (~293 MB device memory) covers any realistic scene outright.
    BUFFER_CAPACITY = 1 << 23
    BUFFER_HASH_TABLE_SIZE = 1 << 24

    def __init__(
        self,
        extractor: FeatureExtractor,
        *,
        pointcloud_dump_steps: str | Sequence[int] | None = None,
        output_dir: str | None = None,
    ) -> None:
        """Bind the extractor and configure the diagnostic dump.

        The accumulator's sizing is class constants, not arguments: an arm that
        maps at a different resolution is a subclass overriding them (as the GNN
        arm overrides ``CLOUD_ENCODER``), so the registry keeps naming variants
        rather than construction recipes.

        Args:
            extractor: Frozen extractor supplying world points and pose.
            pointcloud_dump_steps: Adapter steps at which to snapshot every
                non-empty scene map as a PLY, either comma-separated (the CLI
                form) or as ints. ``None`` disables dumping entirely.
            output_dir: Run directory the snapshots are written below (in
                ``pointcloud_dumps/``). Only read when a schedule is set.

        Raises:
            ValueError: If a schedule is set without a run directory to write
                into - a diagnostic that silently never runs is worse than a
                run that refuses to start.
        """
        self._extractor = extractor
        self._buffers: dict[str, HouseContextPoseBuffer] = {}
        self._dump_steps = _dump_steps(pointcloud_dump_steps)
        if self._dump_steps and output_dir is None:
            raise ValueError(
                "pointcloud_dump_steps needs a run directory to write into; "
                "pass output_dir (the train CLI's --output_dir)"
            )
        self._dump_root = (
            Path(output_dir) / DUMP_SUBDIR
            if self._dump_steps and output_dir is not None
            else None
        )
        self._steps = 0
        self._first_episode_dumped = False

    def _buffer_for(self, scene_id: str) -> HouseContextPoseBuffer:
        """Return this scene's voxel buffer, creating it on first sight."""
        buffer = self._buffers.get(scene_id)
        if buffer is None:
            buffer = HouseContextPoseBuffer(
                confidence_score=self.CONFIDENCE_SCORE,
                scene_id=scene_id,
                voxel_size_m=self.VOXEL_SIZE_M,
                capacity=self.BUFFER_CAPACITY,
                hash_table_size=self.BUFFER_HASH_TABLE_SIZE,
            )
            self._buffers[scene_id] = buffer
        return buffer

    def _fixed_cloud(self, buffer: HouseContextPoseBuffer) -> jnp.ndarray:
        """Return ``(house_points, 6)`` rows that are all real points.

        The buffer's snapshot strides over stored voxels when there are more
        than requested and zero-pads when there are fewer; the modulo gather
        below replaces that padding with a cycle over the valid rows, so the
        cloud branch needs no validity mask.

        float16 rather than the repo-default bfloat16: these are metric world
        coordinates, and bfloat16's 8-bit mantissa resolves ~3 cm at 10 m - three
        times coarser than the 1 cm voxel grid being stored. The cloud branch
        re-centers in float32 before casting to its compute dtype.
        """
        snapshot, count = buffer.house_context_array(self.HOUSE_POINTS, jnp.float16)
        indices = jnp.arange(self.HOUSE_POINTS) % jnp.maximum(count, 1)
        return snapshot[indices]

    def _dump(self, label: str) -> None:
        """Save every non-empty scene map under ``<run>/pointcloud_dumps/<label>/``.

        One file per scene, so a multi-scene run stays inspectable. Reading
        ``point_count`` synchronizes a device scalar, which is why this is
        reached only on the (few) scheduled steps.

        Args:
            label: Snapshot subdirectory name, e.g. ``step_000002000``.
        """
        if self._dump_root is None:
            return
        for buffer in self._buffers.values():
            if buffer.point_count > 0:
                buffer.save(self._dump_root / label)

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        """Accumulate this frame's points and route the three fields."""
        features = self._extractor.extract(frame)
        buffer = self._buffer_for(frame.scene_id)
        buffer.add(features, frame)
        self._steps += 1
        if self._steps in self._dump_steps:
            self._dump(f"step_{self._steps:09d}")
        # Keyed off the observed episode end, not a reset count: composition and
        # prefill each reset the collector before any exploration happens, so
        # counting resets would fire this snapshot on the very first frames and
        # label a near-empty map "one episode".
        if frame.is_episode_end and not self._first_episode_dumped:
            self._first_episode_dumped = True
            self._dump("end_of_first_episode")
        return [
            AdapterField(
                key="image",
                encoder=Encoder.CONV,
                buffer=True,
                value=replay_image(frame.image),
                decoder_target=True,
            ),
            AdapterField(
                key="camera_pose",
                encoder=Encoder.MLP,
                buffer=True,
                # Metric pose: float32, same reasoning as the cloud's float16.
                value=jnp.ravel(features.camera_pose).astype(jnp.float32),
            ),
            AdapterField(
                key="house_context",
                encoder=self.CLOUD_ENCODER,
                buffer=False,
                value=self._fixed_cloud(buffer),
            ),
        ]

    def diagnostics(self) -> dict[str, float]:
        """Return end-of-run voxel-buffer health, aggregated over scenes.

        Synchronizes device scalars to host, so call it once when the run ends.
        """
        if not self._buffers:
            return {}
        counts = [b.point_count for b in self._buffers.values()]
        capacity = max(b.capacity for b in self._buffers.values())
        return {
            "house_buffer/scenes": float(len(self._buffers)),
            "house_buffer/total_points": float(sum(counts)),
            "house_buffer/max_fill_fraction": max(counts) / capacity,
            "house_buffer/overflow_count": float(
                sum(b.overflow_count for b in self._buffers.values())
            ),
            "house_buffer/failed_insert_count": float(
                sum(b.failed_insert_count for b in self._buffers.values())
            ),
        }


class HouseVoxelsGnnAdapter(HouseVoxelsAdapter):
    """Same pipeline, house cloud routed to the k-NN-GCN branch instead.

    Only the routing changes; the replay fields, the voxel accumulator and the
    extractor policy are identical, which is what makes the two arms comparable.
    """

    CLOUD_ENCODER = Encoder.GNN
