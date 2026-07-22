"""Live per-scene house-point adapters for the house-points-pose encoders.

Accumulates VGGT world points into one :class:`HouseContextPoseBuffer` per
scene and, per step, emits a JIT-stable fixed-size snapshot alongside the
current camera pose. :class:`VGGTHybridHousePointsPoseObsAdapter` additionally
replays the 64x64 RGB frame for the additive-hybrid encoder's CNN branch.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.buffer.replay_buffer import ReplayBatch
from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.encoders.constants import (
    HOUSE_CONTEXT_MAX_POINTS,
    HOUSE_POINT_DIM,
)
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
    HYBRID_IMAGE_KEY,
)
from src.r2dreamer.observation_preparation.static_house_context import (
    load_ascii_ply_xyzrgb,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_IMAGE_SHAPE as IMAGE_SHAPE,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_IMAGE_SIZE as IMAGE_SIZE,
)
from src.r2dreamer.observation_preparation.vggt_readouts import VGGTOutputLike
from src.shared.video_utils import resize_hwc_uint8

CAMERA_POSE_SHAPE = (9,)


def _normalise_house_points(points_xyzrgb: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyzrgb, dtype=np.float32).copy()
    rgb = points[:, 3:6]
    if float(np.max(rgb)) > 1.0:
        rgb = rgb / 255.0
    points[:, 3:6] = np.clip(rgb, 0.0, 1.0)
    return points.astype(np.float16)


def _camera_pose_from_output(out: VGGTOutputLike) -> np.ndarray:
    camera_pose = _vggt_output_field(out, "camera_pose")
    if camera_pose is None:
        raise ValueError("VGGT output field 'camera_pose' is required")
    return np.asarray(camera_pose, dtype=np.float16).reshape(CAMERA_POSE_SHAPE)


def _vggt_output_field(out: VGGTOutputLike, field_name: str) -> Any | None:
    """Return one VGGT output field from object or legacy mapping outputs."""
    if isinstance(out, Mapping):
        value = out.get(field_name)
        if value is None and field_name == "world_points":
            value = out.get("dense_world_points")
        if value is None and field_name == "camera_pose":
            value = out.get(CAMERA_POSE_KEY)
        return value
    return getattr(out, field_name, None)


def _required_vggt_output_field(out: VGGTOutputLike, field_name: str) -> Any:
    """Return a required VGGT output field or raise a contract error."""
    value = _vggt_output_field(out, field_name)
    if value is None:
        raise ValueError(f"VGGT output field {field_name!r} is required")
    return value


class VGGTHousePointsPoseObsAdapter(ObsAdapter):
    """Replay current camera pose plus a live, per-scene house point cloud.

    Replay stores only ``camera_pose`` per step. House-context points are
    accumulated live from VGGT world points into one
    :class:`HouseContextPoseBuffer` per ``scene_id`` (mirroring
    ``ScenePointCloudTracker.point_clouds`` in the ``live_vggt`` prototype).
    Every step the adapter emits a fixed-size
    ``(max_points, 6)`` snapshot zero-padded from the growing buffer, plus the
    true valid-row count under ``HOUSE_CONTEXT_SIZE_KEY`` for masked pooling in
    the encoder, so ``jax.jit`` sees a stable house-context shape and never
    recompiles ``train_step``/``act`` as the cloud grows. Only if the buffer
    outgrows ``max_points`` does the snapshot fall back to an even-stride
    subsample.

    An optional ``house_points_path`` warm-starts every new scene buffer from a
    static ASCII XYZRGB PLY; live VGGT points then extend it. The extractor is
    constructed with ``ResetMode.PERSIST_SCENE`` and resets scene-aware inside
    ``extract`` on the first frame of each episode (``reset_for_scene``), so its
    streaming KV-cache is saved/restored per scene rather than wiped — keeping
    every episode of one house in a single world frame — while the per-scene
    point buffers persist across episodes of the same scene.

    The full VGGT point map (518x518 ~ 268k points) is fed to the buffer every
    step by default: voxel dedup runs as a fixed-shape JIT graph on device
    (~2 ms/frame on H100 regardless of stored size), so no subsampling is
    needed. Passing ``max_input_points > 0`` restores an even-stride subsample
    as an opt-in bound for constrained setups.

    Note (single-scene contract): ``augment_replay_batch`` injects the latest
    scene's snapshot, and ``_house_embedding`` broadcasts that one cloud across
    the whole camera batch — matching the model's global-house-context design
    and the previous static-sidecar behaviour. For multi-scene training this is
    an approximation (the sampled camera poses may come from another scene); the
    L1 curriculum is single-house so it is exact there.
    """

    DEFAULT_CONFIDENCE_SCORE = 1.5
    DEFAULT_VOXEL_SIZE_M = 0.01
    DEFAULT_MAX_INPUT_POINTS = 0
    # 1 cm voxels on full-res input reached ~210k points in 50 steps on one L1
    # scene; a whole house is bounded by surface area (~500 m^2 -> ~5M voxels),
    # so 2^23 (~293 MB device memory) covers any realistic scene outright.
    BUFFER_CAPACITY = 1 << 23
    BUFFER_HASH_TABLE_SIZE = 1 << 24

    def __init__(
        self,
        extractor,
        *,
        house_points_path: str | None = None,
        confidence_score: float = DEFAULT_CONFIDENCE_SCORE,
        voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
        max_points: int = HOUSE_CONTEXT_MAX_POINTS,
        max_input_points: int = DEFAULT_MAX_INPUT_POINTS,
        pointcloud_dump_steps: tuple[int, ...] | None = None,
        pointcloud_dump_dir: str | None = None,
    ):
        self._confidence_score = float(confidence_score)
        self._voxel_size_m = float(voxel_size_m)
        self._max_points = int(max_points)
        self._max_input_points = int(max_input_points)
        self._seed_xyzrgb = self._load_seed(house_points_path)
        self._buffers: dict[str, HouseContextPoseBuffer] = {}
        self._latest_house_context = self._empty_house_context()
        self._latest_house_context_size = jnp.zeros((), dtype=jnp.int32)
        self._env_steps = 0
        self._growth_history: list[tuple[int, int]] = []
        # PLY dump schedule (diagnostics only; buffer.save() syncs to host).
        self._dump_steps = frozenset(int(s) for s in (pointcloud_dump_steps or ()))
        self._dump_dir = pointcloud_dump_dir
        self._dump_enabled = bool(self._dump_steps) and pointcloud_dump_dir is not None
        self._episode_starts_seen = 0
        self._first_episode_dumped = False
        self._extractor = extractor
        super().__init__(
            **self._observation_contract_kwargs(),
            on_episode_reset=self._episode_reset_callback,
        )
        # Scene-aware episode reset. The trainer calls the callback above at
        # every episode boundary — including the two prefill sites that discard
        # the reset frame (so the in-extract ``is_first`` reset path never fires
        # during prefill). Without this callback, ``reset_for_scene`` would not
        # run during prefill, ``_current_scene_id`` would stay None, and the
        # first train episode would fresh-``reset()`` (re-anchor), orphaning the
        # prefill frame — the bug found in smoke 5738008 (3.77 M points, no
        # saturation). See src/prototyp/live_house_context/PROTOCOL.md §2.
        # The earlier ``on_episode_reset=None`` was set to avoid a FULL-wipe
        # hazard, but that hazard only applied to a bare ``extractor.reset``
        # callback; ``reset_for_scene`` saves the outgoing scene before
        # restoring the incoming one, so there is no wipe. The in-extract
        # ``is_first`` -> ``reset_for_scene`` (set in feature_extractor.py) stays
        # as a redundant, idempotent safety net for paths that process the reset
        # frame (the train loop). Only the point buffer persists across
        # episodes; the VGGT cache is saved/restored per scene.
        # ``_episode_reset_callback`` wraps that same ``reset_for_scene`` call
        # and additionally triggers the end-of-first-episode PLY dump.

    def _episode_reset_callback(self, scene_id: str = "scene") -> None:
        """Scene-aware episode reset plus the end-of-first-episode PLY dump.

        Args:
          scene_id: Incoming scene identifier forwarded to
            ``extractor.reset_for_scene``.
        """
        self._episode_starts_seen += 1
        if (
            self._dump_enabled
            and not self._first_episode_dumped
            and self._episode_starts_seen >= 2
        ):
            self._first_episode_dumped = True
            self._dump_buffers("end_of_first_episode")
        self._extractor.reset_for_scene(scene_id)

    def _dump_buffers(self, label: str) -> None:
        """Save every non-empty scene buffer under ``<dump_dir>/<label>/``.

        Args:
          label: Snapshot subdirectory name (e.g. ``step_000500000``).
        """
        if self._dump_dir is None:
            return
        dump_root = Path(self._dump_dir) / label
        for buffer in self._buffers.values():
            if buffer.point_count > 0:
                buffer.save(dump_root)

    def _observation_contract_kwargs(self) -> dict[str, Any]:
        """Return the replay/agent observation contract for ``ObsAdapter``.

        Subclasses extend the returned dicts to replay extra per-step fields
        alongside ``camera_pose`` (the house context itself is injected live,
        never stored).

        Returns:
          Keyword arguments (``buffer_dtype``, ``buffer_shape``,
          ``normalize_on_sample``, ``agent_obs_shape``) for
          ``ObsAdapter.__init__``.
        """
        return {
            "buffer_dtype": {CAMERA_POSE_KEY: "float16"},
            "buffer_shape": {CAMERA_POSE_KEY: CAMERA_POSE_SHAPE},
            "normalize_on_sample": {CAMERA_POSE_KEY: False},
            "agent_obs_shape": {
                CAMERA_POSE_KEY: CAMERA_POSE_SHAPE,
                HOUSE_CONTEXT_KEY: (self._max_points, HOUSE_POINT_DIM),
                HOUSE_CONTEXT_SIZE_KEY: (),
            },
        }

    @staticmethod
    def _load_seed(house_points_path: str | None) -> jnp.ndarray | None:
        """Load an optional static PLY as an ``(M, 6)`` [0, 1] warm-start seed."""
        if house_points_path is None:
            return None
        seed = _normalise_house_points(load_ascii_ply_xyzrgb(house_points_path))
        return jnp.asarray(seed, dtype=jnp.float32)

    def _empty_house_context(self) -> jnp.ndarray:
        """Return the fixed-shape all-zeros house context used before any add."""
        return jnp.zeros((self._max_points, HOUSE_POINT_DIM), dtype=jnp.float16)

    def _get_or_create_buffer(self, scene_id: str) -> HouseContextPoseBuffer:
        """Return the buffer for ``scene_id``, creating and seeding it once."""
        key = scene_id or "scene"
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = HouseContextPoseBuffer(
                confidence_score=self._confidence_score,
                scene_id=key,
                voxel_size_m=self._voxel_size_m,
                capacity=self.BUFFER_CAPACITY,
                hash_table_size=self.BUFFER_HASH_TABLE_SIZE,
            )
            if self._seed_xyzrgb is not None:
                buffer.seed_xyzrgb(self._seed_xyzrgb)
            self._buffers[key] = buffer
        return buffer

    def _house_context_snapshot(
        self, buffer: HouseContextPoseBuffer
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return a JIT-stable ``((max_points, 6) float16, () int32)`` snapshot.

        The scalar is the valid-row count; rows beyond it are zero padding the
        encoder masks out during pooling.
        """
        return buffer.house_context_array(self._max_points, dtype=jnp.float16)

    def _input_stride(self, height: int, width: int) -> int:
        """Return the even stride that caps a ``(height, width)`` map to ~budget."""
        total = int(height) * int(width)
        if self._max_input_points <= 0 or total <= self._max_input_points:
            return 1
        return max(1, int(math.ceil(math.sqrt(total / self._max_input_points))))

    def _subsampled_buffer_input(
        self, out: VGGTOutputLike, env_obs: ObservationFrame
    ) -> tuple[Any, ObservationFrame]:
        """Return ``(vggt_output, observation)`` strided to bound ``add`` cost.

        Strides the ``(H, W, 3)`` world map, ``(H, W)`` confidence and
        ``(H, W, 3)`` image together so they stay pixel-aligned, then wraps them
        in lightweight stand-ins that expose exactly the fields ``add`` reads.
        """
        world_points = _required_vggt_output_field(out, "world_points")
        confidence = _required_vggt_output_field(out, "confidence")
        stride = self._input_stride(world_points.shape[0], world_points.shape[1])
        if stride == 1:
            if not isinstance(out, Mapping):
                return out, env_obs
            shim_out = SimpleNamespace(world_points=world_points, confidence=confidence)
            return shim_out, env_obs
        strided_points = world_points[::stride, ::stride, :]
        strided_conf = jnp.asarray(confidence)[::stride, ::stride]
        strided_image = np.asarray(env_obs.image)[::stride, ::stride, :]
        shim_out = SimpleNamespace(
            world_points=strided_points, confidence=strided_conf
        )
        return shim_out, dataclasses.replace(env_obs, image=strided_image)

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[dict[str, np.ndarray], dict]:
        out = self._extractor.extract(env_obs)
        camera_pose = _camera_pose_from_output(out)
        buffer = self._get_or_create_buffer(env_obs.scene_id)
        buffer_out, buffer_obs = self._subsampled_buffer_input(out, env_obs)
        buffer.add(buffer_out, buffer_obs)
        self._record_growth_sample()
        if self._dump_enabled and self._env_steps in self._dump_steps:
            self._dump_buffers(f"step_{self._env_steps:09d}")
        house_context, house_size = self._house_context_snapshot(buffer)
        self._latest_house_context = house_context
        self._latest_house_context_size = house_size
        replay = {CAMERA_POSE_KEY: camera_pose}
        agent_obs = {
            CAMERA_POSE_KEY: jnp.asarray(camera_pose, dtype=jnp.float16),
            HOUSE_CONTEXT_KEY: house_context,
            HOUSE_CONTEXT_SIZE_KEY: house_size,
            "is_first": env_obs.is_first,
        }
        return replay, agent_obs

    def _record_growth_sample(self) -> None:
        """Sample total stored points at log-spaced (doubling) env steps.

        The host sync happens at steps 1, 2, 4, 8, ... only, so over a 2M-step
        run the growth curve costs ~21 scalar reads in total.
        """
        self._env_steps += 1
        if self._env_steps & (self._env_steps - 1):
            return  # sample only at powers of two
        total = sum(buffer.point_count for buffer in self._buffers.values())
        self._growth_history.append((self._env_steps, total))

    @property
    def growth_history(self) -> list[tuple[int, int]]:
        """``(env_step, total_points)`` samples at doubling env steps."""
        return list(self._growth_history)

    def diagnostics(self) -> dict[str, float]:
        """Per-scene house-buffer usage; syncs one scalar per buffer to host."""
        stats: dict[str, float] = {}
        total_points = 0
        total_overflow = 0
        total_failed = 0
        max_fill = 0.0
        for buffer in self._buffers.values():
            points = buffer.point_count
            total_points += points
            total_overflow += buffer.overflow_count
            total_failed += buffer.failed_insert_count
            max_fill = max(max_fill, points / buffer.capacity)
        if self._buffers:
            stats["house_buffer/scenes"] = float(len(self._buffers))
            stats["house_buffer/total_points"] = float(total_points)
            stats["house_buffer/max_fill_fraction"] = max_fill
            stats["house_buffer/overflow_count"] = float(total_overflow)
            stats["house_buffer/failed_insert_count"] = float(total_failed)
        return stats

    def augment_replay_batch(self, batch: ReplayBatch) -> ReplayBatch:
        """Inject the latest live house-context/pose into a sampled batch.

        Args:
            batch: Sampled replay batch (as returned by ``ReplayBuffer.sample``).

        Returns:
            The batch with ``HOUSE_CONTEXT_KEY`` and ``HOUSE_CONTEXT_SIZE_KEY``
            added to its observation mapping.
        """
        obs = dict(batch.obs)
        obs[HOUSE_CONTEXT_KEY] = self._latest_house_context
        obs[HOUSE_CONTEXT_SIZE_KEY] = self._latest_house_context_size
        return dataclasses.replace(batch, obs=obs)


class VGGTHybridHousePointsPoseObsAdapter(VGGTHousePointsPoseObsAdapter):
    """House-points-pose pipeline plus the rgb64 frame in replay.

    Identical live-buffer/PERSIST_SCENE behaviour to the parent; additionally
    resizes each env frame to 64x64 and stores it per step so the additive
    hybrid encoder (``HybridHousePointsCameraEncoder``) can run its CNN
    baseline branch on replayed images. The house context stays
    live-injected — only ``camera_pose`` and the image are replayed.
    """

    def _observation_contract_kwargs(self) -> dict[str, Any]:
        """Extend the parent contract with the rgb64 replay/agent field.

        Returns:
          ``ObsAdapter.__init__`` kwargs with ``HYBRID_IMAGE_KEY`` added to
          every observation form (uint8 in replay, normalized on sample).
        """
        kwargs = super()._observation_contract_kwargs()
        kwargs["buffer_dtype"][HYBRID_IMAGE_KEY] = "uint8"
        kwargs["buffer_shape"][HYBRID_IMAGE_KEY] = IMAGE_SHAPE
        kwargs["normalize_on_sample"][HYBRID_IMAGE_KEY] = True
        kwargs["agent_obs_shape"][HYBRID_IMAGE_KEY] = IMAGE_SHAPE
        return kwargs

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[dict[str, np.ndarray], dict]:
        replay, agent_obs = super().transform(env_obs)
        image64 = resize_hwc_uint8(env_obs.image, IMAGE_SIZE)
        replay[HYBRID_IMAGE_KEY] = image64
        agent_obs[HYBRID_IMAGE_KEY] = image64
        return replay, agent_obs
