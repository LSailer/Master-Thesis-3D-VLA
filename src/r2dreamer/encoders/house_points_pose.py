"""Launcher-side encoder for live house points plus current camera pose."""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType
from typing import Any

import flax.linen as nn

from src.r2dreamer.encoders.base import VGGTEncoder
from src.r2dreamer.encoders.mlp import (
    HybridHousePointsCameraEncoder as ModelHybridHousePointsCameraEncoder,
)
from src.r2dreamer.encoders.pointnet import (
    PointNetHousePointsCameraEncoder as ModelPointNetHousePointsCameraEncoder,
)
from src.vggt.jax.feature_extractor import ResetMode


class VGGTHousePointsPoseEncoder(VGGTEncoder):
    """Live per-scene house points plus current VGGT camera pose.

    House-context points are accumulated live from VGGT world points into one
    buffer per scene; the encoder emits a fixed-size resampled snapshot each
    step. ``house_points_path`` is an optional warm-start seed (static PLY) and
    is no longer required. The agent-side house branch is the classic PointNet
    module (``PointNetHousePointsCameraEncoder``); the earlier per-point-MLP
    branch (``HousePointsCameraEncoder``) remains available as its parent
    class but is no longer selected here.

    The VGGT streaming cache persists per ``scene_id`` (``PERSIST_SCENE``) so
    all episodes of one house share one world frame and the per-scene point
    buffer accumulates geometrically-consistent points instead of ghost copies
    across episodes (see docs/notes/visible-house-context-snapshot.md). The
    scene-aware reset fires inside ``feature_extractor.extract`` on the first
    frame of each episode; the adapter's ``on_episode_reset`` callback is
    deliberately left unset so it cannot FULL-wipe the outgoing scene before
    ``reset_for_scene`` saves it.
    """

    vggt_reset_mode = ResetMode.PERSIST_SCENE

    def __init__(
        self,
        resolution: int = 518,
        *,
        house_points_path: str | None = None,
        pointcloud_dump_steps: tuple[int, ...] | None = None,
        pointcloud_dump_dir: str | None = None,
    ):
        self._house_points_path = house_points_path
        self._pointcloud_dump_steps = pointcloud_dump_steps
        self._pointcloud_dump_dir = pointcloud_dump_dir
        super().__init__(resolution)

    @classmethod
    def from_train_args(cls, args: Any) -> VGGTHousePointsPoseEncoder:
        """Build live house-points pose mode from parsed train args.

        ``--pointcloud_dump_steps`` (comma-separated env steps) plus the run
        ``output_dir`` enable periodic house-buffer PLY snapshots under
        ``<output_dir>/pointcloud_dumps/``.
        """
        steps_raw = getattr(args, "pointcloud_dump_steps", None)
        dump_steps = (
            tuple(int(s) for s in str(steps_raw).split(",") if s.strip())
            if steps_raw
            else None
        )
        dump_dir = getattr(args, "output_dir", None)
        dump_dir = (
            os.path.join(str(dump_dir), "pointcloud_dumps")
            if dump_steps and dump_dir is not None
            else None
        )
        return cls(
            resolution=args.render_resolution,
            house_points_path=getattr(args, "static_house_points_path", None),
            pointcloud_dump_steps=dump_steps,
            pointcloud_dump_dir=dump_dir,
        )

    @property
    def encoder_type(self) -> str:
        return "vggt_house_points_pose"

    @property
    def module_cls(self) -> type[nn.Module]:
        return ModelPointNetHousePointsCameraEncoder

    @property
    def agent_overrides(self) -> Mapping[str, Any]:
        return MappingProxyType({"buffer_capacity": 1_000_000})

    @classmethod
    def module_kwargs_from_config(cls, config: Any) -> dict[str, Any]:
        """Resolve house-points Encoder Module kwargs from config.

        The PointNet and GNN house modules inherit ``HousePointsCameraEncoder``
        and take the same base kwargs (their extra attrs — graph knots, T-Net
        widths — use module defaults), so the ``GnnHousePointsPoseEncoder`` and
        ``GnnEdgeHousePointsPoseEncoder`` selections inherit this formula
        unchanged.

        ``house_point_norm`` is part of the formula (not a factory overlay) so
        the durable snapshot carries it and eval-from-checkpoint reproduces the
        trained norm instead of the module default. ``compute_dtype`` is the
        only factory-only overlay (not snapshot-serializable).

        Args:
          config: Effective agent config supplying embed/MLP widths and the
            house-point normalization knob.

        Returns:
          Constructor kwargs for the house-points Encoder Module.
        """
        return {
            "embed_dim": int(config.vggt_embed_dim),
            "camera_hidden": int(config.mlp_vggt_hidden),
            "camera_layers": int(config.mlp_vggt_layers),
            "point_hidden": int(config.mlp_vggt_hidden),
            "point_layers": int(config.mlp_vggt_layers),
            "house_point_norm": config.house_point_norm,
        }

    @property
    def design_notes(self) -> str:
        return (
            "Replay current camera pose plus a live per-scene house point buffer "
            "accumulated from VGGT world points (optional static PLY warm-start); "
            "house branch is classic PointNet (input/feature T-Nets, shared MLPs, "
            "max pool — src/r2dreamer/encoders/pointnet.py)."
        )

    @property
    def vggt_compute_heads(self) -> bool:
        return True

    def _build_adapter_for_extractor(self, extractor):
        adapter_module = import_module("src.r2dreamer.adapters.house_points_adapter")
        return adapter_module.VGGTHousePointsPoseObsAdapter(
            extractor,
            house_points_path=self._house_points_path,
            pointcloud_dump_steps=self._pointcloud_dump_steps,
            pointcloud_dump_dir=self._pointcloud_dump_dir,
        )


class VGGTHybridHousePointsPoseEncoder(VGGTHousePointsPoseEncoder):
    """Additive hybrid: rgb64 CNN backbone plus gated live house points + pose.

    Reuses the whole live house-points-pose pipeline (per-scene buffer,
    PERSIST_SCENE, camera-pose replay) and additionally replays the 64x64
    frame so the agent-side module can run the image-only CNN baseline as an
    ungated backbone branch. Zero-init gates on the pose/house branches make
    the encoder start exactly at the CNN baseline (see
    docs/notes/2026-07-06-house-points-vs-cnn-baseline-review.md §5).
    """

    @property
    def encoder_type(self) -> str:
        return "vggt_hybrid_house_points_pose"

    @property
    def module_cls(self) -> type[nn.Module]:
        return ModelHybridHousePointsCameraEncoder

    @classmethod
    def module_kwargs_from_config(cls, config: Any) -> dict[str, Any]:
        """Resolve hybrid house-points kwargs (base house-points plus CNN knobs)."""
        kwargs = super().module_kwargs_from_config(config)
        kwargs.update(
            cnn_depth=int(config.encoder_depth),
            cnn_kernel=int(config.encoder_kernel),
            cnn_mults=tuple(config.encoder_mults),
        )
        return kwargs

    @property
    def design_notes(self) -> str:
        return (
            "RGB64 CNN backbone concatenated with zero-init-gated camera-pose "
            "and pooled live house-points branches; house context injected "
            "live, image + camera pose replayed."
        )

    def _build_adapter_for_extractor(self, extractor):
        adapter_module = import_module("src.r2dreamer.adapters.house_points_adapter")
        return adapter_module.VGGTHybridHousePointsPoseObsAdapter(
            extractor,
            house_points_path=self._house_points_path,
            pointcloud_dump_steps=self._pointcloud_dump_steps,
            pointcloud_dump_dir=self._pointcloud_dump_dir,
        )
