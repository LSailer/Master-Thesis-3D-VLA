"""Launcher-side encoder for live house points plus current camera pose."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType
from typing import Any

import flax.linen as nn

from src.r2dreamer.encoders.base import VGGTEncoder
from src.r2dreamer.encoders.mlp import (
    HousePointsCameraEncoder as ModelHousePointsCameraEncoder,
)


class VGGTHousePointsPoseEncoder(VGGTEncoder):
    """Live per-scene house points plus current VGGT camera pose.

    House-context points are accumulated live from VGGT world points into one
    buffer per scene; the encoder emits a fixed-size resampled snapshot each
    step. ``house_points_path`` is an optional warm-start seed (static PLY) and
    is no longer required.
    """

    def __init__(
        self,
        resolution: int = 518,
        *,
        house_points_path: str | None = None,
    ):
        self._house_points_path = house_points_path
        super().__init__(resolution)

    @classmethod
    def from_train_args(cls, args: Any) -> VGGTHousePointsPoseEncoder:
        """Build live house-points pose mode from parsed train args."""
        return cls(
            resolution=args.render_resolution,
            house_points_path=getattr(args, "static_house_points_path", None),
        )

    @property
    def encoder_type(self) -> str:
        return "vggt_house_points_pose"

    @property
    def module_cls(self) -> type[nn.Module]:
        return ModelHousePointsCameraEncoder

    @property
    def agent_overrides(self) -> Mapping[str, Any]:
        return MappingProxyType({"buffer_capacity": 1_000_000})

    @property
    def design_notes(self) -> str:
        return (
            "Replay current camera pose plus a live per-scene house point buffer "
            "accumulated from VGGT world points (optional static PLY warm-start)."
        )

    @property
    def vggt_compute_heads(self) -> bool:
        return True

    def _build_adapter_for_extractor(self, extractor):
        adapter_module = import_module("src.r2dreamer.adapters.hybrid_adapter")
        return adapter_module.VGGTHousePointsPoseObsAdapter(
            extractor,
            house_points_path=self._house_points_path,
        )
