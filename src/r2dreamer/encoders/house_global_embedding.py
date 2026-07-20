"""Launcher-side encoder for the L1 house global embedding (PointNet reducer).

L1 variant: the agent's scene understanding comes from VGGT's global patch
tokens instead of a 3D point map; cross-episode house memory comes from the
extractor running ``ResetMode.PERSIST_SCENE``. The adapter drops the camera
and register slots from the global-half map, caches
``global_patch_tokens`` (1369, 1024) live at act time, and replays only
64x64 RGB; :class:`HouseGlobalEmbeddingEncoder` max-pools the patches and
fuses with the conv branch.

Design: ``src/prototyp/house_global_embedding/IDEA.md`` ("Final design",
2026-07-04). Reuses the whole VGGT extractor pipeline; only the agent-side
Flax module and the adapter change.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

import flax.linen as nn

from src.r2dreamer.encoders.base import VGGTEncoder

from src.r2dreamer.encoders.mlp import HouseGlobalEmbeddingEncoder
from src.vggt.jax.feature_extractor import ResetMode


class VGGTHouseGlobalEmbeddingEncoder(VGGTEncoder):
    """L1 RGB replay + live global patch tokens -> PointNet reducer encoder.

    The extractor runs ``ResetMode.PERSIST_SCENE`` with heads off: the agent
    consumes global patch tokens, not a 3D point map, so the camera/point
    heads never run and the camera-head KV-cache is never allocated (the
    unbounded-growth risk under PERSIST_SCENE is eliminated by construction).
    Replay stores only 64x64 RGB; the latest patch map is injected into
    sampled batches via the adapter's ``augment_replay_batch`` hook.

    Optional PLY snapshots (diagnostics only) are driven by the adapter from
    ``pointcloud_dump_every`` / the run ``output_dir``; the point head runs
    only on dump steps, never for training.
    """

    vggt_reset_mode = ResetMode.PERSIST_SCENE
    ENCODER_TYPE = "vggt_house_global_embedding"

    def __init__(
        self,
        resolution: int = 518,
        *,
        pointcloud_dump_every: int = 0,
        pointcloud_dump_dir: str | None = None,
    ):
        self._pointcloud_dump_every = int(pointcloud_dump_every)
        self._pointcloud_dump_dir = pointcloud_dump_dir
        super().__init__(resolution)

    @classmethod
    def from_train_args(cls, args: Any) -> VGGTHouseGlobalEmbeddingEncoder:
        """Build the house-global-embedding encoder from parsed train args.

        Args:
          args: Parsed train args. Reads ``render_resolution``,
            ``pointcloud_dump_every`` (0 disables), and ``output_dir`` (the
            PLY dump subdirectory is derived from it).

        Returns:
          The configured launcher encoder.
        """
        dump_every = int(getattr(args, "pointcloud_dump_every", 0) or 0)
        dump_dir = getattr(args, "output_dir", None)
        if dump_dir is not None:
            dump_dir = os.path.join(str(dump_dir), "pointcloud_dumps")
        return cls(
            resolution=args.render_resolution,
            pointcloud_dump_every=dump_every,
            pointcloud_dump_dir=dump_dir,
        )

    @property
    def encoder_type(self) -> str:
        return self.ENCODER_TYPE

    @property
    def module_cls(self) -> type[nn.Module]:
        return HouseGlobalEmbeddingEncoder

    @classmethod
    def module_kwargs_from_config(cls, config: Any) -> dict[str, Any]:
        """Resolve HouseGlobalEmbeddingEncoder kwargs from config.

        Args:
          config: Effective agent config supplying MLP reducer widths.

        Returns:
          Constructor kwargs for ``HouseGlobalEmbeddingEncoder``.
        """
        return {
            "mlp_layers": int(config.mlp_vggt_layers),
            "hidden_dim": int(config.mlp_vggt_hidden),
        }

    @property
    def design_notes(self) -> str:
        return (
            "RGB replay plus live-injected VGGT global patch tokens "
            "(1369,1024); PointNet reducer max-pools the patches and fuses "
            "with the RGB conv branch (PERSIST_SCENE, heads off)."
        )

    @property
    def vggt_compute_heads(self) -> bool:
        return False

    def _build_adapter_for_extractor(self, extractor):
        """Build the L1 live-patch-token adapter for one extractor instance.

        Args:
          extractor: A VGGT extractor configured for PERSIST_SCENE + heads off.

        Returns:
          The :class:`VGGTHouseGlobalEmbeddingObsAdapter` wired with the dump
          knob/directory.
        """
        adapter_module = import_module("src.r2dreamer.adapters.token_adapters")
        return adapter_module.VGGTHouseGlobalEmbeddingObsAdapter(
            extractor,
            pointcloud_dump_every=self._pointcloud_dump_every,
            pointcloud_dump_dir=self._pointcloud_dump_dir,
        )
