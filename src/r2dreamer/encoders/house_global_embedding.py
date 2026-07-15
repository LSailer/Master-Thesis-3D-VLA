"""Launcher-side encoder for the L1 house global embedding (PointNet reducer).

L1 variant: the agent's scene understanding comes from VGGT's global patch
tokens instead of a 3D point map; cross-episode house memory comes from the
extractor running ``ResetMode.PERSIST_SCENE``. The adapter splits the
global-half aggregator tokens into ``camera_token_global`` (1, 1024) and
``global_patch_tokens`` (1369, 1024); the PointNet reducer encoder
(:class:`HouseGlobalEmbeddingEncoder`) max-pools the patches and keeps the
camera token on its own side branch.

Design: ``src/prototyp/house_global_embedding/IDEA.md`` ("Final design",
2026-07-04). Reuses the whole VGGT extractor pipeline; only the agent-side
Flax module and the adapter change.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType
from typing import Any

import flax.linen as nn

from src.r2dreamer.encoders.base import VGGTEncoder

from src.r2dreamer.encoders.constants import (
    AGG_REGISTER_TOKENS,
    AGG_TOKEN_TOKENS,
    VGGT_AGGREGATOR_EMBED_DIM,
)
from src.r2dreamer.encoders.mlp import HouseGlobalEmbeddingEncoder
from src.vggt.jax.feature_extractor import ResetMode


class VGGTHouseGlobalEmbeddingEncoder(VGGTEncoder):
    """L1 RGB replay + split VGGT global tokens -> PointNet reducer encoder.

    The extractor runs ``ResetMode.PERSIST_SCENE`` with heads off: the agent
    consumes the global-half aggregator tokens, not a 3D point map, so the
    camera/point heads never run and the camera-head KV-cache is never
    allocated (the unbounded-growth risk under PERSIST_SCENE is eliminated by
    construction). The adapter splits ``global_tokens`` into the camera token
    (1, 1024) and the patch tokens (1369, 1024) and stores both float16 in
    replay; the reducer encoder pools the patches and keeps the camera token
    on its own side branch.

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

    @property
    def agent_overrides(self) -> Mapping[str, Any]:
        # Same small-replay budget as the global-token no-gate variant: replay
        # stores ~2.8 MB/step of float16 tokens (1370*1024*2), so a large
        # buffer is infeasible. vggt_token_dim/vggt_token_count fix the module
        # token layout to the VGGT global-half (camera + 4 registers + 1369
        # patches). Smoke/prod YAML may override these.
        return MappingProxyType(
            {
                "buffer_capacity": 5_000,
                "batch_size": 4,
                "seq_len": 32,
                "train_ratio": 128,
                "vggt_token_dim": VGGT_AGGREGATOR_EMBED_DIM,
                "vggt_token_count": AGG_TOKEN_TOKENS,
            }
        )

    @classmethod
    def module_kwargs_from_config(cls, config: Any) -> dict[str, Any]:
        """Resolve HouseGlobalEmbeddingEncoder kwargs from config.

        ``num_patch_tokens`` is fixed by the VGGT global-half token layout:
        ``vggt_token_count`` minus the camera token and the aggregator
        register tokens. ``compute_dtype`` is a factory-only overlay (not
        snapshot-serializable) and is not emitted here.

        Args:
          config: Effective agent config supplying token/embed widths.

        Returns:
          Constructor kwargs for ``HouseGlobalEmbeddingEncoder``.
        """
        num_patch_tokens = int(config.vggt_token_count) - (1 + AGG_REGISTER_TOKENS)
        return {
            "embed_dim": int(config.vggt_embed_dim),
            "token_dim": int(config.vggt_token_dim),
            "num_patch_tokens": num_patch_tokens,
            "reducer_hidden": int(config.mlp_vggt_hidden),
            "reducer_layers": int(config.mlp_vggt_layers),
            "camera_hidden": int(config.mlp_vggt_hidden),
            "camera_layers": int(config.mlp_vggt_layers),
        }

    @property
    def design_notes(self) -> str:
        return (
            "RGB replay plus split VGGT global-half tokens (camera (1,1024) + "
            "patches (1369,1024)); PointNet reducer max-pools the patches and "
            "keeps the camera token on its own side branch (PERSIST_SCENE, "
            "heads off)."
        )

    @property
    def vggt_compute_heads(self) -> bool:
        return False

    def _build_adapter_for_extractor(self, extractor):
        """Build the L1 split-token adapter for one extractor instance.

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