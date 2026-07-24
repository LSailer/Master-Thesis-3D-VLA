"""HouseContextAdapter: persistent cross-episode world point cloud."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.adapters.contract import (
    AdapterField,
    AdapterOutput,
    Encoder,
    FeatureAdapterFn,
)
from src.environments.observation import ObservationFrame
from src.shared.pointcloud import voxel_down_sample
from src.vggt.jax.feature_extractor import VGGTExtractOutput


class HouseContextAdapter:
    """Accumulates a persistent world point cloud across episodes.

    At each episode boundary (``frame.is_first``) the accumulated cloud is
    voxel-downsampled to a fixed fraction of its extent, then the new
    episode's frames keep appending to it.
    """

    def __init__(self) -> None:
        self._xyz: jnp.ndarray | None = None  # (N, 3) accumulated positions
        self._rgb: jnp.ndarray | None = None  # (N, 3) matching colors in [0, 1]

    def _compact(self, xyz: jnp.ndarray, rgb: jnp.ndarray) -> None:
        extent = jnp.ptp(xyz, axis=0)
        voxel = float(extent.max()) / 1000
        self._xyz, self._rgb = voxel_down_sample(xyz, rgb, voxel)

    def __call__(
        self, frame: ObservationFrame, features: VGGTExtractOutput
    ) -> AdapterOutput:
        if frame.is_first and self._xyz is not None and self._rgb is not None:
            # Previous episode ended: downsample before the new episode appends.
            self._compact(self._xyz, self._rgb)

        xyz_new = features.world_points.reshape(-1, 3)
        rgb_new = frame.image.reshape(-1, 3).astype(jnp.bfloat16) / 255.0
        if self._xyz is None or self._rgb is None:
            self._xyz, self._rgb = xyz_new, rgb_new
        else:
            self._xyz = jnp.concatenate([self._xyz, xyz_new], axis=0)
            self._rgb = jnp.concatenate([self._rgb, rgb_new], axis=0)

        # Replay/encoder image is 64x64 (repo convention); the full-resolution
        # frame stays with VGGT and the rgb cloud above.
        image_64 = jax.image.resize(
            jnp.asarray(frame.image, jnp.float32), (64, 64, 3), method="linear"
        ).astype(jnp.uint8)

        return [
            AdapterField(
                key="house_context",
                encoder=Encoder.POINTNET,
                buffer=False,
                value=jnp.concatenate([self._xyz, self._rgb], axis=-1),  # (N, 6)
            ),
            AdapterField(
                key="image", encoder=Encoder.CONV, buffer=True, value=image_64
            ),
        ]


_check: FeatureAdapterFn = HouseContextAdapter()
