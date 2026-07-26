"""``house_cloud_episodes``: one point cloud carried across episodes."""

from __future__ import annotations

import jax.numpy as jnp

from src.adapters.contract import (
    AdapterField,
    AdapterOutput,
    Encoder,
    FeatureExtractor,
)
from src.adapters.replay_image import replay_image
from src.environments.observation import ObservationFrame
from src.shared.pointcloud import voxel_down_sample
from src.vggt.jax.feature_extractor import VGGT_IMAGE_SIZE


class HouseCloudEpisodesAdapter:
    """Accumulates a world point cloud that survives episode boundaries.

    Every frame's world points are appended raw; at each episode boundary
    (``frame.is_first``) the accumulated cloud is voxel-downsampled to a fixed
    fraction of its own extent, then the new episode keeps appending. The cloud
    is the one live field - a global context, not a per-step observation - so
    replay stores only its latest value.

    Unlike the voxel-buffer arm this keeps no camera pose and no per-scene
    separation: the question it exists to answer is whether context accumulated
    across episodes helps at all.
    """

    RENDER_RESOLUTION = VGGT_IMAGE_SIZE
    NEEDS_FEATURES = True
    EXTRACTOR_KWARGS: dict[str, object] = {}
    ENCODER_OVERRIDES: dict[str, object] = {}

    # Voxel edge = cloud extent / this, so the grid scales with the house
    # instead of with a metric constant.
    VOXELS_PER_EXTENT = 1000
    # Floor for the voxel edge: a degenerate cloud (all points coincident, e.g.
    # a collapsed point map) has zero extent, and a zero edge is not a valid
    # voxel size. Sub-millimeter, so it never affects a real house.
    MIN_VOXEL_M = 1e-6

    def __init__(self, extractor: FeatureExtractor) -> None:
        """Bind the frozen extractor this adapter reads world points from."""
        self._extractor = extractor
        self._xyz: jnp.ndarray | None = None  # (N, 3) accumulated positions
        self._rgb: jnp.ndarray | None = None  # (N, 3) matching colors in [0, 1]

    def _compact(self, xyz: jnp.ndarray, rgb: jnp.ndarray) -> None:
        extent = float(jnp.ptp(xyz, axis=0).max())
        voxel = max(extent / self.VOXELS_PER_EXTENT, self.MIN_VOXEL_M)
        self._xyz, self._rgb = voxel_down_sample(xyz, rgb, voxel)

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        """Extend the cloud with this frame and route both fields."""
        features = self._extractor.extract(frame)
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

        return [
            AdapterField(
                key="house_context",
                encoder=Encoder.POINTNET,
                buffer=False,
                value=jnp.concatenate([self._xyz, self._rgb], axis=-1),  # (N, 6)
            ),
            AdapterField(
                key="image",
                encoder=Encoder.CONV,
                buffer=True,
                # The full-resolution frame stays with VGGT and the rgb cloud
                # above; replay and the conv branch see the 64x64 view.
                value=replay_image(frame.image),
                decoder_target=True,
            ),
        ]
