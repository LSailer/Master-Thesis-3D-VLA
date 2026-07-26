"""Pooled point map plus camera pose, with or without the RGB frame."""

from __future__ import annotations

import jax.numpy as jnp

from src.adapters.contract import (
    AdapterField,
    AdapterOutput,
    Encoder,
    FeatureExtractor,
)
from src.adapters.pointmap import pool_point_map
from src.adapters.replay_image import replay_image
from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGT_IMAGE_SIZE, VGGT_PATCH_GRID

# Coarser reduction used by the 64-side arm. Not a divisor of 518, so pooling
# falls back to an antialiased resize instead of an exact box mean.
WP_SIDE_64 = 64


class PointMapPoseAdapter:
    """Routes the frame to a conv branch and VGGT geometry to an MLP branch.

    The geometry field is the pooled point map flattened and concatenated with
    the camera pose - one vector per step, so it replays like any other
    observation and needs no live slot. Nothing accumulates across steps: this
    is the per-frame-geometry arm.

    Metric XYZ stays float32: the values are world coordinates in meters, whose
    magnitude exceeds bfloat16's resolution. The encoder casts to its own
    compute dtype after normalization.

    Two class constants define the family: ``WP_SIDE`` sets the reduction, and
    ``WITH_RGB`` decides whether the appearance channel is observed at all.
    """

    RENDER_RESOLUTION = VGGT_IMAGE_SIZE
    NEEDS_FEATURES = True
    EXTRACTOR_KWARGS: dict[str, object] = {}
    ENCODER_OVERRIDES: dict[str, object] = {}

    # Point-map side after pooling. The patch grid, so pooling is an exact box
    # mean (518 / 37 = 14 pixels per patch).
    WP_SIDE = VGGT_PATCH_GRID
    WITH_RGB = True

    def __init__(self, extractor: FeatureExtractor) -> None:
        """Bind the frozen extractor this adapter reads geometry from."""
        self._extractor = extractor

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        """Route one env frame and its frozen features to the branches."""
        features = self._extractor.extract(frame)
        world_points = pool_point_map(features.world_points, self.WP_SIDE)
        # ravel also drops the extractor's leading singleton frame axis on pose.
        wp_cp = jnp.concatenate(
            [world_points.reshape(-1), jnp.ravel(features.camera_pose)]
        ).astype(jnp.float32)
        fields = [
            AdapterField(
                key="wp_cp", encoder=Encoder.MLP, buffer=True, value=wp_cp
            )
        ]
        if self.WITH_RGB:
            fields.append(
                AdapterField(
                    key="image",
                    encoder=Encoder.CONV,
                    buffer=True,
                    value=replay_image(frame.image),
                    decoder_target=True,
                )
            )
        return fields


class PointMapPoseOnlyAdapter(PointMapPoseAdapter):
    """Geometry only: the same pooled point map and pose, no appearance channel.

    The arm that answers whether geometry alone carries the task. There is no
    RGB field, hence no decoder target - a ``decoder=True`` run is not available
    for this variant.
    """

    WITH_RGB = False


class PointMapPose64OnlyAdapter(PointMapPoseOnlyAdapter):
    """Geometry only at a 64-side reduction instead of the 37 patch grid."""

    WP_SIDE = WP_SIDE_64
