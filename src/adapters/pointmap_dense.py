"""``pointmap_dense``: the full-resolution point map straight into a conv branch."""

from __future__ import annotations

import jax.numpy as jnp

from src.adapters.contract import (
    AdapterField,
    AdapterOutput,
    Encoder,
    FeatureExtractor,
)
from src.adapters.pointmap import squeeze_frame_axis
from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGT_IMAGE_SIZE


class PointMapDenseAdapter:
    """Routes the unpooled ``(518, 518, 3)`` point map to a spatial conv branch.

    No pooling and no appearance channel: the conv stack sees metric world
    coordinates at render resolution, so the geometry keeps its spatial structure
    instead of being flattened into a vector. ``Encoder.CONV_POINTS`` selects the
    symlog input transform - RGB centering would be meaningless on unbounded
    coordinates.

    Replay cost: 518 x 518 x 3 float16 is 1.6 MB per step, and the buffer
    preallocates ``capacity`` rows, so runs of this variant must cap
    ``--buffer_capacity``.
    """

    RENDER_RESOLUTION = VGGT_IMAGE_SIZE
    NEEDS_FEATURES = True
    EXTRACTOR_KWARGS: dict[str, object] = {}
    ENCODER_OVERRIDES: dict[str, object] = {}

    def __init__(self, extractor: FeatureExtractor) -> None:
        """Bind the frozen extractor this adapter reads the point map from."""
        self._extractor = extractor

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        """Route one env frame's raw point map to the conv branch."""
        world_points = squeeze_frame_axis(self._extractor.extract(frame).world_points)
        return [
            AdapterField(
                key="world_points",
                encoder=Encoder.CONV_POINTS,
                buffer=True,
                # float16 halves an already large replay row; the branch's
                # symlog runs in the encoder's compute dtype anyway.
                value=world_points.astype(jnp.float16),
            )
        ]
