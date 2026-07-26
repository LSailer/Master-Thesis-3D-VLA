"""``rgb``: the RGB-only baseline, no frozen feature extraction."""

from __future__ import annotations

from src.adapters.contract import AdapterField, AdapterOutput, Encoder
from src.adapters.replay_image import REPLAY_IMAGE_SIZE, replay_image
from src.environments.observation import ObservationFrame


class RgbAdapter:
    """Routes the rendered frame to a single conv branch.

    The control baseline: no VGGT, no house context, one field. Its replay
    image is also the decoder probe's reconstruction target.
    """

    # The only variant that renders at replay size: nothing here needs the full
    # VGGT input frame, so the env renders 64x64 directly.
    RENDER_RESOLUTION = REPLAY_IMAGE_SIZE
    NEEDS_FEATURES = False
    EXTRACTOR_KWARGS: dict[str, object] = {}
    ENCODER_OVERRIDES: dict[str, object] = {}

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        """Route one env frame to the conv branch."""
        return [
            AdapterField(
                key="image",
                encoder=Encoder.CONV,
                buffer=True,
                value=replay_image(frame.image),
                decoder_target=True,
            )
        ]
