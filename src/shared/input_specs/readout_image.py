"""RGB image readout for the CNN encoder branch.

Defines ``IMAGE``: downsamples ``ObservationFrame.image`` for replay/encoder
use. Replay stores uint8; ``/255`` centering happens in ``ConvEncoder``.
Package-independent of ``r2dreamer`` / ``vggt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from src.environments.observation import ObservationFrame
from src.shared.input_specs.readout import EncoderKind, Readout


@dataclass(frozen=True, kw_only=True)
class IMAGE(Readout):
    """Downsampled HWC RGB for the CNN encoder branch.

    Replay stores uint8; ``/255`` centering happens in ``ConvEncoder``, not here.
    """

    size: int = 64
    key: str = "image"
    dtype: str = "uint8"

    COMPATIBLE: ClassVar[frozenset[EncoderKind]] = frozenset({"cnn"})
    DEFAULT_ENCODER: ClassVar[EncoderKind] = "cnn"

    def prepare(self, frame: ObservationFrame) -> jnp.ndarray:
        """Resize ``frame.image`` to ``(size, size, 3)`` uint8 HWC."""
        return jax.image.resize(frame.image, shape=(self.size, self.size, 3), method="bilinear")
