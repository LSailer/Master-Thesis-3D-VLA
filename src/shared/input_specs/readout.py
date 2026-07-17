"""Readout library: what is taken from an observation and how it is prepared.

Each concrete ``Readout`` owns preparation (``prepare``), launch-time shape
resolution (``shape``), storage source/dtype, and the set of legal encoder
kinds. ``EncoderSpec`` composes one or more readouts; a hybrid is simply a
spec with more than one readout.

This module is package-independent: it must not import ``r2dreamer`` or
``vggt``. Extractor outputs arrive via :class:`FeatureBag`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal


import jax.numpy as jnp
from src.environments.observation import ObservationFrame


EncoderKind = Literal["cnn", "mlp", "transformer", "linear", "pointnet", "gnn"]
TokenHalf = Literal["global", "full"]
TokenPool = Literal["mean_max", "flatten"]
WorldPointsSide = int | Literal["dense"]

# VGGT aggregator layout: camera token at 0, four register tokens, then patches.
_AGGREGATOR_PATCH_START_IDX = 5
_XYZ_CHANNELS = 3


class FieldSource(str, Enum):
    """Where a readout's tensor is sourced relative to replay sampling."""

    REPLAY = "replay"
    LIVE_LATEST = "live_latest"

@dataclass(frozen=True, kw_only=True)
class Readout(ABC):
    """Abstract observation readout: prepare + shape + legal encoder kinds.

    Attributes:
        key: Observation dict key this readout produces.
        source: Replay-stored vs live-latest (broadcast over the sequence).
        dtype: Storage / contract dtype string.
        encoder: Chosen encoder kind; defaults to ``DEFAULT_ENCODER``.
    """

    key: str
    source: FieldSource = FieldSource.REPLAY
    dtype: str = "float32"
    encoder: EncoderKind | None = None

    COMPATIBLE: ClassVar[frozenset[EncoderKind]]
    DEFAULT_ENCODER: ClassVar[EncoderKind]

    def __post_init__(self) -> None:
        if self.encoder not in self.COMPATIBLE:
            raise ValueError(f"{type(self).__name__} supports {self.COMPATIBLE}, got {self.encoder}")

    @abstractmethod
    def prepare(self, frame: ObservationFrame) -> jnp.ndarray : ...

