from abc import ABC, abstractmethod

from modules.r2dreamer.adapters.obs_adapter import ObsAdapter
from modules.r2dreamer.adapters.vggt_adapter import VGGTObsAdapter
from modules.vggt.feature_extractor import VGGTFeatureExtractor


class Encoder(ABC):
    """Base class for everything an agent might consume as input."""

    @abstractmethod
    def make_adapter(self) -> ObsAdapter:
        """Return the ObsAdapter that bridges env obs to agent input."""


class CNNEncoder(Encoder):
    """Identity encoder — agent's internal CNN handles RGB -> embedding -> RSSM."""

    def make_adapter(self) -> ObsAdapter:
        return ObsAdapter()  # passthrough, default behavior


class VGGTEncoder(Encoder):
    """External feature extractor — 518x518 RGB -> 4116-dim flat vector."""

    def __init__(self, resolution: int = 518):
        self._extractor = VGGTFeatureExtractor()  # device="cuda" default

    def make_adapter(self) -> ObsAdapter:
        return VGGTObsAdapter(self._extractor)
