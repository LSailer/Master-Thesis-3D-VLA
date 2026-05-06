from abc import ABC, abstractmethod

from modules.r2dreamer.adapters.obs_adapter import ObsAdapter
from modules.r2dreamer.adapters.vggt_adapter import VGGTObsAdapter
from modules.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


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

    # Match the fast JAX benchmark configuration (`--jax-static-budgets`).
    # Dynamic per-layer budgets trigger JAX/XLA recompilation after cache
    # eviction starts because the budget tuple is a jit static argument.
    VGGT_TOTAL_BUDGET = 200_000
    VGGT_STATIC_BUDGETS = tuple([8333] * 24)

    def __init__(self, resolution: int = 518):
        self._extractor = VGGTFeatureExtractor(
            total_budget=self.VGGT_TOTAL_BUDGET,
            budgets_static=self.VGGT_STATIC_BUDGETS,
        )  # device="cuda" default

    def make_adapter(self) -> ObsAdapter:
        return VGGTObsAdapter(self._extractor)
