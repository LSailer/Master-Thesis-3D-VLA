from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from modules.r2dreamer.adapters.obs_adapter import ObsAdapter
from modules.r2dreamer.adapters.vggt_adapter import VGGTObsAdapter
from modules.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


@dataclass(frozen=True)
class EncoderSpec:
    """Observation and agent requirements exposed by an encoder."""

    obs_shape: tuple[int, ...]
    env_render_resolution: int
    encoder_type: str
    agent_overrides: dict[str, Any] = field(default_factory=dict)
    design_notes: str = ""


class Encoder(ABC):
    """Base class for everything an agent might consume as input."""

    @classmethod
    def from_train_args(cls, args: Any) -> "Encoder":
        """Construct an encoder from train() CLI args."""
        return cls()

    @abstractmethod
    def make_adapter(self) -> ObsAdapter:
        """Return the ObsAdapter that bridges env obs to agent input."""

    @abstractmethod
    def spec(self) -> EncoderSpec:
        """Return static observation/env/agent requirements for train()."""


class CNNEncoder(Encoder):
    """Identity encoder — agent's internal CNN handles RGB -> embedding -> RSSM."""

    def make_adapter(self) -> ObsAdapter:
        return ObsAdapter()  # passthrough, default behavior

    def spec(self) -> EncoderSpec:
        adapter = self.make_adapter()
        return EncoderSpec(
            obs_shape=adapter.buffer_shape,
            env_render_resolution=64,
            encoder_type="cnn",
        )


class VGGTEncoder(Encoder):
    """External feature extractor — 518x518 RGB -> 4116-dim flat vector."""

    # Match the fast JAX benchmark configuration (`--jax-static-budgets`).
    # Dynamic per-layer budgets trigger JAX/XLA recompilation after cache
    # eviction starts because the budget tuple is a jit static argument.
    VGGT_TOTAL_BUDGET = 200_000
    VGGT_STATIC_BUDGETS = tuple([8333] * 24)
    feature_kind = "wp_cp"
    encoder_type = "vggt"

    @classmethod
    def from_train_args(cls, args: Any) -> "VGGTEncoder":
        return cls(resolution=args.render_resolution)

    def __init__(self, resolution: int = 518):
        self.resolution = resolution
        self._adapter: VGGTObsAdapter | None = None
        self._extractor = VGGTFeatureExtractor(
            total_budget=self.VGGT_TOTAL_BUDGET,
            budgets_static=self.VGGT_STATIC_BUDGETS,
        )  # device="cuda" default

    def make_adapter(self) -> ObsAdapter:
        if self._adapter is None:
            self._adapter = VGGTObsAdapter(self._extractor, feature_kind=self.feature_kind)
        return self._adapter

    def spec(self) -> EncoderSpec:
        adapter = self.make_adapter()
        return EncoderSpec(
            obs_shape=adapter.buffer_shape,
            env_render_resolution=self.resolution,
            encoder_type=self.encoder_type,
            agent_overrides={"buffer_capacity": 1_000_000},
        )


class VGGTAggregatorMLPEncoder(VGGTEncoder):
    """External VGGT extractor exposing pre-head aggregator patch features."""

    feature_kind = "aggregator"
    encoder_type = "vggt_aggregator_mlp"
    design_notes = (
        "Variant 1 encoder: VGGT final pre-head aggregator global patch features "
        "(37x37x1024) -> 1x1 Conv 1024->64 -> flatten 87616 -> "
        "2-layer MLP 87616->1024->1024; excludes VGGT world-points and camera-pose heads."
    )

    def spec(self) -> EncoderSpec:
        adapter = self.make_adapter()
        return EncoderSpec(
            obs_shape=adapter.buffer_shape,
            env_render_resolution=self.resolution,
            encoder_type=self.encoder_type,
            agent_overrides={
                "buffer_capacity": 5_000,
                "batch_size": 4,
                "seq_len": 32,
                "train_ratio": 128,
            },
            design_notes=self.design_notes,
        )
