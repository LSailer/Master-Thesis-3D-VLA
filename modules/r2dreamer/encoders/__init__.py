"""Encoder specs: launcher-side description of the full input pipeline.

Each subclass of `Encoder` is the single source of truth for one encoder choice:
the env-side adapter, the env render resolution, agent config overrides, AND the
matching Flax `nn.Module` class consumed inside the agent. The agent reads
`module_cls` from the resolved `EncoderSpec` and instantiates the network on
its side of the `jax.jit` boundary — the launcher never holds a live module
instance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn

from modules.r2dreamer.adapters.obs_adapter import ObsAdapter
from modules.r2dreamer.adapters.vggt_adapter import VGGTObsAdapter
from modules.r2dreamer.world_model import encoders as wm_encoders
from modules.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


@dataclass(frozen=True)
class EncoderSpec:
    """Full description of an encoder choice, including its `nn.Module` class.

    `module_cls` is the Flax module the agent will instantiate (with kwargs
    pulled from `R2DreamerConfig`) — passing the class object rather than an
    instance keeps the launcher on the non-JIT side of the boundary.
    """

    obs_shape: tuple[int, ...]
    env_render_resolution: int
    encoder_type: str
    module_cls: type[nn.Module]
    agent_overrides: dict[str, Any] = field(default_factory=dict)
    design_notes: str = ""


class Encoder(ABC):
    """Base class for everything an agent might consume as input.

    Subclasses declare *what they are* via class attributes
    (encoder_type / module_cls / agent_overrides / design_notes /
    env_render_resolution) and *how to build their adapter* via _build_adapter().
    The base spec() sources obs_shape from the adapter so encoder and adapter
    can never disagree on observation shape.
    """

    encoder_type: str = ""
    module_cls: type[nn.Module] | None = None
    env_render_resolution: int = 64
    # Class-attribute defaults are read-only here: subclasses *reassign*
    # them (e.g. agent_overrides = {...}) and spec() copies via dict(...)
    # before exposure, so the shared instance is never mutated in place.
    agent_overrides: dict[str, Any] = {}
    design_notes: str = ""

    _adapter: ObsAdapter | None = None

    @classmethod
    def from_train_args(cls, args: Any) -> "Encoder":
        """Construct an encoder from train() CLI args."""
        return cls()

    def make_adapter(self) -> ObsAdapter:
        """Return the ObsAdapter that bridges env obs to agent input (cached)."""
        if self._adapter is None:
            self._adapter = self._build_adapter()
        return self._adapter

    @abstractmethod
    def _build_adapter(self) -> ObsAdapter:
        """Build the adapter. Called at most once per encoder; result is cached."""

    def spec(self) -> EncoderSpec:
        """Static observation/env/agent requirements for train()."""
        if self.module_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set module_cls (a Flax nn.Module class)"
            )
        adapter = self.make_adapter()
        return EncoderSpec(
            obs_shape=adapter.buffer_shape,
            env_render_resolution=self.env_render_resolution,
            encoder_type=self.encoder_type,
            module_cls=self.module_cls,
            agent_overrides=dict(self.agent_overrides),
            design_notes=self.design_notes,
        )


class CNNEncoder(Encoder):
    """Identity encoder — agent's internal CNN handles RGB -> embedding -> RSSM."""

    encoder_type = "cnn"
    module_cls = wm_encoders.ConvEncoder

    def _build_adapter(self) -> ObsAdapter:
        return ObsAdapter()  # passthrough, default behavior


class VGGTEncoder(Encoder):
    """External feature extractor — 518x518 RGB -> 4116-dim flat vector."""

    # Match the fast JAX benchmark configuration (`--jax-static-budgets`).
    # Dynamic per-layer budgets trigger JAX/XLA recompilation after cache
    # eviction starts because the budget tuple is a jit static argument.
    VGGT_TOTAL_BUDGET = 200_000
    VGGT_STATIC_BUDGETS = tuple([8333] * 24)

    feature_kind = "wp_cp"
    encoder_type = "vggt"
    module_cls = wm_encoders.VGGTEncoder
    agent_overrides = {"buffer_capacity": 1_000_000}
    # Subclasses set vggt_compute_heads = False when they consume only the
    # pre-head aggregator tokens, so the extractor can skip camera_head +
    # point_head + world_points wrapper on every frame.
    vggt_compute_heads = True

    @classmethod
    def from_train_args(cls, args: Any) -> "VGGTEncoder":
        return cls(resolution=args.render_resolution)

    def __init__(self, resolution: int = 518):
        self.env_render_resolution = resolution
        self._extractor = VGGTFeatureExtractor(
            total_budget=self.VGGT_TOTAL_BUDGET,
            budgets_static=self.VGGT_STATIC_BUDGETS,
            compute_heads=self.vggt_compute_heads,
        )  # device="cuda" default

    def _build_adapter(self) -> ObsAdapter:
        return VGGTObsAdapter(self._extractor, feature_kind=self.feature_kind)


class VGGTAggregatorMLPEncoder(VGGTEncoder):
    """External VGGT extractor exposing pooled pre-head aggregator features."""

    feature_kind = "aggregator"
    encoder_type = "vggt_aggregator_mlp"
    module_cls = wm_encoders.VGGTAggregatorMLPEncoder
    # Aggregator-only path: skip camera_head + point_head + world_points wrapper
    # in the extractor; only `aggregator_features` is needed downstream.
    vggt_compute_heads = False
    agent_overrides = {
        "buffer_capacity": 5_000,
        "batch_size": 4,
        "seq_len": 32,
        "train_ratio": 128,
    }
    design_notes = (
        "Variant 1 encoder: VGGT final pre-head all-token global aggregator features "
        "(1374x1024 = 5 camera/register special tokens + 37x37 patch tokens) "
        "-> mean-pooled to 1024-dim float32 before replay storage -> linear projection; "
        "excludes VGGT world-points and camera-pose heads. This preserves the global "
        "aggregator-token signal while avoiding huge all-token replay batches."
    )


__all__ = [
    "EncoderSpec",
    "Encoder",
    "CNNEncoder",
    "VGGTEncoder",
    "VGGTAggregatorMLPEncoder",
]
