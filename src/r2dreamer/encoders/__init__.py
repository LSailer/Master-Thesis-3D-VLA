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

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.vggt_adapter import VGGTObsAdapter
from src.r2dreamer.world_model import encoders as wm_encoders
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


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
        "Variant 1 encoder: VGGT final pre-head global aggregator tokens "
        "(1374x1024 = 1 camera token + 4 register tokens + 37x37 patch tokens) "
        "are pooled adapter-side into three 1024-dim vectors before replay: "
        "(a) the camera token (idx 0) is kept unmixed because VGGT's own "
        "camera_head reads it for pose; (b) mean over patch tokens (idx 5:) "
        "is a smooth global summary; (c) max over the same patches surfaces "
        "salient features. The three are concatenated to a flat (3072,) "
        "vector, stored as float32 in replay, and the encoder applies a "
        "per-slice RMSNorm followed by a 2-layer MLP -> embed_dim. The "
        "camera-pose head is skipped (vggt_compute_heads=False) since the "
        "camera-token embedding itself already carries pose information."
    )


class VGGTAggregatorBothMLPEncoder(VGGTAggregatorMLPEncoder):
    """VGGT extractor pooling the full frame ⊕ global aggregator token (3D-47).

    Same pre-head, heads-skipped pipeline as the global-only Aggregator variant,
    but the adapter keeps the aggregator's native 2048-d token (frame/local ⊕
    global/contextual) instead of the global half alone, yielding a (6144,)
    pooled vector. The only thing that differs from `vggt_aggregator_mlp` is the
    token set fed to the encoder; this is the variable under test.
    """

    feature_kind = "aggregator_both"
    encoder_type = "vggt_aggregator_both_mlp"
    module_cls = wm_encoders.VGGTAggregatorBothMLPEncoder
    design_notes = (
        "3D-47 encoder: identical to vggt_aggregator_mlp but the adapter pools "
        "the aggregator's native [frame_inter, global_inter] 2048-d token rather "
        "than the 1024-d global stream alone, so the discarded per-frame/local "
        "stream is included. Pooling is the same cam/mean/max recipe over the "
        "wider token -> flat (6144,) float32 in replay; the encoder applies a "
        "per-slice RMSNorm (3 x 2048) + 2-layer MLP -> embed_dim. Compared "
        "head-to-head against the global-only readout under an identical buffer."
    )


__all__ = [
    "EncoderSpec",
    "Encoder",
    "CNNEncoder",
    "VGGTEncoder",
    "VGGTAggregatorMLPEncoder",
    "VGGTAggregatorBothMLPEncoder",
]
