"""Launcher-side encoder specs and adapter construction.

Each subclass of `Encoder` is the single source of truth for one encoder choice:
the env-side adapter, the env render resolution, agent config overrides, AND the
matching Flax `nn.Module` class consumed inside the agent. The agent reads
`module_cls` from the resolved `EncoderSpec` and instantiates the network on
its side of the `jax.jit` boundary - the launcher never holds a live module
instance.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import flax.linen as nn

from src.r2dreamer.adapters.hybrid_adapter import (
    HybridObsAdapter,
    VGGTHouseContextObsAdapter,
    VGGTHouseFullTokenObsAdapter,
    VGGTHouseGlobalTokenObsAdapter,
)
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.vggt_adapter import VGGTFeatureKind, VGGTObsAdapter
from src.r2dreamer.observation_preparation import CNNObservationPreparation
from src.r2dreamer.observation_preparation.vggt import (
    VGGT_DREAMER_SPECS,
    VGGTDreamerSpec,
)
from src.r2dreamer.world_model import encoders as wm_encoders
from src.vggt.jax.feature_extractor import (
    JAXVGGTFeatureExtractor as VGGTFeatureExtractor,
)


@dataclass(frozen=True)
class EncoderSpec:
    """Full description of an encoder choice, including its `nn.Module` class.

    `module_cls` is the Flax module the agent will instantiate (with kwargs
    pulled from `R2DreamerConfig`) - passing the class object rather than an
    instance keeps the launcher on the non-JIT side of the boundary.
    """

    obs_shape: tuple[int, ...] | Mapping[str, tuple[int, ...]]
    env_render_resolution: int
    encoder_type: str
    module_cls: type[nn.Module]
    agent_overrides: dict[str, Any] = field(default_factory=dict)
    design_notes: str = ""
    contract_snapshot: dict[str, Any] | None = None


VGGTVariantSpec = VGGTDreamerSpec
VGGT_VARIANTS = VGGT_DREAMER_SPECS


class Encoder(ABC):
    """Base class for everything an agent might consume as input.

    Subclasses declare what they are via class attributes and how to build their
    adapter via `_build_adapter()`. The base `spec()` sources `obs_shape` from
    the adapter so encoder and adapter can never disagree on observation shape.
    """

    encoder_type: str = ""
    module_cls: type[nn.Module] | None = None
    env_render_resolution: int = 64
    agent_overrides: Mapping[str, Any] = {}
    design_notes: str = ""

    _adapter: ObsAdapter | None = None

    @classmethod
    def from_train_args(cls, _args: Any) -> "Encoder":
        """Construct an encoder from train() CLI args."""
        return cls()

    def make_adapter(self) -> ObsAdapter:
        """Return the ObsAdapter that bridges env obs to agent input (cached)."""
        if self._adapter is None:
            self._adapter = self._build_adapter()
        return self._adapter

    def new_adapter(self) -> ObsAdapter:
        """Build an uncached adapter with independent mutable state."""
        return self._build_adapter()

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
        contract = getattr(adapter, "contract", None)
        if contract is not None:
            return EncoderSpec(
                obs_shape=contract.encoder_input.buffer_shape(),
                env_render_resolution=contract.env_render_resolution,
                encoder_type=contract.encoder_type,
                module_cls=contract.encoder_module_cls,
                agent_overrides=dict(contract.agent_overrides),
                design_notes=contract.design_notes,
                contract_snapshot=contract.to_snapshot(),
            )
        return EncoderSpec(
            obs_shape=adapter.encoder_obs_shape,
            env_render_resolution=self.env_render_resolution,
            encoder_type=self.encoder_type,
            module_cls=self.module_cls,
            agent_overrides=dict(self.agent_overrides),
            design_notes=self.design_notes,
        )


class CNNEncoder(Encoder):
    """CNN Observation Preparation feeding the internal ConvEncoder."""

    encoder_type = "cnn"
    module_cls = wm_encoders.ConvEncoder

    def _build_adapter(self) -> ObsAdapter:
        return CNNObservationPreparation()


class VGGTEncoder(Encoder):
    """External feature extractor - 518x518 RGB -> configured VGGT readout."""

    # Match the fast JAX benchmark configuration (`--jax-static-budgets`).
    # Dynamic per-layer budgets trigger JAX/XLA recompilation after cache
    # eviction starts because the budget tuple is a jit static argument.
    VGGT_TOTAL_BUDGET = 200_000
    VGGT_STATIC_BUDGETS = tuple([8333] * 24)

    variant = VGGT_VARIANTS["vggt"]

    @property
    def feature_kind(self) -> VGGTFeatureKind:
        """VGGT readout key used by the adapter."""
        return self.variant.feature_kind

    @property
    def encoder_type(self) -> str:
        """Agent encoder type string."""
        return self.variant.encoder_type

    @property
    def module_cls(self) -> type[nn.Module]:
        """Flax encoder module class."""
        return self.variant.module_cls

    @property
    def agent_overrides(self) -> Mapping[str, Any]:
        """Variant-specific R2DreamerConfig overrides."""
        return self.variant.agent_overrides

    @property
    def design_notes(self) -> str:
        """Human-readable variant notes for manifests."""
        return self.variant.design_notes

    @property
    def vggt_compute_heads(self) -> bool:
        """Whether the frozen VGGT extractor computes point/camera heads."""
        return self.variant.compute_heads

    @property
    def wp_pool_size(self) -> int:
        """World-point pooling side length."""
        return self.variant.wp_pool_size

    @classmethod
    def from_train_args(cls, args: Any) -> "VGGTEncoder":
        return cls(resolution=args.render_resolution)

    def __init__(self, resolution: int = 518):
        self.env_render_resolution = resolution
        self._extractor = self._make_extractor()

    def _make_extractor(self):
        return VGGTFeatureExtractor(
            total_budget=self.VGGT_TOTAL_BUDGET,
            budgets_static=self.VGGT_STATIC_BUDGETS,
            compute_heads=self.vggt_compute_heads,
            wp_pool_size=self.wp_pool_size,
        )  # device="cuda" default

    def _build_adapter(self) -> ObsAdapter:
        return self._build_adapter_for_extractor(self._extractor)

    def new_adapter(self) -> ObsAdapter:
        """Build an adapter backed by a fresh VGGT extractor/cache."""
        return self._build_adapter_for_extractor(self._make_extractor())

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        return VGGTObsAdapter(
            extractor,
            feature_kind=self.feature_kind,
            env_render_resolution=self.env_render_resolution,
            encoder_type=self.encoder_type,
            encoder_module_cls=self.module_cls,
            agent_overrides=self.agent_overrides,
            design_notes=self.design_notes,
        )


class VGGTAggregatorMLPEncoder(VGGTEncoder):
    """External VGGT extractor exposing pooled pre-head aggregator features."""

    variant = VGGT_VARIANTS["vggt_aggregator_mlp"]


class VGGTAggTokenTransformerEncoder(VGGTEncoder):
    """Full VGGT aggregator token replay -> trainable Token Transformer."""

    variant = VGGT_VARIANTS["vggt_agg_token_transformer"]


class VGGTDenseWPEncoder(VGGTEncoder):
    """Full-resolution world-point map (518x518x3) -> Conv encoder (3D-53)."""

    variant = VGGT_VARIANTS["vggt_wp_dense_cnn"]


class HybridEncoder(VGGTEncoder):
    """CNN(RGB 64) + gated MLP(WP+CP 4116) fused into a single latent."""

    variant = VGGT_VARIANTS["hybrid"]

    def _build_adapter(self) -> ObsAdapter:
        return self._build_adapter_for_extractor(self._extractor)

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        return HybridObsAdapter(
            extractor,
            env_render_resolution=self.env_render_resolution,
            encoder_module_cls=self.module_cls,
            agent_overrides=self.agent_overrides,
            design_notes=self.design_notes,
        )


class VGGTHouseContextEncoder(VGGTEncoder):
    """L1 RGB replay + live bounded InfiniteVGGT house-context readout."""

    variant = VGGT_VARIANTS["vggt_house_context"]

    @classmethod
    def from_train_args(cls, args: Any) -> "VGGTHouseContextEncoder":
        return cls(
            resolution=args.render_resolution,
            transformer_layers=(
                args.vggt_token_transformer_layers
                if args.vggt_token_transformer_layers is not None
                else 2
            ),
            transformer_heads=(
                args.vggt_token_transformer_heads
                if args.vggt_token_transformer_heads is not None
                else 8
            ),
            transformer_mlp_ratio=(
                args.vggt_token_transformer_mlp_ratio
                if args.vggt_token_transformer_mlp_ratio is not None
                else 2
            ),
            transformer_dropout=(
                args.vggt_token_transformer_dropout
                if args.vggt_token_transformer_dropout is not None
                else 0.0
            ),
        )

    def __init__(
        self,
        resolution: int = 518,
        *,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_ratio: int = 2,
        transformer_dropout: float = 0.0,
    ):
        super().__init__(resolution)
        self._context_transformer = wm_encoders.VGGTFullTokenContextTransformer(
            context_dim=wm_encoders.HOUSE_CONTEXT_DIM,
            token_dim=2048,
            num_tokens=wm_encoders.AGG_TOKEN_TOKENS,
            layers=transformer_layers,
            heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout,
        )

    @property
    def agent_overrides(self) -> Mapping[str, Any]:
        overrides = dict(self.variant.agent_overrides)
        overrides.update(
            {
                "vggt_token_transformer_layers": self._context_transformer.layers,
                "vggt_token_transformer_heads": self._context_transformer.heads,
                "vggt_token_transformer_mlp_ratio": self._context_transformer.mlp_ratio,
                "vggt_token_transformer_dropout": self._context_transformer.dropout,
            }
        )
        return MappingProxyType(overrides)

    def _build_adapter(self) -> ObsAdapter:
        return self._build_adapter_for_extractor(self._extractor)

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        return VGGTHouseContextObsAdapter(
            extractor,
            context_transformer=self._context_transformer,
        )


class VGGTHouseFullTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + live full-token Transformer inside the agent, no gate."""

    variant = VGGT_VARIANTS["vggt_house_full_tokens_nogate"]

    def _build_adapter(self) -> ObsAdapter:
        return self._build_adapter_for_extractor(self._extractor)

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        return VGGTHouseFullTokenObsAdapter(extractor)


class VGGTHouseGlobalTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + singleton global-token Transformer, no gate."""

    variant = VGGT_VARIANTS["vggt_house_global_tokens_nogate"]

    def _build_adapter(self) -> ObsAdapter:
        return self._build_adapter_for_extractor(self._extractor)

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        return VGGTHouseGlobalTokenObsAdapter(extractor)


class VGGTWPCP64Encoder(VGGTEncoder):
    """WP+CP MLP at a finer 64x64 world-point grid."""

    variant = VGGT_VARIANTS["vggt_wp_cp_64"]


class VGGTWP64CNNCPMLPEncoder(VGGTEncoder):
    """64x64 world-point image through CNN plus camera-pose MLP (3D-89)."""

    variant = VGGT_VARIANTS["vggt_wp64_cnn_cp_mlp"]


__all__ = [
    "EncoderSpec",
    "VGGTVariantSpec",
    "VGGT_VARIANTS",
    "Encoder",
    "CNNEncoder",
    "VGGTEncoder",
    "VGGTAggregatorMLPEncoder",
    "VGGTAggTokenTransformerEncoder",
    "VGGTDenseWPEncoder",
    "HybridEncoder",
    "VGGTHouseContextEncoder",
    "VGGTHouseFullTokenNoGateEncoder",
    "VGGTHouseGlobalTokenNoGateEncoder",
    "VGGTWPCP64Encoder",
    "VGGTWP64CNNCPMLPEncoder",
]
