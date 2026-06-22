"""Launcher-side encoder specs and adapter construction."""

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
    """Full description of one encoder choice."""

    obs_shape: tuple[int, ...] | Mapping[str, tuple[int, ...]]
    env_render_resolution: int
    encoder_type: str
    module_cls: type[nn.Module]
    agent_overrides: dict[str, Any] = field(default_factory=dict)
    design_notes: str = ""
    contract_snapshot: dict[str, Any] | None = None


VGGTVariantSpec = VGGTDreamerSpec
VGGT_VARIANTS = VGGT_DREAMER_SPECS


class Encoder:
    """Base launcher-side input mode."""

    encoder_type: str = ""
    module_cls: type[nn.Module] | None = None
    env_render_resolution: int = 64
    agent_overrides: Mapping[str, Any] = {}
    design_notes: str = ""

    _adapter: ObsAdapter | None = None

    @classmethod
    def from_train_args(cls, _args: Any) -> "Encoder":
        return cls()

    def make_adapter(self) -> ObsAdapter:
        if self._adapter is None:
            self._adapter = self._build_adapter()
        return self._adapter

    def new_adapter(self) -> ObsAdapter:
        return self._build_adapter()

    def _build_adapter(self) -> ObsAdapter:
        raise NotImplementedError(f"{type(self).__name__} must build an adapter")

    def spec(self) -> EncoderSpec:
        if self.module_cls is None:
            raise NotImplementedError(f"{type(self).__name__} must set module_cls")
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
    """External feature extractor -> configured VGGT readout."""

    VGGT_TOTAL_BUDGET = 200_000
    VGGT_STATIC_BUDGETS = tuple([8333] * 24)

    variant = VGGT_VARIANTS["vggt"]

    @property
    def feature_kind(self) -> VGGTFeatureKind:
        return self.variant.feature_kind

    @property
    def encoder_type(self) -> str:
        return self.variant.encoder_type

    @property
    def module_cls(self) -> type[nn.Module]:
        return self.variant.module_cls

    @property
    def agent_overrides(self) -> Mapping[str, Any]:
        return self.variant.agent_overrides

    @property
    def design_notes(self) -> str:
        return self.variant.design_notes

    @property
    def vggt_compute_heads(self) -> bool:
        return self.variant.compute_heads

    @property
    def wp_pool_size(self) -> int:
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
        )

    def _build_adapter(self) -> ObsAdapter:
        return self._build_adapter_for_extractor(self._extractor)

    def new_adapter(self) -> ObsAdapter:
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


def _variant_encoder_class(name: str, key: str) -> type[VGGTEncoder]:
    cls = type(name, (VGGTEncoder,), {"variant": VGGT_VARIANTS[key]})
    cls.__module__ = __name__
    return cls


VGGTAggregatorMLPEncoder = _variant_encoder_class(
    "VGGTAggregatorMLPEncoder", "vggt_aggregator_mlp"
)
VGGTAggTokenTransformerEncoder = _variant_encoder_class(
    "VGGTAggTokenTransformerEncoder", "vggt_agg_token_transformer"
)
VGGTDenseWPEncoder = _variant_encoder_class("VGGTDenseWPEncoder", "vggt_wp_dense_cnn")
VGGTWPCP64Encoder = _variant_encoder_class("VGGTWPCP64Encoder", "vggt_wp_cp_64")
VGGTWP64CNNCPMLPEncoder = _variant_encoder_class(
    "VGGTWP64CNNCPMLPEncoder", "vggt_wp64_cnn_cp_mlp"
)


class HybridEncoder(VGGTEncoder):
    """CNN(RGB 64) + gated MLP(WP+CP 4116)."""

    variant = VGGT_VARIANTS["hybrid"]

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        return HybridObsAdapter(
            extractor,
            env_render_resolution=self.env_render_resolution,
            encoder_module_cls=self.module_cls,
            agent_overrides=self.agent_overrides,
        )


class VGGTHouseContextEncoder(VGGTEncoder):
    """L1 RGB replay + live bounded InfiniteVGGT house-context readout."""

    variant = VGGT_VARIANTS["vggt_house_context"]

    @classmethod
    def from_train_args(cls, args: Any) -> "VGGTHouseContextEncoder":
        return cls(
            resolution=args.render_resolution,
            transformer_layers=args.vggt_token_transformer_layers or 2,
            transformer_heads=args.vggt_token_transformer_heads or 8,
            transformer_mlp_ratio=args.vggt_token_transformer_mlp_ratio or 2,
            transformer_dropout=args.vggt_token_transformer_dropout or 0.0,
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

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        return VGGTHouseContextObsAdapter(
            extractor,
            context_transformer=self._context_transformer,
            contract_options={
                "env_render_resolution": self.env_render_resolution,
                "encoder_module_cls": self.module_cls,
                "agent_overrides": self.agent_overrides,
            },
        )


class VGGTHouseFullTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + live full-token Transformer inside the agent, no gate."""

    variant = VGGT_VARIANTS["vggt_house_full_tokens_nogate"]

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        return VGGTHouseFullTokenObsAdapter(
            extractor,
            env_render_resolution=self.env_render_resolution,
            encoder_module_cls=self.module_cls,
            agent_overrides=self.agent_overrides,
        )


class VGGTHouseGlobalTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + singleton global-token Transformer, no gate."""

    variant = VGGT_VARIANTS["vggt_house_global_tokens_nogate"]

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        return VGGTHouseGlobalTokenObsAdapter(
            extractor,
            env_render_resolution=self.env_render_resolution,
            encoder_module_cls=self.module_cls,
            agent_overrides=self.agent_overrides,
        )
