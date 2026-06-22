"""Launcher-side encoder specs and adapter construction."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, cast

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


def _instantiate_adapter(
    adapter_cls: Callable[..., ObsAdapter], *args: Any, **kwargs: Any
) -> ObsAdapter:
    return adapter_cls(*args, **kwargs)


class Encoder:
    """Base launcher-side input mode."""

    encoder_type: str = ""
    module_cls: type[nn.Module] | None = None
    adapter_cls: Callable[..., ObsAdapter] | None = None
    env_render_resolution: int = 64
    agent_overrides: Mapping[str, Any] = {}
    design_notes: str = ""

    _adapter: ObsAdapter | None = None

    @classmethod
    def from_train_args(cls, _args: Any) -> "Encoder":
        return cls()

    def make_adapter(self) -> ObsAdapter:
        if self._adapter is None:
            self._adapter = self.new_adapter()
        return self._adapter

    def new_adapter(self) -> ObsAdapter:
        adapter_cls = self.adapter_cls
        if adapter_cls is None:
            raise NotImplementedError(f"{type(self).__name__} must set adapter_cls")
        adapter_factory = cast(Callable[..., ObsAdapter], adapter_cls)
        return _instantiate_adapter(adapter_factory)

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
    adapter_cls = CNNObservationPreparation


class VGGTEncoder(Encoder):
    """External feature extractor -> configured VGGT readout."""

    VGGT_TOTAL_BUDGET = 200_000
    VGGT_STATIC_BUDGETS = tuple([8333] * 24)

    variant = VGGT_VARIANTS["vggt"]
    adapter_cls = VGGTObsAdapter

    @classmethod
    def from_train_args(cls, args: Any) -> "VGGTEncoder":
        return cls(resolution=args.render_resolution)

    def __init__(self, resolution: int = 518):
        self.env_render_resolution = resolution
        self._extractor = VGGTFeatureExtractor(
            total_budget=self.VGGT_TOTAL_BUDGET,
            budgets_static=self.VGGT_STATIC_BUDGETS,
            compute_heads=self.variant.compute_heads,
            wp_pool_size=self.variant.wp_pool_size,
        )

    def make_adapter(self) -> ObsAdapter:
        if self._adapter is None:
            adapter_cls = self.adapter_cls
            if adapter_cls is None:
                raise NotImplementedError(f"{type(self).__name__} must set adapter_cls")
            adapter_factory = cast(Callable[..., ObsAdapter], adapter_cls)
            self._adapter = _instantiate_adapter(
                adapter_factory,
                self._extractor,
                **self._adapter_kwargs(),
            )
        return self._adapter

    def new_adapter(self) -> ObsAdapter:
        extractor = VGGTFeatureExtractor(
            total_budget=self.VGGT_TOTAL_BUDGET,
            budgets_static=self.VGGT_STATIC_BUDGETS,
            compute_heads=self.variant.compute_heads,
            wp_pool_size=self.variant.wp_pool_size,
        )
        adapter_cls = self.adapter_cls
        if adapter_cls is None:
            raise NotImplementedError(f"{type(self).__name__} must set adapter_cls")
        adapter_factory = cast(Callable[..., ObsAdapter], adapter_cls)
        return _instantiate_adapter(
            adapter_factory,
            extractor,
            **self._adapter_kwargs(),
        )

    def _adapter_kwargs(self) -> dict[str, Any]:
        return {
            "feature_kind": self.variant.feature_kind,
            "env_render_resolution": self.env_render_resolution,
            "encoder_type": self.variant.name,
            "encoder_module_cls": self.variant.dreamer.module_cls,
            "agent_overrides": self.variant.agent_overrides,
            "design_notes": self.variant.design_notes,
        }

    def spec(self) -> EncoderSpec:
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
        overrides = (
            self.agent_overrides
            if isinstance(getattr(type(self), "agent_overrides", None), property)
            else self.variant.agent_overrides
        )
        return EncoderSpec(
            obs_shape=adapter.encoder_obs_shape,
            env_render_resolution=self.env_render_resolution,
            encoder_type=self.variant.name,
            module_cls=self.variant.dreamer.module_cls,
            agent_overrides=dict(overrides),
            design_notes=self.variant.design_notes,
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
    adapter_cls = HybridObsAdapter

    def _adapter_kwargs(self) -> dict[str, Any]:
        return {
            "env_render_resolution": self.env_render_resolution,
            "encoder_module_cls": self.variant.dreamer.module_cls,
            "agent_overrides": self.variant.agent_overrides,
            "design_notes": self.variant.design_notes,
        }


class VGGTHouseContextEncoder(VGGTEncoder):
    """L1 RGB replay + live bounded InfiniteVGGT house-context readout."""

    variant = VGGT_VARIANTS["vggt_house_context"]
    adapter_cls = VGGTHouseContextObsAdapter

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

    def _adapter_kwargs(self) -> dict[str, Any]:
        return {"context_transformer": self._context_transformer}


class VGGTHouseFullTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + live full-token Transformer inside the agent, no gate."""

    variant = VGGT_VARIANTS["vggt_house_full_tokens_nogate"]
    adapter_cls = VGGTHouseFullTokenObsAdapter

    def _adapter_kwargs(self) -> dict[str, Any]:
        return {}


class VGGTHouseGlobalTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + singleton global-token Transformer, no gate."""

    variant = VGGT_VARIANTS["vggt_house_global_tokens_nogate"]
    adapter_cls = VGGTHouseGlobalTokenObsAdapter

    def _adapter_kwargs(self) -> dict[str, Any]:
        return {}
