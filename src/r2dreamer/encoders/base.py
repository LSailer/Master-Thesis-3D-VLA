"""Launcher-side base classes for encoder selections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

import flax.linen as nn

from src.r2dreamer.encoders.cnn import ConvEncoder

if TYPE_CHECKING:
    from src.r2dreamer.adapters.obs_adapter import ObsAdapter
    from src.r2dreamer.adapters.vggt_adapter import VGGTFeatureKind


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


@dataclass(frozen=True)
class HouseContextTransformerConfig:
    """Launcher-side settings for the house-context token Transformer."""

    layers: int = 2
    heads: int = 8
    mlp_ratio: int = 2
    dropout: float = 0.0


class _LazyVGGTVariants(Mapping[str, Any]):
    """Lazy mapping to avoid importing Observation Preparation during package init."""

    @staticmethod
    def _data() -> Mapping[str, Any]:
        module = import_module("src.r2dreamer.observation_preparation.vggt")
        return module.VGGT_DREAMER_SPECS

    def __getitem__(self, key: str) -> Any:
        """Return the loaded variant for ``key``."""
        return self._data()[key]

    def __iter__(self):
        """Iterate over loaded variant keys."""
        return iter(self._data())

    def __len__(self) -> int:
        """Return the number of loaded variants."""
        return len(self._data())


class _VariantDescriptor:
    """Class/instance descriptor resolving a VGGT variant by ``variant_key``."""

    @staticmethod
    def resolve(owner: type) -> Any:
        """Return the variant configured on ``owner``."""
        return VGGT_VARIANTS[owner.variant_key]

    def __get__(self, instance: Any, owner: type) -> Any:
        return self.resolve(owner)


VGGT_VARIANTS: Mapping[str, Any] = _LazyVGGTVariants()


class Encoder:
    """Base launcher-side input mode."""

    encoder_type: str = ""
    module_cls: type[nn.Module] | None = None
    env_render_resolution: int = 64
    agent_overrides: Mapping[str, Any] = {}
    design_notes: str = ""

    _adapter: ObsAdapter | None = None

    @classmethod
    def from_train_args(cls, _args: Any) -> Encoder:
        """Build this encoder selection from parsed training arguments."""
        return cls()

    def make_adapter(self) -> ObsAdapter:
        """Return a cached observation adapter for environment interaction."""
        if self._adapter is None:
            self._adapter = self._build_adapter()
        return self._adapter

    def new_adapter(self) -> ObsAdapter:
        """Build a fresh observation adapter with independent mutable state."""
        return self._build_adapter()

    def _build_adapter(self) -> ObsAdapter:
        raise NotImplementedError(f"{type(self).__name__} must build an adapter")

    def spec(self) -> EncoderSpec:
        """Resolve the full launcher/agent contract for this encoder selection."""
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
    module_cls = ConvEncoder

    def _build_adapter(self) -> ObsAdapter:
        module = import_module("src.r2dreamer.observation_preparation")
        return module.CNNObservationPreparation()


class VGGTEncoder(Encoder):
    """External feature extractor -> configured VGGT readout."""

    # Match the InfiniteVGGT default token space (streamvggt.py: total_budget=1_200_000)
    # for a fair comparison against the reference extractor. Uniform split gives
    # 50_000 tokens/block (24 blocks); per-frame tokens/block P=1374, so the
    # streaming window is ~36 frames/block (vs ~6 at 200_000). Cost is ~6x
    # aggregator KV memory — watch eviction stability over long horizons.
    VGGT_TOTAL_BUDGET = 1_200_000
    VGGT_STATIC_BUDGETS = tuple([50_000] * 24)

    variant_key = "vggt"
    variant = _VariantDescriptor()

    @property
    def feature_kind(self) -> VGGTFeatureKind:
        """Return the extractor readout kind requested by this variant."""
        return self.variant.feature_kind

    @property
    def encoder_type(self) -> str:
        """Return the agent encoder type string."""
        return self.variant.encoder_type

    @property
    def module_cls(self) -> type[nn.Module]:
        """Return the low-level Flax encoder module class."""
        return self.variant.module_cls

    @property
    def agent_overrides(self) -> Mapping[str, Any]:
        """Return config overrides implied by this encoder variant."""
        return self.variant.agent_overrides

    @property
    def design_notes(self) -> str:
        """Return human-readable design notes for manifests."""
        return self.variant.design_notes

    @property
    def vggt_compute_heads(self) -> bool:
        """Return whether the VGGT point/camera heads must run."""
        return self.variant.compute_heads

    @property
    def wp_pool_size(self) -> int:
        """Return the VGGT world-point pooling side length."""
        return self.variant.wp_pool_size

    @classmethod
    def from_train_args(cls, args: Any) -> VGGTEncoder:
        """Build a VGGT encoder selection from parsed training arguments."""
        return cls(resolution=args.render_resolution)

    def __init__(self, resolution: int = 518, *, build_extractor: bool = True):
        self.env_render_resolution = resolution
        self._extractor = self._make_extractor() if build_extractor else None

    def _make_extractor(self):
        extractor_module = import_module("src.vggt.jax.feature_extractor")
        return extractor_module.JAXVGGTFeatureExtractor(
            total_budget=self.VGGT_TOTAL_BUDGET,
            budgets_static=self.VGGT_STATIC_BUDGETS,
            compute_heads=self.vggt_compute_heads,
            wp_pool_size=self.wp_pool_size,
        )

    def _build_adapter(self) -> ObsAdapter:
        return self._build_adapter_for_extractor(self._extractor)

    def new_adapter(self) -> ObsAdapter:
        """Build a fresh VGGT adapter and extractor instance."""
        return self._build_adapter_for_extractor(self._make_extractor())

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        adapter_module = import_module("src.r2dreamer.adapters.vggt_adapter")
        return adapter_module.VGGTObsAdapter(
            extractor,
            feature_kind=self.feature_kind,
            env_render_resolution=self.env_render_resolution,
            encoder_type=self.encoder_type,
            encoder_module_cls=self.module_cls,
            agent_overrides=self.agent_overrides,
            design_notes=self.design_notes,
        )


def variant_encoder_class(name: str, key: str) -> type[VGGTEncoder]:
    """Build a small VGGT variant subclass."""
    cls = type(name, (VGGTEncoder,), {"variant_key": key})
    cls.__module__ = "src.r2dreamer.encoders"
    return cls
