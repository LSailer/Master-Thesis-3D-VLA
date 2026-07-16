"""Launcher-side base classes for encoder selections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

import flax.linen as nn

from src.r2dreamer.encoders.cnn import ConvEncoder
from src.r2dreamer.observation_keys import (
    FULL_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HYBRID_IMAGE_KEY,
)
from src.vggt.jax.feature_extractor import ResetMode

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


def _vggt_module_kwargs(module_cls: type, config: Any) -> dict[str, Any]:
    """Resolve constructor kwargs for a VGGT-variant Encoder Module.

    Dispatch is by module-class name (not ``issubclass``), so this stays free
    of the encoder-module imports at module load; the bare VGGT variant
    subclasses whose module is an ``MLPEncoder`` subclass (e.g.
    ``VGGTAggregatorMLPEncoder``) fall through to the MLP tail. Mirrors the
    former ``contracts.py`` table verbatim so durable snapshots are unchanged.

    ``compute_dtype`` is intentionally NOT emitted here: it is a JAX dtype
    (not JSON-serializable) and so cannot enter the durable contract
    snapshot. The factory adds it as a runtime-only overlay.

    Args:
      module_cls: The Flax Encoder Module class resolved from the variant.
      config: Effective agent config supplying the encoder knob values.

    Returns:
      Constructor kwargs for ``module_cls`` (no ``compute_dtype``).
    """
    name = module_cls.__name__
    encoder_type = getattr(config, "encoder_type", None)
    if name == "ConvEncoder":
        kwargs = {
            "depth": int(config.encoder_depth),
            "kernel_size": int(config.encoder_kernel),
            "mults": tuple(config.encoder_mults),
        }
        if encoder_type == "vggt_wp_dense_cnn":
            kwargs.update(
                input_kind="world_points",
                embed_dim=int(config.vggt_embed_dim),
            )
        return kwargs
    if name == "WP64CNNCPMLPEncoder":
        return {
            "embed_dim": int(config.vggt_embed_dim),
            "conv_depth": int(config.encoder_depth),
            "conv_kernel": int(config.encoder_kernel),
            "conv_mults": tuple(config.encoder_mults),
            "cp_hidden": int(config.mlp_vggt_hidden),
            "cp_layers": int(config.mlp_vggt_layers),
        }
    if name == "HybridEncoder":
        return {
            "cnn_depth": int(config.encoder_depth),
            "cnn_kernel": int(config.encoder_kernel),
            "cnn_mults": tuple(config.encoder_mults),
            "vggt_embed_dim": int(config.vggt_embed_dim),
            "mlp_hidden": int(config.mlp_vggt_hidden),
            "mlp_layers": int(config.mlp_vggt_layers),
            "vggt_dim": int(config.vggt_feature_dim),
        }
    if name == "TokenTransformerEncoder":
        common = {
            "embed_dim": int(config.vggt_embed_dim),
            "token_dim": int(config.vggt_token_dim),
            "num_tokens": int(config.vggt_token_count),
            "layers": int(config.vggt_token_transformer_layers),
            "heads": int(config.vggt_token_transformer_heads),
            "mlp_ratio": int(config.vggt_token_transformer_mlp_ratio),
            "dropout": float(config.vggt_token_transformer_dropout),
        }
        if encoder_type == "vggt_agg_token_transformer":
            return {
                **common,
                "model_dim": int(config.vggt_token_projection_dim),
                "readout": "camera_register_patch",
                "norm_kind": "rms",
                "activation": "silu",
                "keep_register_tokens": bool(config.vggt_keep_register_tokens),
            }
        token_key = FULL_TOKENS_KEY
        singleton_tokens = False
        if encoder_type == "vggt_house_global_tokens_nogate":
            token_key = GLOBAL_TOKENS_KEY
            singleton_tokens = True
        return {
            **common,
            "model_dim": None,
            "readout": "mean",
            "norm_kind": "layer",
            "activation": "gelu",
            "token_key": token_key,
            "image_key": HYBRID_IMAGE_KEY,
            "singleton_tokens": singleton_tokens,
            "cnn_depth": int(config.encoder_depth),
            "cnn_kernel": int(config.encoder_kernel),
            "cnn_mults": tuple(config.encoder_mults),
        }
    # MLP tail: MLPEncoder, VGGTAggregatorMLPEncoder, VGGTAggRawMLPEncoder.
    return {
        "embed_dim": int(config.vggt_embed_dim),
        "hidden": int(config.vggt_embed_dim),
        "num_layers": int(config.vggt_mlp_layers),
    }


class Encoder:
    """Base launcher-side input mode."""

    _encoder_type: ClassVar[str] = ""
    _module_cls: ClassVar[type[nn.Module] | None] = None
    _agent_overrides: ClassVar[Mapping[str, Any]] = {}
    _design_notes: ClassVar[str] = ""
    env_render_resolution: int = 64

    _adapter: ObsAdapter | None = None

    @property
    def encoder_type(self) -> str:
        """Return the agent encoder type string."""
        return type(self)._encoder_type

    @property
    def module_cls(self) -> type[nn.Module] | None:
        """Return the low-level Flax encoder module class."""
        return type(self)._module_cls

    @property
    def agent_overrides(self) -> Mapping[str, Any]:
        """Return config overrides implied by this encoder selection."""
        return type(self)._agent_overrides

    @property
    def design_notes(self) -> str:
        """Return human-readable design notes for manifests."""
        return type(self)._design_notes

    @classmethod
    def module_kwargs_from_config(cls, config: Any) -> dict[str, Any]:
        """Resolve Encoder Module constructor kwargs from the effective config.

        Subclasses co-locate the config->kwargs formula next to their
        ``module_cls`` so the constructor signature and the resolved kwargs
        cannot desync (the structural fix for the Cause A drift). The base
        selection carries no module, so it has no formula to resolve.

        Args:
          config: Effective agent config supplying the encoder knob values.

        Returns:
          Constructor kwargs for ``cls.module_cls``.

        Raises:
          NotImplementedError: If the subclass does not set ``module_cls``.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement module_kwargs_from_config"
        )

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

    _encoder_type = "cnn"
    _module_cls = ConvEncoder

    @classmethod
    def module_kwargs_from_config(cls, config: Any) -> dict[str, Any]:
        """Resolve ConvEncoder kwargs (depth/kernel_size/mults) from config.

        The ``vggt_wp_dense_cnn`` world-point conv is a VGGT variant, not a
        CNN selection, so its ``input_kind``/``embed_dim`` extras live in the
        VGGT dispatch (:func:`_vggt_module_kwargs`) — not here.

        Args:
          config: Effective agent config supplying encoder depth/kernel/mults.

        Returns:
          Constructor kwargs for ``ConvEncoder``.
        """
        return {
            "depth": int(config.encoder_depth),
            "kernel_size": int(config.encoder_kernel),
            "mults": tuple(config.encoder_mults),
        }

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
    # VGGT streaming-cache reset policy at episode boundaries. ``FULL`` wipes
    # every episode (default, re-anchors each episode into its own world frame).
    # ``PERSIST_SCENE`` saves/restores the cache per ``scene_id`` so all episodes
    # of one house share one world frame — required for the live per-scene house
    # point buffer to accumulate geometrically-consistent points instead of
    # re-anchoring each episode (which produces ghost copies; see
    # docs/notes/visible-house-context-snapshot.md). Override on subclasses.
    vggt_reset_mode: ResetMode = ResetMode.FULL

    variant_key = "vggt"
    variant = _VariantDescriptor()

    @classmethod
    def module_kwargs_from_config(cls, config: Any) -> dict[str, Any]:
        """Resolve this variant's Encoder Module kwargs from the effective config.

        The variant identifies the Flax module class (via ``VGGT_VARIANTS``);
        the knob values come from ``config``. Dispatches by module-class name
        to the per-module formula (:func:`_vggt_module_kwargs`) — the single
        source of truth for VGGT-variant kwargs, co-located with the variant
        selection. Bare variant subclasses (``VGGTAggregatorMLPEncoder``,
        ``VGGTDenseWPEncoder``, the no-gate house-token encoders, etc.) inherit
        this unchanged; the standalone house encoders (points-pose, hybrid
        points-pose, global-embedding) override it with their own formula.

        Args:
          config: Effective agent config supplying the encoder knob values.

        Returns:
          Constructor kwargs for this variant's Encoder Module (no
          ``compute_dtype`` — that stays a factory-only overlay).
        """
        module_cls = VGGT_VARIANTS[cls.variant_key].module_cls
        return _vggt_module_kwargs(module_cls, config)

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
            reset_mode=self.vggt_reset_mode,
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
