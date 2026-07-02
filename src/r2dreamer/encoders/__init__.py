"""Launcher-side encoder specs and adapter construction."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from src.r2dreamer.encoders.base import (
    CNNEncoder,
    ConvEncoder,
    Encoder,
    EncoderSpec,
    HouseContextTransformerConfig,
    VGGTEncoder,
    VGGT_VARIANTS,
    variant_encoder_class,
)
from src.r2dreamer.encoders.constants import AGG_TOKEN_TOKENS, HOUSE_CONTEXT_DIM
from src.r2dreamer.encoders.house_points_pose import VGGTHousePointsPoseEncoder
from src.r2dreamer.encoders.pointnet2 import (
    PointNet2BackboneOutput,
    PointNet2Encoder,
    PointNet2FeatureEncoder,
    PointNet2PipelineSpec,
)
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder

if TYPE_CHECKING:
    from src.r2dreamer.adapters.obs_adapter import ObsAdapter


VGGTAggregatorMLPEncoder = variant_encoder_class(
    "VGGTAggregatorMLPEncoder", "vggt_aggregator_mlp"
)
VGGTAggTokenTransformerEncoder = variant_encoder_class(
    "VGGTAggTokenTransformerEncoder", "vggt_agg_token_transformer"
)
VGGTDenseWPEncoder = variant_encoder_class("VGGTDenseWPEncoder", "vggt_wp_dense_cnn")
VGGTWPCP64Encoder = variant_encoder_class("VGGTWPCP64Encoder", "vggt_wp_cp_64")
VGGTWP64CNNCPMLPEncoder = variant_encoder_class(
    "VGGTWP64CNNCPMLPEncoder", "vggt_wp64_cnn_cp_mlp"
)


class HybridEncoder(VGGTEncoder):
    """CNN(RGB 64) + gated MLP(WP+CP 4116)."""

    variant_key = "hybrid"

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        adapter_module = import_module("src.r2dreamer.adapters.hybrid_adapter")
        return adapter_module.HybridObsAdapter(
            extractor,
            env_render_resolution=self.env_render_resolution,
            encoder_module_cls=self.module_cls,
            agent_overrides=self.agent_overrides,
            design_notes=self.design_notes,
        )


class VGGTHouseContextEncoder(VGGTEncoder):
    """L1 RGB replay + live bounded InfiniteVGGT house-context readout."""

    variant_key = "vggt_house_context"

    @classmethod
    def from_train_args(cls, args: Any) -> VGGTHouseContextEncoder:
        """Build a house-context encoder selection from parsed train args."""
        static_context_path = (
            getattr(args, "static_house_context_path", None)
            if cls is VGGTHouseContextEncoder
            else None
        )
        return cls(
            resolution=args.render_resolution,
            transformer_config=HouseContextTransformerConfig(
                layers=args.vggt_token_transformer_layers or 2,
                heads=args.vggt_token_transformer_heads or 8,
                mlp_ratio=args.vggt_token_transformer_mlp_ratio or 2,
                dropout=args.vggt_token_transformer_dropout or 0.0,
            ),
            static_house_context_path=static_context_path,
        )

    def __init__(
        self,
        resolution: int = 518,
        *,
        transformer_config: HouseContextTransformerConfig | None = None,
        static_house_context_path: str | None = None,
    ):
        self._static_house_context_path = static_house_context_path
        super().__init__(resolution, build_extractor=static_house_context_path is None)
        config = transformer_config or HouseContextTransformerConfig()
        self._context_transformer = TokenTransformerEncoder(
            embed_dim=HOUSE_CONTEXT_DIM,
            token_dim=2048,
            num_tokens=AGG_TOKEN_TOKENS,
            model_dim=None,
            layers=config.layers,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            readout="mean",
            norm_kind="layer",
            activation="gelu",
        )

    @property
    def agent_overrides(self) -> Mapping[str, Any]:
        """Return variant overrides plus context Transformer settings."""
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

    @property
    def design_notes(self) -> str:
        """Return design notes for live or static house-context mode."""
        if self._static_house_context_path is not None:
            return "RGB replay plus deterministic static RGB point-cloud house context."
        return self.variant.design_notes

    def new_adapter(self) -> ObsAdapter:
        """Build a fresh house-context adapter with independent mutable state."""
        if self._static_house_context_path is not None:
            return self._build_adapter_for_extractor(None)
        return super().new_adapter()

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        adapter_module = import_module("src.r2dreamer.adapters.hybrid_adapter")
        return adapter_module.VGGTHouseContextObsAdapter(
            extractor,
            context_transformer=self._context_transformer,
            static_house_context_path=self._static_house_context_path,
        )


class VGGTHouseFullTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + live full-token Transformer inside the agent, no gate."""

    variant_key = "vggt_house_full_tokens_nogate"

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        adapter_module = import_module("src.r2dreamer.adapters.hybrid_adapter")
        return adapter_module.VGGTHouseFullTokenObsAdapter(extractor)


class VGGTHouseGlobalTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + singleton global-token Transformer, no gate."""

    variant_key = "vggt_house_global_tokens_nogate"

    def _build_adapter_for_extractor(self, extractor) -> ObsAdapter:
        adapter_module = import_module("src.r2dreamer.adapters.hybrid_adapter")
        return adapter_module.VGGTHouseGlobalTokenObsAdapter(extractor)
