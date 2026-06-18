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

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.vggt_adapter import VGGTFeatureKind, VGGTObsAdapter
from src.r2dreamer.observation_preparation import CNNObservationPreparation
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


@dataclass(frozen=True)
class VGGTVariantSpec:
    """Shared launcher metadata for a concrete VGGT readout variant."""

    encoder_type: str
    feature_kind: VGGTFeatureKind
    module_cls: type[nn.Module]
    compute_heads: bool
    wp_pool_size: int = 37
    agent_overrides: Mapping[str, Any] = field(default_factory=dict)
    design_notes: str = ""


def _vggt_variant(
    *,
    encoder_type: str,
    feature_kind: VGGTFeatureKind,
    module_cls: type[nn.Module],
    compute_heads: bool,
    wp_pool_size: int = 37,
    agent_overrides: Mapping[str, Any] | None = None,
    design_notes: str = "",
) -> VGGTVariantSpec:
    return VGGTVariantSpec(
        encoder_type=encoder_type,
        feature_kind=feature_kind,
        module_cls=module_cls,
        compute_heads=compute_heads,
        wp_pool_size=wp_pool_size,
        agent_overrides=MappingProxyType(dict(agent_overrides or {})),
        design_notes=design_notes,
    )


VGGT_VARIANTS: dict[str, VGGTVariantSpec] = {
    "vggt": _vggt_variant(
        encoder_type="vggt",
        feature_kind="wp_cp",
        module_cls=wm_encoders.VGGTEncoder,
        compute_heads=True,
        agent_overrides={"buffer_capacity": 1_000_000},
    ),
    "vggt_aggregator_mlp": _vggt_variant(
        encoder_type="vggt_aggregator_mlp",
        feature_kind="aggregator",
        module_cls=wm_encoders.VGGTAggregatorMLPEncoder,
        compute_heads=False,
        agent_overrides={
            "buffer_capacity": 5_000,
            "batch_size": 4,
            "seq_len": 32,
            "train_ratio": 128,
        },
        design_notes=(
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
        ),
    ),
    "vggt_agg_token_transformer": _vggt_variant(
        encoder_type="vggt_agg_token_transformer",
        feature_kind="agg_tokens",
        module_cls=wm_encoders.VGGTAggTokenTransformerEncoder,
        compute_heads=False,
        agent_overrides={
            "buffer_capacity": 5_000,
            "batch_size": 1,
            "seq_len": 8,
            "train_ratio": 32,
        },
        design_notes=(
            "3D-75 token-preserving VGGT aggregator encoder. The frozen JAX VGGT "
            "extractor runs headless (compute_heads=False) and emits full global "
            "aggregator tokens: 1 camera + 4 register + 37x37 patch tokens = "
            "(1374, 1024). The adapter stores the flattened full-token sequence "
            "as float16 in replay; the trainable Flax Token Transformer upcasts to "
            "float32, keeps register tokens by default, projects each token before "
            "self-attention, pools the camera/register/patch outputs, and returns "
            "cfg.vggt_embed_dim for the unchanged R2RSSM.observe() path. The small "
            "default batch/sequence overrides are intentional because attention "
            "runs over all 1374 tokens for every sampled replay position."
        ),
    ),
    "vggt_wp_dense_cnn": _vggt_variant(
        encoder_type="vggt_wp_dense_cnn",
        feature_kind="wp_dense",
        module_cls=wm_encoders.ConvEncoder,
        compute_heads=True,
        agent_overrides={
            "buffer_capacity": 5_000,
            "batch_size": 4,
            "seq_len": 32,
            "train_ratio": 128,
        },
        design_notes=(
            "Variant: full-resolution VGGT world points. The DPT point head's dense "
            "518x518x3 per-pixel map (one metric XYZ point per pixel) is NOT pooled "
            "to 37x37; it is stored channel-first as a (3, 518, 518) float16 image "
            "and fed to ConvEncoder(input_kind='world_points'), which symlog-normalises the metric XYZ range "
            "and runs the RGB Conv+MaxPool+RMSNorm+SiLU stack before a linear "
            "readout to embed_dim. WP-only (no camera pose): a 9-vector cannot be a "
            "spatial channel. See issue 3D-53."
        ),
    ),
    "hybrid": _vggt_variant(
        encoder_type="hybrid",
        feature_kind="wp_cp",
        module_cls=wm_encoders.HybridEncoder,
        compute_heads=True,
        agent_overrides={"buffer_capacity": 100_000},
        design_notes=(
            "Hybrid encoder: a CNN branch over the 64x64 RGB frame and a gated MLP "
            "branch over the 4116-dim VGGT world_points+camera_pose vector are fused "
            "(concatenated) into the latent. A zero-init scalar gate on the WP/CP "
            "branch means training starts as plain CNN-Dreamer and only blends in the "
            "geometric features as the gate opens; per-branch contributions are logged "
            "as hybrid/* metrics (3D-50/51/52)."
        ),
    ),
    "vggt_house_context": _vggt_variant(
        encoder_type="vggt_house_context",
        feature_kind="agg_tokens",
        module_cls=wm_encoders.HybridEncoder,
        compute_heads=False,
        agent_overrides={
            "buffer_capacity": 1_000_000,
            "vggt_feature_dim": wm_encoders.HOUSE_CONTEXT_DIM,
            "vggt_token_dim": 2048,
            "vggt_token_count": wm_encoders.AGG_TOKEN_TOKENS,
        },
        design_notes=(
            "L1 house-context variant: replay stores only the 64x64 RGB frame, "
            "while a live bounded InfiniteVGGT stream remains active across "
            "episode resets and exposes the full 1374x2048 frozen aggregator "
            "tokens. A 2048-wide token Transformer consumes those tokens directly, "
            "projects the resulting context to 1024, and injects that cached "
            "context at acting and training time. Replay remains RGB-only and the "
            "existing hybrid gate fuses RGB 1024 + VGGT context 1024."
        ),
    ),
    "vggt_house_full_tokens_nogate": _vggt_variant(
        encoder_type="vggt_house_full_tokens_nogate",
        feature_kind="agg_tokens",
        module_cls=wm_encoders.RGBFullTokenTransformerEncoder,
        compute_heads=False,
        agent_overrides={
            "buffer_capacity": 1_000_000,
            "vggt_token_dim": 2048,
            "vggt_token_count": wm_encoders.AGG_TOKEN_TOKENS,
        },
        design_notes=(
            "L1 no-gate full-token variant: replay stores only RGB64 while a live "
            "bounded InfiniteVGGT stream exposes the full 1374x2048 aggregator "
            "tokens. The trainable agent encoder consumes image+full_tokens as "
            "separate fields, runs CNN(image) and a full-token Transformer, then "
            "concatenates them directly without the hybrid scalar gate or WP/CP MLP."
        ),
    ),
    "vggt_house_global_tokens_nogate": _vggt_variant(
        encoder_type="vggt_house_global_tokens_nogate",
        feature_kind="agg_tokens",
        module_cls=wm_encoders.RGBGlobalTokenTransformerEncoder,
        compute_heads=False,
        agent_overrides={
            "buffer_capacity": 1_000_000,
            "vggt_token_dim": 1024,
            "vggt_token_count": wm_encoders.AGG_TOKEN_TOKENS,
        },
        design_notes=(
            "3D-90 no-gate global-token variant: replay stores only RGB64 while a "
            "live bounded InfiniteVGGT stream exposes the 1374x1024 global-half "
            "aggregator tokens. The trainable agent encoder computes one token "
            "Transformer context from the singleton live token sequence per train "
            "step, broadcasts that embedding across sampled RGB rows, and "
            "concatenates without the hybrid scalar gate."
        ),
    ),
    "vggt_wp_cp_64": _vggt_variant(
        encoder_type="vggt_wp_cp_64",
        feature_kind="wp_cp",
        module_cls=wm_encoders.VGGTEncoder,
        compute_heads=True,
        wp_pool_size=64,
        agent_overrides={"buffer_capacity": 1_000_000},
        design_notes=(
            "WP+CP MLP at a 64x64 world-point grid. VGGT's dense 518x518x3 point map "
            "is average-pooled (antialiased area resample, since 518 is not divisible "
            "by 64) to 64x64x3, flattened (12288) and concatenated with the 9-D camera "
            "pose into a 12297-D observation, then encoded by the same multi-layer MLP "
            "as the 37x37 WP+CP variant. Only the WP resolution differs (37 -> 64): a "
            "controlled resolution ablation, at the RGB-CNN baseline resolution."
        ),
    ),
    "vggt_wp64_cnn_cp_mlp": _vggt_variant(
        encoder_type="vggt_wp64_cnn_cp_mlp",
        feature_kind="wp64_cp",
        module_cls=wm_encoders.WP64CNNCPMLPEncoder,
        compute_heads=True,
        wp_pool_size=64,
        agent_overrides={"buffer_capacity": 1_000_000},
        design_notes=(
            "3D-89 hypothesis encoder: VGGT world points are pooled to a 64x64x3 "
            "metric XYZ image and stored separately from the 9-D camera pose, both "
            "as float16 replay fields. The trainable encoder applies the point-cloud "
            "ConvEncoder path to world points and a small MLP to camera pose before fusion."
        ),
    ),
}


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
        self._extractor = VGGTFeatureExtractor(
            total_budget=self.VGGT_TOTAL_BUDGET,
            budgets_static=self.VGGT_STATIC_BUDGETS,
            compute_heads=self.vggt_compute_heads,
            wp_pool_size=self.wp_pool_size,
        )  # device="cuda" default

    def _build_adapter(self) -> ObsAdapter:
        return VGGTObsAdapter(
            self._extractor,
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
        from src.r2dreamer.adapters.hybrid_adapter import HybridObsAdapter

        return HybridObsAdapter(
            self._extractor,
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
        from src.r2dreamer.adapters.hybrid_adapter import VGGTHouseContextObsAdapter

        return VGGTHouseContextObsAdapter(
            self._extractor,
            context_transformer=self._context_transformer,
        )


class VGGTHouseFullTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + live full-token Transformer inside the agent, no gate."""

    variant = VGGT_VARIANTS["vggt_house_full_tokens_nogate"]

    def _build_adapter(self) -> ObsAdapter:
        from src.r2dreamer.adapters.hybrid_adapter import VGGTHouseFullTokenObsAdapter

        return VGGTHouseFullTokenObsAdapter(self._extractor)


class VGGTHouseGlobalTokenNoGateEncoder(VGGTHouseContextEncoder):
    """L1 RGB replay + singleton global-token Transformer, no gate."""

    variant = VGGT_VARIANTS["vggt_house_global_tokens_nogate"]

    def _build_adapter(self) -> ObsAdapter:
        from src.r2dreamer.adapters.hybrid_adapter import VGGTHouseGlobalTokenObsAdapter

        return VGGTHouseGlobalTokenObsAdapter(self._extractor)


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
