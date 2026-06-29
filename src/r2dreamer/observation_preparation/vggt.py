"""VGGT-family Observation Preparation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Any, cast

import flax.linen as nn

from src.r2dreamer.config import (
    ObservationDims,
    ObservationRunConfig,
    ReplayObservationConfig,
)
from src.r2dreamer.obs_batch import (
    CAMERA_POSE_KEY,
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
    WORLD_POINTS_KEY,
)
from src.r2dreamer.observation_preparation.contracts import (
    EncoderInputContract,
    ObservationField,
    ObservationFormContract,
    replay_observation_form,
)
from src.r2dreamer.world_model import encoders as wm_encoders


VGGTFeatureKind = Literal[
    "wp_cp", "wp64_cp", "aggregator", "wp_dense", "agg_raw", "agg_tokens"
]
DEFAULT_OBSERVATION_DIMS = ObservationDims()
VGGT_IMAGE_SIZE = DEFAULT_OBSERVATION_DIMS.render_size
VGGT_IMAGE_SHAPE = DEFAULT_OBSERVATION_DIMS.render_shape
VGGT_DEFAULT_WP_POOL_SIZE = DEFAULT_OBSERVATION_DIMS.wp_side
VGGT_CAMERA_POSE_DIM = DEFAULT_OBSERVATION_DIMS.camera_pose_dim
VGGT_XYZ_CHANNELS = DEFAULT_OBSERVATION_DIMS.xyz_channels
VGGT_AGGREGATOR_POOL_COUNT = 3
VGGT_AGGREGATOR_PATCH_START_IDX = 5
VGGT_AGGREGATOR_TOKEN_COUNT = DEFAULT_OBSERVATION_DIMS.token_count
VGGT_AGGREGATOR_EMBED_DIM = DEFAULT_OBSERVATION_DIMS.token_dim
VGGT_FULL_TOKEN_EMBED_DIM = 2048
VGGT_DEFAULT_AGGREGATOR_SHAPE = (
    VGGT_AGGREGATOR_TOKEN_COUNT,
    VGGT_AGGREGATOR_EMBED_DIM,
)

HYBRID_IMAGE_SHAPE = DEFAULT_OBSERVATION_DIMS.image_shape
HYBRID_IMAGE_SIZE = HYBRID_IMAGE_SHAPE[-1]
HYBRID_RGB_DIM = HYBRID_IMAGE_SHAPE[0] * HYBRID_IMAGE_SHAPE[1] * HYBRID_IMAGE_SHAPE[2]


TokenSource = Literal["pooled", "flattened", "full", "global"]
DreamerEncoderKind = Literal["mlp", "cnn", "transformer", "linear", "hybrid"]
DreamerInputLayout = Literal[
    "flat_wp_cp",
    "structured_wp_cp",
    "world_points",
    "flat_features",
    "rgb_plus_flat",
    "rgb_plus_context",
    "rgb_plus_tokens",
]


@dataclass(frozen=True)
class HeadReadout:
    """VGGT head readout: world-points, optionally plus camera pose."""

    wp_side: int | Literal["dense"] = VGGT_DEFAULT_WP_POOL_SIZE
    include_camera_pose: bool = True
    kind: Literal["heads"] = "heads"


@dataclass(frozen=True)
class TokenReadout:
    """VGGT token readout before task-specific Dreamer projection."""

    token_source: TokenSource = "pooled"
    token_dim: int = VGGT_AGGREGATOR_EMBED_DIM
    num_tokens: int = VGGT_AGGREGATOR_TOKEN_COUNT
    keep_register_tokens: bool = True
    kind: Literal["tokens"] = "tokens"


VGGTReadout = HeadReadout | TokenReadout


@dataclass(frozen=True)
class StorageSpec:
    """Where RGB and the selected VGGT readout live relative to replay."""

    replay_rgb: bool
    replay_readout: bool
    readout_dtype: str = "float16"


@dataclass(frozen=True)
class DreamerEncoderSpec:
    """Dreamer-side encoder consuming the stored/live readout layout."""

    kind: DreamerEncoderKind
    module_cls: type[nn.Module]
    input_layout: DreamerInputLayout


@dataclass(frozen=True)
class VGGTDreamerSpec:
    """Single source for one VGGT-to-Dreamer integration."""

    name: str
    readout: VGGTReadout
    storage: StorageSpec
    dreamer: DreamerEncoderSpec
    agent_overrides: Mapping[str, Any] = field(default_factory=dict)
    design_notes: str = ""

    @property
    def encoder_type(self) -> str:
        """Agent encoder type string."""
        return self.name

    @property
    def feature_kind(self) -> VGGTFeatureKind:
        """Legacy adapter readout key derived from the readout/layout axes."""
        if isinstance(self.readout, HeadReadout):
            if self.readout.wp_side == "dense":
                return "wp_dense"
            if self.dreamer.input_layout == "structured_wp_cp":
                return "wp64_cp"
            return "wp_cp"
        return cast(
            VGGTFeatureKind,
            {
                "pooled": "aggregator",
                "flattened": "agg_raw",
                "full": "agg_tokens",
                "global": "agg_tokens",
            }[self.readout.token_source],
        )

    @property
    def module_cls(self) -> type[nn.Module]:
        """Flax encoder module class."""
        return self.dreamer.module_cls

    @property
    def compute_heads(self) -> bool:
        """Whether VGGT camera/point heads must run."""
        return isinstance(self.readout, HeadReadout)

    @property
    def wp_pool_size(self) -> int:
        """World-point pooling side passed to the VGGT extractor."""
        if isinstance(self.readout, HeadReadout) and isinstance(self.readout.wp_side, int):
            return self.readout.wp_side
        return VGGT_DEFAULT_WP_POOL_SIZE


_SMALL_REPLAY_OVERRIDES = {
    "buffer_capacity": 5_000,
    "batch_size": 4,
    "seq_len": 32,
    "train_ratio": 128,
}


VGGT_DREAMER_SPECS: dict[str, VGGTDreamerSpec] = {
    "vggt": VGGTDreamerSpec(
        name="vggt",
        readout=HeadReadout(37),
        storage=StorageSpec(replay_rgb=False, replay_readout=True),
        dreamer=DreamerEncoderSpec("mlp", wm_encoders.VGGTEncoder, "flat_wp_cp"),
        agent_overrides={"buffer_capacity": 1_000_000},
    ),
    "vggt_wp_cp_64": VGGTDreamerSpec(
        name="vggt_wp_cp_64",
        readout=HeadReadout(64),
        storage=StorageSpec(replay_rgb=False, replay_readout=True),
        dreamer=DreamerEncoderSpec("mlp", wm_encoders.VGGTEncoder, "flat_wp_cp"),
        agent_overrides={"buffer_capacity": 1_000_000},
        design_notes="WP/CP MLP with 64x64 pooled world points.",
    ),
    "vggt_wp64_cnn_cp_mlp": VGGTDreamerSpec(
        name="vggt_wp64_cnn_cp_mlp",
        readout=HeadReadout(64),
        storage=StorageSpec(replay_rgb=False, replay_readout=True),
        dreamer=DreamerEncoderSpec(
            "hybrid", wm_encoders.WP64CNNCPMLPEncoder, "structured_wp_cp"
        ),
        agent_overrides={"buffer_capacity": 1_000_000},
        design_notes="64x64 world-point CNN plus camera-pose MLP.",
    ),
    "vggt_wp_dense_cnn": VGGTDreamerSpec(
        name="vggt_wp_dense_cnn",
        readout=HeadReadout("dense"),
        storage=StorageSpec(replay_rgb=False, replay_readout=True),
        dreamer=DreamerEncoderSpec("cnn", wm_encoders.ConvEncoder, "world_points"),
        agent_overrides=_SMALL_REPLAY_OVERRIDES,
        design_notes="Dense 518x518 VGGT world-point map through a CNN encoder.",
    ),
    "vggt_aggregator_mlp": VGGTDreamerSpec(
        name="vggt_aggregator_mlp",
        readout=TokenReadout("pooled"),
        storage=StorageSpec(
            replay_rgb=False, replay_readout=True, readout_dtype="float32"
        ),
        dreamer=DreamerEncoderSpec(
            "mlp", wm_encoders.VGGTAggregatorMLPEncoder, "flat_features"
        ),
        agent_overrides=_SMALL_REPLAY_OVERRIDES,
        design_notes="Pooled [camera token, mean patches, max patches] VGGT tokens.",
    ),
    "vggt_agg_raw": VGGTDreamerSpec(
        name="vggt_agg_raw",
        readout=TokenReadout("flattened"),
        storage=StorageSpec(replay_rgb=False, replay_readout=True),
        dreamer=DreamerEncoderSpec(
            "mlp", wm_encoders.VGGTAggRawMLPEncoder, "flat_features"
        ),
        agent_overrides=_SMALL_REPLAY_OVERRIDES,
        design_notes="Flattened VGGT camera and patch tokens for an MLP.",
    ),
    "vggt_agg_token_transformer": VGGTDreamerSpec(
        name="vggt_agg_token_transformer",
        readout=TokenReadout("global"),
        storage=StorageSpec(replay_rgb=False, replay_readout=True),
        dreamer=DreamerEncoderSpec(
            "transformer", wm_encoders.VGGTAggTokenTransformerEncoder, "flat_features"
        ),
        agent_overrides={
            "buffer_capacity": 5_000,
            "batch_size": 1,
            "seq_len": 8,
            "train_ratio": 32,
        },
        design_notes="Full 1374-token VGGT aggregator sequence through a Transformer.",
    ),
    "hybrid": VGGTDreamerSpec(
        name="hybrid",
        readout=HeadReadout(37),
        storage=StorageSpec(
            replay_rgb=True, replay_readout=True, readout_dtype="float32"
        ),
        dreamer=DreamerEncoderSpec(
            "hybrid", wm_encoders.HybridEncoder, "rgb_plus_flat"
        ),
        agent_overrides={"buffer_capacity": 100_000},
        design_notes="RGB64 CNN plus gated WP/CP MLP branch.",
    ),
    "vggt_house_context": VGGTDreamerSpec(
        name="vggt_house_context",
        readout=TokenReadout("full", token_dim=VGGT_FULL_TOKEN_EMBED_DIM),
        storage=StorageSpec(
            replay_rgb=True, replay_readout=True, readout_dtype="float32"
        ),
        dreamer=DreamerEncoderSpec(
            "hybrid", wm_encoders.HybridEncoder, "rgb_plus_context"
        ),
        agent_overrides={
            "buffer_capacity": 1_000_000,
            "vggt_feature_dim": wm_encoders.HOUSE_CONTEXT_DIM,
            "vggt_token_dim": VGGT_FULL_TOKEN_EMBED_DIM,
            "vggt_token_count": wm_encoders.AGG_TOKEN_TOKENS,
        },
        design_notes="RGB replay plus live full-token VGGT house context.",
    ),
    "vggt_house_full_tokens_nogate": VGGTDreamerSpec(
        name="vggt_house_full_tokens_nogate",
        readout=TokenReadout("full", token_dim=VGGT_FULL_TOKEN_EMBED_DIM),
        storage=StorageSpec(replay_rgb=True, replay_readout=True),
        dreamer=DreamerEncoderSpec(
            "transformer", wm_encoders.RGBFullTokenTransformerEncoder, "rgb_plus_tokens"
        ),
        agent_overrides={
            **_SMALL_REPLAY_OVERRIDES,
            "vggt_token_dim": VGGT_FULL_TOKEN_EMBED_DIM,
            "vggt_token_count": wm_encoders.AGG_TOKEN_TOKENS,
        },
        design_notes="RGB replay plus per-step full-width VGGT tokens, no gate.",
    ),
    "vggt_house_global_tokens_nogate": VGGTDreamerSpec(
        name="vggt_house_global_tokens_nogate",
        readout=TokenReadout("global"),
        storage=StorageSpec(replay_rgb=True, replay_readout=True),
        dreamer=DreamerEncoderSpec(
            "transformer", wm_encoders.RGBGlobalTokenTransformerEncoder, "rgb_plus_tokens"
        ),
        agent_overrides={
            **_SMALL_REPLAY_OVERRIDES,
            "vggt_token_dim": VGGT_AGGREGATOR_EMBED_DIM,
            "vggt_token_count": wm_encoders.AGG_TOKEN_TOKENS,
        },
        design_notes="RGB replay plus per-step global-half VGGT tokens, no gate.",
    ),
}


def wp_cp_dim(wp_pool_size: int = VGGT_DEFAULT_WP_POOL_SIZE) -> int:
    """Flat VGGT world-points + camera-pose feature dimension."""
    return wp_pool_size * wp_pool_size * VGGT_XYZ_CHANNELS + VGGT_CAMERA_POSE_DIM


def world_points_hwc_shape(wp_side: int) -> tuple[int, int, int]:
    """Canonical VGGT extractor output shape before adapter transposition."""
    return (int(wp_side), int(wp_side), VGGT_XYZ_CHANNELS)


def world_points_side_for_head_readout(extractor: Any, readout: HeadReadout) -> int:
    """Resolve the variant/extractor-dependent world-points side length."""
    if readout.wp_side == "dense":
        return int(getattr(extractor, "image_size", VGGT_IMAGE_SIZE))
    return int(readout.wp_side)


def contract_world_points_hwc_shape(
    contract: EncoderInputContract,
) -> tuple[int, int, int]:
    """Resolve the extractor-facing HWC point-map shape from a structured contract."""
    shape_by_field = contract.replay_observation.buffer_shape()
    if not isinstance(shape_by_field, dict):
        raise ValueError("head VGGT readouts require structured replay fields")
    chw_shape = tuple(shape_by_field[WORLD_POINTS_KEY])
    if len(chw_shape) != 3:
        raise ValueError(f"expected CHW world-points contract shape, got {chw_shape}")
    channels, height, width = chw_shape
    if height != width:
        raise ValueError(
            f"expected square world-points contract shape, got {chw_shape}"
        )
    expected_hwc_shape = world_points_hwc_shape(height)
    if channels != expected_hwc_shape[-1]:
        raise ValueError(
            f"expected {expected_hwc_shape[-1]} world-point channels, got {channels}"
        )
    return expected_hwc_shape


def aggregator_pooled_dim(aggregator_feature_shape: tuple[int, ...]) -> int:
    """Flat dimension after [camera, mean patches, max patches] pooling."""
    return VGGT_AGGREGATOR_POOL_COUNT * int(aggregator_feature_shape[-1])


def aggregator_raw_dim(aggregator_feature_shape: tuple[int, ...]) -> int:
    """Flat dimension for camera + patch tokens with register tokens dropped."""
    n_tokens = 1 + (int(aggregator_feature_shape[0]) - VGGT_AGGREGATOR_PATCH_START_IDX)
    return n_tokens * int(aggregator_feature_shape[-1])


def aggregator_token_dim(aggregator_feature_shape: tuple[int, ...]) -> int:
    """Flat dimension for all VGGT aggregator tokens, including registers."""
    return int(aggregator_feature_shape[0]) * int(aggregator_feature_shape[-1])


def hybrid_feature_dim(wp_pool_size: int = VGGT_DEFAULT_WP_POOL_SIZE) -> int:
    """Packed hybrid Encoder Module input dimension."""
    return HYBRID_RGB_DIM + wp_cp_dim(wp_pool_size)


HYBRID_FEATURE_DIM = hybrid_feature_dim()


def _env_observation(env_render_resolution: int) -> ObservationFormContract:
    return ObservationFormContract(
        {
            "image": ObservationField(
                (3, env_render_resolution, env_render_resolution), "uint8"
            ),
            "is_first": ObservationField((), "bool"),
        }
    )


def _agent_features_observation(shape: tuple[int, ...]) -> ObservationFormContract:
    return ObservationFormContract(
        {
            "features": ObservationField(shape, "float32"),
            "is_first": ObservationField((), "bool"),
        }
    )


def _wp_cp_fields(
    world_points_shape: tuple[int, int, int],
    *,
    world_points_dtype: str,
    camera_pose_dtype: str,
) -> dict[str, ObservationField]:
    return {
        WORLD_POINTS_KEY: ObservationField(
            world_points_shape, world_points_dtype, normalize_on_sample=False
        ),
        CAMERA_POSE_KEY: ObservationField(
            (VGGT_CAMERA_POSE_DIM,), camera_pose_dtype, normalize_on_sample=False
        ),
    }


HEAD_AGENT_DTYPES = {
    WORLD_POINTS_KEY: "float16",
    CAMERA_POSE_KEY: "float16",
}


def _spec_for_feature_kind(
    feature_kind: VGGTFeatureKind,
    wp_pool_size: int = VGGT_DEFAULT_WP_POOL_SIZE,
) -> VGGTDreamerSpec:
    if feature_kind == "wp_cp":
        return VGGT_DREAMER_SPECS[
            "vggt_wp_cp_64" if wp_pool_size == 64 else "vggt"
        ]
    return {
        "wp64_cp": VGGT_DREAMER_SPECS["vggt_wp64_cnn_cp_mlp"],
        "wp_dense": VGGT_DREAMER_SPECS["vggt_wp_dense_cnn"],
        "aggregator": VGGT_DREAMER_SPECS["vggt_aggregator_mlp"],
        "agg_raw": VGGT_DREAMER_SPECS["vggt_agg_raw"],
        "agg_tokens": VGGT_DREAMER_SPECS["vggt_agg_token_transformer"],
    }[feature_kind]


def _resolve_dreamer_spec(
    *,
    feature_kind: VGGTFeatureKind,
    wp_pool_size: int,
    encoder_type: str | None,
) -> VGGTDreamerSpec:
    if encoder_type is not None:
        return VGGT_DREAMER_SPECS[encoder_type]
    return _spec_for_feature_kind(feature_kind, wp_pool_size)


def head_readout_spec(feature_kind: VGGTFeatureKind) -> HeadReadout | None:
    """Return the head readout for legacy readout construction."""
    spec = _spec_for_feature_kind(feature_kind)
    return spec.readout if isinstance(spec.readout, HeadReadout) else None


def _vggt_shape_dtype(
    extractor: Any, feature_kind: VGGTFeatureKind
) -> tuple[tuple[int, ...], str]:
    if feature_kind == "aggregator":
        return (
            aggregator_pooled_dim(tuple(extractor.aggregator_feature_shape)),
        ), "float32"
    if feature_kind == "agg_raw":
        return (
            aggregator_raw_dim(tuple(extractor.aggregator_feature_shape)),
        ), "float16"
    if feature_kind == "agg_tokens":
        return (
            aggregator_token_dim(tuple(extractor.aggregator_feature_shape)),
        ), "float16"
    raise ValueError(f"unknown non-head VGGT feature_kind {feature_kind!r}")


def _observation_dims(
    *,
    render_size: int,
    wp_side: int = VGGT_DEFAULT_WP_POOL_SIZE,
    token_count: int = VGGT_AGGREGATOR_TOKEN_COUNT,
    token_dim: int = VGGT_AGGREGATOR_EMBED_DIM,
) -> ObservationDims:
    return ObservationDims(
        render_size=render_size,
        replay_image_size=HYBRID_IMAGE_SIZE,
        wp_side=wp_side,
        camera_pose_dim=VGGT_CAMERA_POSE_DIM,
        xyz_channels=VGGT_XYZ_CHANNELS,
        token_count=token_count,
        token_dim=token_dim,
    )


def _head_replay_config(
    spec: VGGTDreamerSpec,
    *,
    dims: ObservationDims,
) -> ObservationRunConfig:
    return ObservationRunConfig(
        encoder=spec.encoder_type,
        dims=dims,
        replay=ReplayObservationConfig(
            components=("image", "wp_cp") if spec.storage.replay_rgb else ("world_points",),
            feature_dtype=spec.storage.readout_dtype,
            normalize_image=False,
        ),
    )


def _feature_replay_config(
    spec: VGGTDreamerSpec,
    *,
    dims: ObservationDims,
    feature_shape: tuple[int, ...],
) -> ObservationRunConfig:
    return ObservationRunConfig(
        encoder=spec.encoder_type,
        dims=dims,
        replay=ReplayObservationConfig(
            components=("features",),
            feature_dtype=spec.storage.readout_dtype,
            normalize_image=False,
            feature_shape=feature_shape,
        ),
    )


def build_vggt_contract(  # pylint: disable=too-many-arguments,too-many-locals
    extractor: Any,
    *,
    feature_kind: VGGTFeatureKind,
    env_render_resolution: int | None = None,
    encoder_type: str | None = None,
    encoder_module_cls: type[nn.Module] | None = None,
    agent_overrides: Mapping[str, Any] | None = None,
    design_notes: str = "",
) -> EncoderInputContract:
    """Build the Encoder Input Contract for one VGGT readout variant."""
    extractor_wp_pool_size = int(
        getattr(extractor, "wp_pool_size", VGGT_DEFAULT_WP_POOL_SIZE)
    )
    spec = _resolve_dreamer_spec(
        feature_kind=feature_kind,
        wp_pool_size=extractor_wp_pool_size,
        encoder_type=encoder_type,
    )
    render_resolution = int(
        env_render_resolution or getattr(extractor, "image_size", VGGT_IMAGE_SIZE)
    )
    resolved_overrides = dict(
        agent_overrides if agent_overrides is not None else spec.agent_overrides
    )
    resolved_notes = design_notes or spec.design_notes
    if spec.storage.replay_rgb or not spec.storage.replay_readout:
        raise ValueError(
            f"{spec.encoder_type} must store only the VGGT readout; got "
            f"replay_rgb={spec.storage.replay_rgb}, "
            f"replay_readout={spec.storage.replay_readout}"
        )

    if isinstance(spec.readout, HeadReadout):
        wp_side = world_points_side_for_head_readout(extractor, spec.readout)
        dims = _observation_dims(render_size=render_resolution, wp_side=wp_side)
        wp_shape = dims.world_points_shape
        replay_config = _head_replay_config(spec, dims=dims)
        agent_fields = _wp_cp_fields(
            wp_shape,
            world_points_dtype=HEAD_AGENT_DTYPES[WORLD_POINTS_KEY],
            camera_pose_dtype=HEAD_AGENT_DTYPES[CAMERA_POSE_KEY],
        )
        encoder_fields = _wp_cp_fields(
            wp_shape,
            world_points_dtype="float32",
            camera_pose_dtype="float32",
        )
        encoder_input_by_layout = {
            "flat_wp_cp": ObservationFormContract(
                ObservationField(
                    dims.wp_cp_shape, "float32", normalize_on_sample=False
                )
            ),
            "structured_wp_cp": ObservationFormContract(encoder_fields),
            "world_points": ObservationFormContract(
                ObservationField(wp_shape, "float32", normalize_on_sample=False)
            ),
        }
        return EncoderInputContract(
            observation_preparation_type=spec.encoder_type,
            encoder_type=spec.encoder_type,
            env_render_resolution=render_resolution,
            encoder_module_cls=encoder_module_cls or spec.module_cls,
            env_observation=_env_observation(render_resolution),
            replay_observation=replay_observation_form(replay_config),
            agent_observation=ObservationFormContract(
                {**agent_fields, "is_first": ObservationField((), "bool")}
            ),
            encoder_input=encoder_input_by_layout[spec.dreamer.input_layout],
            decoder_target=None,
            agent_overrides=resolved_overrides,
            design_notes=resolved_notes,
        )

    shape, _dtype = _vggt_shape_dtype(extractor, spec.feature_kind)
    token_shape = tuple(getattr(extractor, "aggregator_feature_shape", ()))
    token_count = int(token_shape[0]) if token_shape else VGGT_AGGREGATOR_TOKEN_COUNT
    token_dim = int(token_shape[-1]) if token_shape else VGGT_AGGREGATOR_EMBED_DIM
    dims = _observation_dims(
        render_size=render_resolution,
        token_count=token_count,
        token_dim=token_dim,
    )
    replay_config = _feature_replay_config(spec, dims=dims, feature_shape=shape)
    encoder_field = ObservationField(shape, "float32", normalize_on_sample=False)
    return EncoderInputContract(
        observation_preparation_type=spec.encoder_type,
        encoder_type=spec.encoder_type,
        env_render_resolution=render_resolution,
        encoder_module_cls=encoder_module_cls or spec.module_cls,
        env_observation=_env_observation(render_resolution),
        replay_observation=replay_observation_form(replay_config),
        agent_observation=_agent_features_observation(shape),
        encoder_input=ObservationFormContract(encoder_field),
        decoder_target=None,
        agent_overrides=resolved_overrides,
        design_notes=resolved_notes,
    )


def build_hybrid_contract(
    extractor: Any,
    *,
    env_render_resolution: int | None = None,
    encoder_module_cls: type[nn.Module] | None = None,
    agent_overrides: Mapping[str, Any] | None = None,
    design_notes: str = "",
) -> EncoderInputContract:
    """Build the Encoder Input Contract for the RGB + VGGT WP/CP hybrid."""
    spec = VGGT_DREAMER_SPECS["hybrid"]
    wp_pool_size = spec.wp_pool_size
    render_resolution = int(
        env_render_resolution or getattr(extractor, "image_size", VGGT_IMAGE_SIZE)
    )
    dims = _observation_dims(render_size=render_resolution, wp_side=wp_pool_size)
    wp_cp_shape = dims.wp_cp_shape
    replay_config = _head_replay_config(spec, dims=dims)
    if not (spec.storage.replay_rgb and spec.storage.replay_readout):
        raise ValueError("hybrid requires RGB and VGGT readout in replay")

    agent_fields = {
        HYBRID_IMAGE_KEY: ObservationField(
            dims.image_shape, "uint8", normalize_on_sample=False
        ),
        HYBRID_WP_CP_KEY: ObservationField(
            wp_cp_shape, "float32", normalize_on_sample=False
        ),
        "is_first": ObservationField((), "bool"),
    }

    return EncoderInputContract(
        observation_preparation_type=spec.encoder_type,
        encoder_type=spec.encoder_type,
        env_render_resolution=render_resolution,
        encoder_module_cls=encoder_module_cls or spec.module_cls,
        env_observation=_env_observation(render_resolution),
        replay_observation=replay_observation_form(replay_config),
        agent_observation=ObservationFormContract(agent_fields),
        encoder_input=ObservationFormContract(
            ObservationField((HYBRID_RGB_DIM + dims.wp_cp_dim,), "float32")
        ),
        decoder_target=ObservationFormContract(
            ObservationField(dims.image_shape, "float32")
        ),
        agent_overrides=dict(
            agent_overrides if agent_overrides is not None else spec.agent_overrides
        ),
        design_notes=design_notes or spec.design_notes,
    )
