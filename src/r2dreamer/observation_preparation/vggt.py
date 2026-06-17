"""VGGT-family Observation Preparation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Any

import flax.linen as nn

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
)
from src.r2dreamer.world_model import encoders as wm_encoders


VGGTFeatureKind = Literal["wp_cp", "wp64_cp", "aggregator", "wp_dense", "agg_raw", "agg_tokens"]

VGGT_IMAGE_SIZE = 518
VGGT_IMAGE_SHAPE = (3, VGGT_IMAGE_SIZE, VGGT_IMAGE_SIZE)
VGGT_DEFAULT_WP_POOL_SIZE = 37
VGGT_CAMERA_POSE_DIM = 9
VGGT_XYZ_CHANNELS = 3
VGGT_AGGREGATOR_POOL_COUNT = 3
VGGT_AGGREGATOR_PATCH_START_IDX = 5

HYBRID_IMAGE_SHAPE = (3, 64, 64)
HYBRID_RGB_DIM = HYBRID_IMAGE_SHAPE[0] * HYBRID_IMAGE_SHAPE[1] * HYBRID_IMAGE_SHAPE[2]


_DEFAULT_AGENT_OVERRIDES: dict[str, dict[str, Any]] = {
    "vggt": {"buffer_capacity": 1_000_000},
    "vggt_wp_cp_64": {"buffer_capacity": 1_000_000},
    "vggt_wp64_cnn_cp_mlp": {"buffer_capacity": 1_000_000},
    "vggt_aggregator_mlp": {
        "buffer_capacity": 5_000,
        "batch_size": 4,
        "seq_len": 32,
        "train_ratio": 128,
    },
    "vggt_wp_dense_cnn": {
        "buffer_capacity": 5_000,
        "batch_size": 4,
        "seq_len": 32,
        "train_ratio": 128,
    },
    "vggt_agg_token_transformer": {
        "buffer_capacity": 5_000,
        "batch_size": 1,
        "seq_len": 8,
        "train_ratio": 32,
    },
    "hybrid": {"buffer_capacity": 100_000},
}


def wp_cp_dim(wp_pool_size: int = VGGT_DEFAULT_WP_POOL_SIZE) -> int:
    """Flat VGGT world-points + camera-pose feature dimension."""
    return wp_pool_size * wp_pool_size * VGGT_XYZ_CHANNELS + VGGT_CAMERA_POSE_DIM


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
            "image": ObservationField((3, env_render_resolution, env_render_resolution), "uint8"),
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


def _default_encoder_type(feature_kind: VGGTFeatureKind, wp_pool_size: int) -> str:
    if feature_kind == "wp_cp":
        return "vggt_wp_cp_64" if wp_pool_size == 64 else "vggt"
    if feature_kind == "wp64_cp":
        return "vggt_wp64_cnn_cp_mlp"
    if feature_kind == "aggregator":
        return "vggt_aggregator_mlp"
    if feature_kind == "wp_dense":
        return "vggt_wp_dense_cnn"
    if feature_kind == "agg_raw":
        return "vggt_agg_raw"
    if feature_kind == "agg_tokens":
        return "vggt_agg_token_transformer"
    raise ValueError(f"unknown VGGT feature_kind {feature_kind!r}")


def _default_module_cls(encoder_type: str, feature_kind: VGGTFeatureKind) -> type[nn.Module]:
    if encoder_type == "vggt_aggregator_mlp" or feature_kind == "aggregator":
        return wm_encoders.VGGTAggregatorMLPEncoder
    if encoder_type == "vggt_wp64_cnn_cp_mlp" or feature_kind == "wp64_cp":
        return wm_encoders.WP64CNNCPMLPEncoder
    if encoder_type == "vggt_wp_dense_cnn" or feature_kind == "wp_dense":
        return wm_encoders.ConvEncoder
    if encoder_type == "vggt_agg_token_transformer" or feature_kind == "agg_tokens":
        return wm_encoders.VGGTAggTokenTransformerEncoder
    if encoder_type == "hybrid":
        return wm_encoders.HybridEncoder
    return wm_encoders.VGGTEncoder


def _vggt_shape_dtype(extractor: Any, feature_kind: VGGTFeatureKind) -> tuple[tuple[int, ...], str]:
    if feature_kind == "wp_cp":
        k = int(getattr(extractor, "wp_pool_size", VGGT_DEFAULT_WP_POOL_SIZE))
        return (wp_cp_dim(k),), "float32"
    if feature_kind == "aggregator":
        return (aggregator_pooled_dim(tuple(extractor.aggregator_feature_shape)),), "float32"
    if feature_kind == "wp_dense":
        image_size = int(getattr(extractor, "image_size", VGGT_IMAGE_SIZE))
        return (3, image_size, image_size), "float16"
    if feature_kind == "agg_raw":
        return (aggregator_raw_dim(tuple(extractor.aggregator_feature_shape)),), "float16"
    if feature_kind == "agg_tokens":
        return (aggregator_token_dim(tuple(extractor.aggregator_feature_shape)),), "float16"
    raise ValueError(f"unknown VGGT feature_kind {feature_kind!r}")


def build_vggt_contract(
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
    wp_pool_size = int(getattr(extractor, "wp_pool_size", VGGT_DEFAULT_WP_POOL_SIZE))
    resolved_encoder_type = encoder_type or _default_encoder_type(feature_kind, wp_pool_size)
    if feature_kind == "wp64_cp":
        replay_fields = {
            WORLD_POINTS_KEY: ObservationField((3, wp_pool_size, wp_pool_size), "float16", normalize_on_sample=False),
            CAMERA_POSE_KEY: ObservationField((VGGT_CAMERA_POSE_DIM,), "float16", normalize_on_sample=False),
        }
        encoder_fields = {
            WORLD_POINTS_KEY: ObservationField((3, wp_pool_size, wp_pool_size), "float32", normalize_on_sample=False),
            CAMERA_POSE_KEY: ObservationField((VGGT_CAMERA_POSE_DIM,), "float32", normalize_on_sample=False),
        }
        resolved_overrides = dict(
            agent_overrides
            if agent_overrides is not None
            else _DEFAULT_AGENT_OVERRIDES.get(resolved_encoder_type, {})
        )
        render_resolution = int(env_render_resolution or getattr(extractor, "image_size", VGGT_IMAGE_SIZE))
        return EncoderInputContract(
            observation_preparation_type=resolved_encoder_type,
            encoder_type=resolved_encoder_type,
            env_render_resolution=render_resolution,
            encoder_module_cls=encoder_module_cls or _default_module_cls(resolved_encoder_type, feature_kind),
            env_observation=_env_observation(render_resolution),
            replay_observation=ObservationFormContract(replay_fields),
            agent_observation=ObservationFormContract({**encoder_fields, "is_first": ObservationField((), "bool")}),
            encoder_input=ObservationFormContract(encoder_fields),
            decoder_target=None,
            agent_overrides=resolved_overrides,
            design_notes=design_notes,
        )
    shape, dtype = _vggt_shape_dtype(extractor, feature_kind)
    replay_field = ObservationField(shape, dtype, normalize_on_sample=False)
    encoder_field = ObservationField(shape, "float32", normalize_on_sample=False)
    resolved_overrides = dict(
        agent_overrides
        if agent_overrides is not None
        else _DEFAULT_AGENT_OVERRIDES.get(resolved_encoder_type, {})
    )

    return EncoderInputContract(
        observation_preparation_type=resolved_encoder_type,
        encoder_type=resolved_encoder_type,
        env_render_resolution=int(env_render_resolution or getattr(extractor, "image_size", VGGT_IMAGE_SIZE)),
        encoder_module_cls=encoder_module_cls or _default_module_cls(resolved_encoder_type, feature_kind),
        env_observation=_env_observation(int(env_render_resolution or getattr(extractor, "image_size", VGGT_IMAGE_SIZE))),
        replay_observation=ObservationFormContract(replay_field),
        agent_observation=_agent_features_observation(shape),
        encoder_input=ObservationFormContract(encoder_field),
        decoder_target=None,
        agent_overrides=resolved_overrides,
        design_notes=design_notes,
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
    wp_pool_size = int(getattr(extractor, "wp_pool_size", VGGT_DEFAULT_WP_POOL_SIZE))
    wp_cp_shape = (wp_cp_dim(wp_pool_size),)
    render_resolution = int(env_render_resolution or getattr(extractor, "image_size", VGGT_IMAGE_SIZE))

    replay_fields = {
        HYBRID_IMAGE_KEY: ObservationField(HYBRID_IMAGE_SHAPE, "uint8", normalize_on_sample=False),
        HYBRID_WP_CP_KEY: ObservationField(wp_cp_shape, "float32", normalize_on_sample=False),
    }
    agent_fields = {
        HYBRID_IMAGE_KEY: ObservationField(HYBRID_IMAGE_SHAPE, "uint8", normalize_on_sample=False),
        HYBRID_WP_CP_KEY: ObservationField(wp_cp_shape, "float32", normalize_on_sample=False),
        "is_first": ObservationField((), "bool"),
    }

    return EncoderInputContract(
        observation_preparation_type="hybrid",
        encoder_type="hybrid",
        env_render_resolution=render_resolution,
        encoder_module_cls=encoder_module_cls or wm_encoders.HybridEncoder,
        env_observation=_env_observation(render_resolution),
        replay_observation=ObservationFormContract(replay_fields),
        agent_observation=ObservationFormContract(agent_fields),
        encoder_input=ObservationFormContract(ObservationField((hybrid_feature_dim(wp_pool_size),), "float32")),
        decoder_target=ObservationFormContract(ObservationField(HYBRID_IMAGE_SHAPE, "float32")),
        agent_overrides=dict(agent_overrides if agent_overrides is not None else _DEFAULT_AGENT_OVERRIDES["hybrid"]),
        design_notes=design_notes,
    )
