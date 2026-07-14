"""Flax module factories for the R2Dreamer agent.

These functions turn an :class:`R2DreamerConfig` into the stateless Flax
modules the agent instantiates for ``.init`` / ``.apply`` — the encoder, the
RSSM, and the dummy observation used to discover ``embed_size`` at init time.

They were extracted from ``src/r2dreamer/agent.py`` so the agent file can
focus on orchestration (params, optimizer, EMA, JIT train/act steps, loss
composition) while this module owns the cfg -> Flax module construction
machinery: the ``encoder_type`` -> class resolution, the per-encoder
``_make_*`` builders, the Encoder-Input-Contract override prelude, and the
dummy-obs factories. Keeping the three parallel dispatches over
``encoder_type`` in one place is the point of the extraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from src.configs.config import R2DreamerConfig
from src.shared.dtypes import compute_jnp_dtype

from .constants import AGG_REGISTER_TOKENS, HYBRID_RGB_DIM
from .cnn import ConvEncoder
from .mlp import (
    HouseGlobalEmbeddingEncoder as WMHouseGlobalEmbeddingEncoder,
)
from .mlp import (
    HousePointsCameraEncoder,
    HybridHousePointsCameraEncoder,
    WP64CNNCPMLPEncoder,
)
from .mlp import (
    HybridEncoder as WMHybridEncoder,
)
from .mlp import (
    MLPEncoder as WMMLPEncoder,
)
from .mlp import (
    VGGTAggRawMLPEncoder as WMVGGTAggRawMLPEncoder,
)
from .mlp import (
    VGGTAggregatorMLPEncoder as WMVGGTAggregatorMLPEncoder,
)
from .pointnet import PointNetHousePointsCameraEncoder
from .transformer import TokenTransformerEncoder as WMTokenTransformerEncoder
from ..observation_keys import (
    CAMERA_POSE_KEY,
    CAMERA_TOKEN_GLOBAL_KEY,
    FULL_TOKENS_KEY,
    GLOBAL_PATCH_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
    HYBRID_IMAGE_KEY,
    WORLD_POINTS_KEY,
)
from ..observation_preparation.contracts import normalize_encoder_module_kwargs
from ..world_model.rssm import R2RSSM


# ---------------------------------------------------------------------------
# Module factories
# ---------------------------------------------------------------------------


def _compute_dtype_kwargs(cfg: R2DreamerConfig) -> dict[str, Any]:
    """Return the ``compute_dtype`` override for the ``full_bf16`` gate.

    Only supplies ``compute_dtype`` when ``cfg.full_bf16`` is set, so that
    with the gate off each module keeps its own default — historically
    float32 for the CNN/house/pose/RSSM/head path, but bfloat16 for modules
    that already opted in on their own (e.g. the PointNet house branch).

    Args:
      cfg: Agent config supplying ``full_bf16`` and ``compute_dtype``.

    Returns:
      ``{"compute_dtype": <jnp dtype>}`` when the gate is on, else ``{}``.
    """
    if getattr(cfg, "full_bf16", False):
        return {"compute_dtype": compute_jnp_dtype(cfg.compute_dtype)}
    return {}


def _make_rssm(cfg: R2DreamerConfig) -> R2RSSM:
    return R2RSSM(
        deter_size=cfg.deter_size,
        stoch_classes=cfg.stoch_classes,
        stoch_discrete=cfg.stoch_discrete,
        num_actions=cfg.num_actions,
        hidden=cfg.hidden_size,
        blocks=cfg.blocks,
        dyn_layers=cfg.dyn_layers,
        obs_layers=cfg.obs_layers,
        img_layers=cfg.img_layers,
        unimix_ratio=cfg.unimix_ratio,
        **_compute_dtype_kwargs(cfg),
    )


def _resolve_encoder_cls(cfg: R2DreamerConfig):
    # Launcher-created configs pass EncoderSpec.module_cls explicitly. Unit tests
    # and direct R2DreamerConfig() construction rely on encoder_type, so map the
    # documented names to their Flax modules when no class is supplied.
    cls = cfg.encoder_module_cls
    if cls is None:
        cls = {
            "cnn": ConvEncoder,
            "vggt": WMMLPEncoder,
            "vggt_wp_cp_64": WMMLPEncoder,  # same MLP module, finer WP grid (obs 12297)
            "vggt_aggregator_mlp": WMVGGTAggregatorMLPEncoder,
            "vggt_agg_raw": WMVGGTAggRawMLPEncoder,
            "vggt_agg_token_transformer": WMTokenTransformerEncoder,
            "vggt_wp_dense_cnn": ConvEncoder,
            "vggt_wp64_cnn_cp_mlp": WP64CNNCPMLPEncoder,
            "hybrid": WMHybridEncoder,
            "vggt_house_context": WMHybridEncoder,
            "vggt_house_points_pose": PointNetHousePointsCameraEncoder,
            "pointnet": PointNetHousePointsCameraEncoder,
            "vggt_hybrid_house_points_pose": HybridHousePointsCameraEncoder,
            "vggt_house_full_tokens_nogate": WMTokenTransformerEncoder,
            "vggt_house_global_tokens_nogate": WMTokenTransformerEncoder,
            "vggt_house_global_embedding": WMHouseGlobalEmbeddingEncoder,
        }.get(cfg.encoder_type)
        if cls is None:
            raise ValueError(f"unknown encoder_type {cfg.encoder_type!r}")
    return cls


def _validate_encoder_config(cfg: R2DreamerConfig, cls) -> None:
    if cls in (ConvEncoder, WP64CNNCPMLPEncoder) and cfg.vggt_mlp_layers != 1:
        # Fail loud instead of silently dropping the knob: conv encoders have no
        # MLP depth, so a non-default vggt_mlp_layers here is a misconfiguration.
        raise ValueError(
            f"vggt_mlp_layers={cfg.vggt_mlp_layers} has no effect on "
            f"{cls.__name__} (a conv encoder, no MLP blocks). Only the 'vggt' and "
            f"'vggt_aggregator_mlp' encoders consume vggt_mlp_layers; leave it at 1 "
            f"for cnn / vggt_wp_dense_cnn."
        )


def _contract_encoder_kwargs(cfg: R2DreamerConfig) -> dict[str, Any]:
    snapshot = getattr(cfg, "encoder_input_contract", None)
    if snapshot is None:
        return {}
    return normalize_encoder_module_kwargs(snapshot.get("encoder_module_kwargs", {}))


def _make_conv_encoder(cfg: R2DreamerConfig):
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return ConvEncoder(**kwargs)
    return ConvEncoder(
        depth=cfg.encoder_depth,
        kernel_size=cfg.encoder_kernel,
        mults=cfg.encoder_mults,
        **_compute_dtype_kwargs(cfg),
    )


def _make_wp_conv_encoder(cfg: R2DreamerConfig):
    # Full-res world-point map -> conv stack -> embed_dim (3D-53). Reuses the
    # RGB conv hyperparameters; symlog (not /255) handles the metric XYZ range.
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return ConvEncoder(**kwargs)
    return ConvEncoder(
        input_kind="world_points",
        embed_dim=cfg.vggt_embed_dim,
        depth=cfg.encoder_depth,
        kernel_size=cfg.encoder_kernel,
        mults=cfg.encoder_mults,
    )


def _make_wp64_cnn_cp_mlp_encoder(cfg: R2DreamerConfig):
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return WP64CNNCPMLPEncoder(**kwargs)
    return WP64CNNCPMLPEncoder(
        embed_dim=cfg.vggt_embed_dim,
        conv_depth=cfg.encoder_depth,
        conv_kernel=cfg.encoder_kernel,
        conv_mults=cfg.encoder_mults,
        cp_hidden=cfg.mlp_vggt_hidden,
        cp_layers=cfg.mlp_vggt_layers,
    )


def _make_hybrid_encoder(cfg: R2DreamerConfig):
    # CNN(RGB) + gated MLP(WP/CP) fused into one embed (3D-50/51/52).
    # HybridEncoder now owns structured replay/live layout handling.
    expected_shape = (HYBRID_RGB_DIM + cfg.vggt_feature_dim,)
    if not isinstance(cfg.obs_shape, tuple):
        raise ValueError(f"hybrid expects flat obs_shape, got {cfg.obs_shape}")
    if not (
        cfg.obs_shape == expected_shape
        and cfg.obs_shape[0] - cfg.vggt_feature_dim == HYBRID_RGB_DIM
    ):
        raise ValueError(
            "hybrid obs_shape/split mismatch: expected "
            f"{expected_shape} with vggt_feature_dim={cfg.vggt_feature_dim}, "
            f"got obs_shape={cfg.obs_shape}, "
            f"vggt_feature_dim={cfg.vggt_feature_dim}"
        )
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return WMHybridEncoder(**kwargs)
    return WMHybridEncoder(
        cnn_depth=cfg.encoder_depth,
        cnn_kernel=cfg.encoder_kernel,
        cnn_mults=cfg.encoder_mults,
        vggt_embed_dim=cfg.vggt_embed_dim,
        mlp_hidden=cfg.mlp_vggt_hidden,
        mlp_layers=cfg.mlp_vggt_layers,
        vggt_dim=cfg.vggt_feature_dim,
    )


def _make_house_points_camera_encoder(
    cfg: R2DreamerConfig, cls: type = HousePointsCameraEncoder
):
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return cls(**kwargs)
    kwargs = dict(
        embed_dim=cfg.vggt_embed_dim,
        camera_hidden=cfg.mlp_vggt_hidden,
        camera_layers=cfg.mlp_vggt_layers,
        point_hidden=cfg.mlp_vggt_hidden,
        point_layers=cfg.mlp_vggt_layers,
        house_point_norm=cfg.house_point_norm,
        **_compute_dtype_kwargs(cfg),
    )
    if issubclass(cls, HybridHousePointsCameraEncoder):
        kwargs.update(
            cnn_depth=cfg.encoder_depth,
            cnn_kernel=cfg.encoder_kernel,
            cnn_mults=cfg.encoder_mults,
        )
    return cls(**kwargs)


def _make_house_global_embedding_encoder(
    cfg: R2DreamerConfig, cls: type = WMHouseGlobalEmbeddingEncoder
):
    # PointNet reducer over VGGT global patch tokens + camera side branch.
    # token_dim and num_patch_tokens are fixed by the VGGT global-half token
    # layout (camera token + 4 registers dropped): num_patch_tokens =
    # vggt_token_count - (1 camera + AGG_REGISTER_TOKENS). Prod sets
    # vggt_token_dim=1024 / vggt_token_count=1374 via agent_overrides; tests
    # may inject small dims.
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return cls(**kwargs)
    num_patch_tokens = int(cfg.vggt_token_count) - (1 + AGG_REGISTER_TOKENS)
    return cls(
        embed_dim=cfg.vggt_embed_dim,
        token_dim=cfg.vggt_token_dim,
        num_patch_tokens=num_patch_tokens,
        reducer_hidden=cfg.mlp_vggt_hidden,
        reducer_layers=cfg.mlp_vggt_layers,
        camera_hidden=cfg.mlp_vggt_hidden,
        camera_layers=cfg.mlp_vggt_layers,
        rgb_branch=cfg.vggt_house_global_rgb_branch,
    )


def _make_mlp_encoder(cfg: R2DreamerConfig, cls):
    # wp_cp + aggregator MLP encoders: depth from cfg.vggt_mlp_layers (3D-52).
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return cls(**kwargs)
    return cls(
        embed_dim=cfg.vggt_embed_dim,
        hidden=cfg.vggt_embed_dim,
        num_layers=cfg.vggt_mlp_layers,
    )


def _make_rgb_token_encoder(cfg: R2DreamerConfig):
    token_key = FULL_TOKENS_KEY
    singleton_tokens = False
    if cfg.encoder_type == "vggt_house_global_tokens_nogate":
        token_key = GLOBAL_TOKENS_KEY
        singleton_tokens = True
    return WMTokenTransformerEncoder(
        embed_dim=cfg.vggt_embed_dim,
        token_dim=cfg.vggt_token_dim,
        num_tokens=cfg.vggt_token_count,
        model_dim=None,
        layers=cfg.vggt_token_transformer_layers,
        heads=cfg.vggt_token_transformer_heads,
        mlp_ratio=cfg.vggt_token_transformer_mlp_ratio,
        dropout=cfg.vggt_token_transformer_dropout,
        readout="mean",
        norm_kind="layer",
        activation="gelu",
        token_key=token_key,
        image_key=HYBRID_IMAGE_KEY,
        singleton_tokens=singleton_tokens,
        compute_dtype=compute_jnp_dtype(cfg.compute_dtype),
        cnn_depth=cfg.encoder_depth,
        cnn_kernel=cfg.encoder_kernel,
        cnn_mults=cfg.encoder_mults,
    )


def _make_token_transformer_encoder(cfg: R2DreamerConfig):
    return WMTokenTransformerEncoder(
        embed_dim=cfg.vggt_embed_dim,
        token_dim=cfg.vggt_token_dim,
        num_tokens=cfg.vggt_token_count,
        model_dim=cfg.vggt_token_projection_dim,
        layers=cfg.vggt_token_transformer_layers,
        heads=cfg.vggt_token_transformer_heads,
        mlp_ratio=cfg.vggt_token_transformer_mlp_ratio,
        readout="camera_register_patch",
        norm_kind="rms",
        activation="silu",
        keep_register_tokens=cfg.vggt_keep_register_tokens,
        compute_dtype=compute_jnp_dtype(cfg.compute_dtype),
    )


def _make_encoder(cfg: R2DreamerConfig):
    cls = _resolve_encoder_cls(cfg)
    _validate_encoder_config(cfg, cls)
    if cls is ConvEncoder:
        if cfg.encoder_type == "vggt_wp_dense_cnn":
            return _make_wp_conv_encoder(cfg)
        return _make_conv_encoder(cfg)
    if cls is WP64CNNCPMLPEncoder:
        return _make_wp64_cnn_cp_mlp_encoder(cfg)
    if cls is WMHybridEncoder:
        return _make_hybrid_encoder(cfg)
    if cls is WMHouseGlobalEmbeddingEncoder:
        return _make_house_global_embedding_encoder(cfg, cls)
    if issubclass(cls, HousePointsCameraEncoder):
        # issubclass so GNN variants (src/r2dreamer/encoders/gnn_house.py)
        # reuse this builder with their own module class.
        return _make_house_points_camera_encoder(cfg, cls)
    if cls is WMTokenTransformerEncoder:
        if cfg.encoder_type == "vggt_agg_token_transformer":
            return _make_token_transformer_encoder(cfg)
        return _make_rgb_token_encoder(cfg)
    return _make_mlp_encoder(cfg, cls)


def _dummy_encoder_obs(cfg: R2DreamerConfig):
    if cfg.encoder_type == "vggt_house_full_tokens_nogate":
        return {
            HYBRID_IMAGE_KEY: jnp.zeros((1, 3, 64, 64), dtype=jnp.float32),
            FULL_TOKENS_KEY: jnp.zeros(
                (1, cfg.vggt_token_count, cfg.vggt_token_dim),
                dtype=compute_jnp_dtype(cfg.compute_dtype),
            ),
        }
    if cfg.encoder_type == "vggt_house_global_tokens_nogate":
        return {
            HYBRID_IMAGE_KEY: jnp.zeros((1, 3, 64, 64), dtype=jnp.float32),
            GLOBAL_TOKENS_KEY: jnp.zeros(
                (1, cfg.vggt_token_count, cfg.vggt_token_dim),
                dtype=compute_jnp_dtype(cfg.compute_dtype),
            ),
        }
    if cfg.encoder_type == "vggt_house_global_embedding":
        if not isinstance(cfg.obs_shape, Mapping):
            raise TypeError(f"{cfg.encoder_type} expects structured obs_shape")
        return {
            HYBRID_IMAGE_KEY: jnp.zeros(
                (1, *cfg.obs_shape[HYBRID_IMAGE_KEY]), dtype=jnp.float32
            ),
            CAMERA_TOKEN_GLOBAL_KEY: jnp.zeros(
                (1, *cfg.obs_shape[CAMERA_TOKEN_GLOBAL_KEY]), dtype=jnp.float32
            ),
            GLOBAL_PATCH_TOKENS_KEY: jnp.zeros(
                (1, *cfg.obs_shape[GLOBAL_PATCH_TOKENS_KEY]), dtype=jnp.float32
            ),
        }
    if cfg.encoder_type == "vggt_wp64_cnn_cp_mlp":
        return {
            WORLD_POINTS_KEY: jnp.zeros((1, 3, 64, 64), dtype=jnp.float32),
            CAMERA_POSE_KEY: jnp.zeros((1, 9), dtype=jnp.float32),
        }
    if cfg.encoder_type in (
        "vggt_house_points_pose",
        "vggt_hybrid_house_points_pose",
        "gnn_house_points_pose",
        "gnn_edge_house_points_pose",
        "pointnet",
    ):
        if not isinstance(cfg.obs_shape, Mapping):
            raise TypeError(f"{cfg.encoder_type} expects structured obs_shape")
        dummy = {
            CAMERA_POSE_KEY: jnp.zeros((1, 9), dtype=jnp.float32),
            HOUSE_CONTEXT_KEY: jnp.zeros(
                (1, *cfg.obs_shape[HOUSE_CONTEXT_KEY]), dtype=jnp.float32
            ),
            HOUSE_CONTEXT_SIZE_KEY: jnp.zeros((), dtype=jnp.int32),
        }
        if cfg.encoder_type == "vggt_hybrid_house_points_pose":
            dummy[HYBRID_IMAGE_KEY] = jnp.zeros((1, 3, 64, 64), dtype=jnp.float32)
        return dummy
    return jnp.zeros((1, *cfg.obs_shape))