"""Encoder recipes: one registry entry per encoder type (design card 6a).

An :class:`EncoderRecipe` couples everything one encoder type needs, per the
encoder-split design (prototyp/prototyp-encoder-split):

- ``make_adapter`` builds the *frozen* observation adapter (VGGT extraction,
  house-point accumulation, symlog) that the ``ExperienceCollector`` owns;
- ``build_module`` builds the *trainable* Flax encoder module the learner
  differentiates — a :class:`CompositeEncoder` for declarative combinations
  (``cnn``, ``hybrid``), the mechanism/combination module for the rest;
- ``dummy_obs`` builds the batch-1 init dummy matching the prepared-frame
  schema (used when no real first frame is available, e.g. unit tests and
  direct config construction);
- ``rgb_key`` marks recipes that carry an RGB modality (decoder capability),
  replacing the global ``*_RGB_ENCODER_TYPES`` name lists over time.

This module is the single source of truth for cfg -> encoder-module
construction: the former ``encoders/factory.py`` dispatch moved here verbatim
(same kwargs helpers, same contract-snapshot override, same fail-loud
validation), so every variant stays byte-identical to its legacy construction.

There is deliberately no declared obs-spec: it is inferred from the first
prepared frame at startup, and the single remaining consistency check is
``set(composite.branches) == set(inferred keys)`` (:func:`check_branch_keys`)
for composite encoders. Non-composite mechanism modules own their observation
layout themselves (they accept both dict and packed forms), so the check does
not apply to them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

import flax.linen as nn
import jax.numpy as jnp

from src.configs.config import R2DreamerConfig
from src.r2dreamer.encoders.cnn import ConvEncoder, make_rgb_conv_encoder
from src.r2dreamer.encoders.composite import (
    BranchSpec,
    CompositeEncoder,
    CompositeSpec,
)
from src.r2dreamer.encoders.constants import HYBRID_RGB_DIM
from src.r2dreamer.encoders.mlp import (
    HouseGlobalEmbeddingEncoder,
    HousePointsCameraEncoder,
    HybridEncoder as WMHybridEncoder,
    HybridHousePointsCameraEncoder,
    MLPEncoder,
    VGGTAggRawMLPEncoder,
    VGGTAggregatorMLPEncoder,
    WP64CNNCPMLPEncoder,
)
from src.r2dreamer.encoders.pointnet import PointNetHousePointsCameraEncoder
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    FULL_TOKENS_KEY,
    GLOBAL_PATCH_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
    WORLD_POINTS_KEY,
)
from src.r2dreamer.observation_preparation.contracts import (
    encoder_module_kwargs_from_config,
    normalize_encoder_module_kwargs,
)
from src.r2dreamer.world_model.factory import compute_dtype_kwargs
from src.r2dreamer.world_model.heads import R2MLP
from src.shared.dtypes import compute_jnp_dtype


@dataclass(frozen=True)
class EncoderRecipe:
    """Everything needed to build one encoder type's adapter and module.

    Attributes:
        encoder_type: Canonical encoder-type name (also the registry key).
        make_adapter: Builds the frozen observation adapter from the parsed
            training args (delegating to the launcher encoder selection so the
            frozen extraction path is byte-for-byte unchanged).
        build_module: Builds the trainable Flax encoder module from the
            effective agent config. Honors ``cfg.encoder_module_cls`` overrides
            and durable Encoder-Input-Contract kwargs exactly as the legacy
            factory did.
        dummy_obs: Builds the batch-1 init dummy for this encoder type.
        rgb_key: Observation key carrying an RGB image, or ``None``. A decoder
            target is available iff this is set.
        composite: When the module is a :class:`CompositeEncoder`, the builder
            for its :class:`CompositeSpec` (exposed for tests/tools); ``None``
            for mechanism/combination modules.
    """

    encoder_type: str
    make_adapter: Callable[[Any], Any]
    build_module: Callable[[R2DreamerConfig], nn.Module]
    dummy_obs: Callable[[R2DreamerConfig], Any]
    rgb_key: str | None = None
    composite: Callable[[R2DreamerConfig], CompositeSpec] | None = None


# ---------------------------------------------------------------------------
# Shared construction helpers (moved verbatim from encoders/factory.py)
# ---------------------------------------------------------------------------


def _contract_module_kwargs(cfg: R2DreamerConfig) -> dict:
    """Return the durable Encoder Input Contract's module kwargs, if any.

    A checkpoint's contract snapshot pins the exact kwargs the encoder was
    built with, and those win over the current config so an old checkpoint
    rebuilds the same module even when the config has drifted.
    """
    snapshot = getattr(cfg, "encoder_input_contract", None)
    if snapshot is None:
        return {}
    return normalize_encoder_module_kwargs(snapshot.get("encoder_module_kwargs", {}))


def validate_encoder_config(cfg: R2DreamerConfig, cls: type) -> None:
    """Fail loud when config knobs cannot apply to the selected module class."""
    if cls in (ConvEncoder, WP64CNNCPMLPEncoder) and cfg.vggt_mlp_layers != 1:
        # Fail loud instead of silently dropping the knob: conv encoders have no
        # MLP depth, so a non-default vggt_mlp_layers here is a misconfiguration.
        raise ValueError(
            f"vggt_mlp_layers={cfg.vggt_mlp_layers} has no effect on "
            f"{cls.__name__} (a conv encoder, no MLP blocks). Only the 'vggt' and "
            f"'vggt_aggregator_mlp' encoders consume vggt_mlp_layers; leave it at 1 "
            f"for cnn / vggt_wp_dense_cnn."
        )


def validate_hybrid_split(cfg: R2DreamerConfig) -> None:
    """Fail loud on a hybrid obs_shape / feature-split mismatch (3D-50/51/52)."""
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


def _module_cls(cfg: R2DreamerConfig, default: type) -> type:
    """Return the launcher-supplied module class override, or the default.

    Launcher-created configs pass ``EncoderSpec.module_cls`` explicitly (the
    GNN house variants rely on this); unit tests and direct config construction
    fall back to the recipe's default class.
    """
    return cfg.encoder_module_cls if cfg.encoder_module_cls is not None else default


# ---------------------------------------------------------------------------
# Composite builders (cnn + hybrid — golden-gated, bit-identical to legacy)
# ---------------------------------------------------------------------------


def build_cnn_composite(cfg: R2DreamerConfig) -> CompositeSpec:
    """Single ConvEncoder branch — byte-identical to the legacy ``cnn`` module.

    Honors the contract-pinned kwargs when present (checkpoint portability),
    else derives depth/kernel/mults from config plus the full_bf16 dtype
    overlay — exactly matching the legacy ``_make_conv_encoder``.
    """
    contract_kwargs = _contract_module_kwargs(cfg)
    conv_kwargs = contract_kwargs or {
        "depth": int(cfg.encoder_depth),
        "kernel_size": int(cfg.encoder_kernel),
        "mults": tuple(cfg.encoder_mults),
        **compute_dtype_kwargs(cfg),
    }
    return CompositeSpec(
        branches=(
            BranchSpec(
                obs_key=HYBRID_IMAGE_KEY,
                module_name="cnn",
                make=lambda name: ConvEncoder(name=name, **conv_kwargs),
            ),
        ),
        fusion="concat",
    )


def build_hybrid_composite(cfg: R2DreamerConfig) -> CompositeSpec:
    """CNN(RGB) + gated R2MLP(WP/CP) — reproduces ``WMHybridEncoder`` exactly.

    Mirrors ``base.py::_vggt_module_kwargs`` for ``HybridEncoder``: the CNN
    branch uses ``encoder_{depth,kernel,mults}`` and the gated MLP branch uses
    ``mlp_vggt_{hidden,layers}`` projecting to ``vggt_embed_dim``. The legacy
    RGB conv inside the hybrid runs float32 (no compute_dtype overlay), so the
    branch here does too. Contract kwargs (WMHybridEncoder parameter names)
    win over config when present.
    """
    contract = _contract_module_kwargs(cfg)
    depth = int(contract.get("cnn_depth", cfg.encoder_depth))
    kernel = int(contract.get("cnn_kernel", cfg.encoder_kernel))
    mults = tuple(contract.get("cnn_mults", cfg.encoder_mults))
    mlp_hidden = int(contract.get("mlp_hidden", cfg.mlp_vggt_hidden))
    mlp_layers = int(contract.get("mlp_layers", cfg.mlp_vggt_layers))
    embed_dim = int(contract.get("vggt_embed_dim", cfg.vggt_embed_dim))
    return CompositeSpec(
        branches=(
            BranchSpec(
                obs_key=HYBRID_IMAGE_KEY,
                module_name="cnn",
                make=lambda name: make_rgb_conv_encoder(
                    depth=depth, kernel_size=kernel, mults=mults, name=name
                ),
            ),
            BranchSpec(
                obs_key=HYBRID_WP_CP_KEY,
                module_name="vggt_mlp",
                make=lambda name: R2MLP(
                    hidden=mlp_hidden, layers=mlp_layers, out_dim=embed_dim, name=name
                ),
            ),
        ),
        fusion="gate",
    )


def _build_cnn_module(cfg: R2DreamerConfig) -> nn.Module:
    validate_encoder_config(cfg, ConvEncoder)
    return CompositeEncoder(build_cnn_composite(cfg))


def _build_hybrid_module(cfg: R2DreamerConfig) -> nn.Module:
    validate_hybrid_split(cfg)
    return CompositeEncoder(build_hybrid_composite(cfg))


# ---------------------------------------------------------------------------
# Mechanism/combination module builders (moved verbatim from factory.py)
# ---------------------------------------------------------------------------


def _build_wp_conv_module(cfg: R2DreamerConfig) -> nn.Module:
    # Full-res world-point map -> conv stack -> embed_dim (3D-53). Reuses the
    # RGB conv hyperparameters; symlog (not /255) handles the metric XYZ range.
    # No compute_dtype overlay: this conv path stays float32 by default.
    validate_encoder_config(cfg, ConvEncoder)
    kwargs = _contract_module_kwargs(cfg)
    if kwargs:
        return ConvEncoder(**kwargs)
    return ConvEncoder(**encoder_module_kwargs_from_config(cfg, ConvEncoder))


def _build_wp64_cnn_cp_mlp_module(cfg: R2DreamerConfig) -> nn.Module:
    validate_encoder_config(cfg, WP64CNNCPMLPEncoder)
    kwargs = _contract_module_kwargs(cfg)
    if kwargs:
        return WP64CNNCPMLPEncoder(**kwargs)
    return WP64CNNCPMLPEncoder(
        **encoder_module_kwargs_from_config(cfg, WP64CNNCPMLPEncoder)
    )


def _build_house_context_module(cfg: R2DreamerConfig) -> nn.Module:
    # The house-context variant keeps the WMHybridEncoder combination class:
    # its live/replay layout handling (wp_cp vs house_context key fallback) is
    # variant-owned and its param tree must stay checkpoint-compatible. It
    # shares the hybrid split validation (legacy _make_hybrid_encoder path).
    validate_hybrid_split(cfg)
    kwargs = _contract_module_kwargs(cfg)
    if kwargs:
        return WMHybridEncoder(**kwargs)
    return WMHybridEncoder(**encoder_module_kwargs_from_config(cfg, WMHybridEncoder))


def _build_house_points_module(
    cfg: R2DreamerConfig, default_cls: type = HousePointsCameraEncoder
) -> nn.Module:
    cls = _module_cls(cfg, default_cls)
    kwargs = _contract_module_kwargs(cfg)
    if kwargs:
        return cls(**kwargs)
    # The launcher module_kwargs_from_config owns the config->kwargs formula
    # (incl. house_point_norm and the hybrid CNN knobs). compute_dtype is the
    # only recipe-only overlay — a JAX dtype, not snapshot-serializable.
    return cls(
        **{
            **encoder_module_kwargs_from_config(cfg, cls),
            **compute_dtype_kwargs(cfg),
        }
    )


def _build_gnn_house_points_module(cfg: R2DreamerConfig) -> nn.Module:
    if cfg.encoder_module_cls is None:
        raise ValueError(f"unknown encoder_type {cfg.encoder_type!r}")
    return _build_house_points_module(cfg)


def _build_house_global_embedding_module(cfg: R2DreamerConfig) -> nn.Module:
    # PointNet reducer over VGGT global patch tokens + camera side branch.
    cls = _module_cls(cfg, HouseGlobalEmbeddingEncoder)
    kwargs = _contract_module_kwargs(cfg)
    if kwargs:
        return cls(**kwargs)
    return cls(**encoder_module_kwargs_from_config(cfg, cls))


def _build_mlp_module(
    cfg: R2DreamerConfig, default_cls: type = MLPEncoder
) -> nn.Module:
    # wp_cp + aggregator MLP encoders: depth from cfg.vggt_mlp_layers (3D-52).
    cls = _module_cls(cfg, default_cls)
    kwargs = _contract_module_kwargs(cfg)
    if kwargs:
        return cls(**kwargs)
    return cls(**encoder_module_kwargs_from_config(cfg, cls))


def _build_token_transformer_module(cfg: R2DreamerConfig) -> nn.Module:
    # Aggregator-token transformer AND the RGB-replay no-gate variants:
    # compute_dtype is always applied (the module opted into bfloat16 by
    # default), unlike the full_bf16-gated path.
    return TokenTransformerEncoder(
        **{
            **encoder_module_kwargs_from_config(cfg, TokenTransformerEncoder),
            "compute_dtype": compute_jnp_dtype(cfg.compute_dtype),
        }
    )


# ---------------------------------------------------------------------------
# Init dummies (moved verbatim from factory._dummy_encoder_obs)
# ---------------------------------------------------------------------------


def _dummy_flat(cfg: R2DreamerConfig):
    return jnp.zeros((1, *cfg.obs_shape))


def _dummy_hybrid(cfg: R2DreamerConfig):
    # CompositeEncoder hybrid consumes the structured dict (the packed-array
    # legacy layout is gone); init shapes match the WMHybridEncoder submodules
    # so encoder params stay bit-identical.
    return {
        HYBRID_IMAGE_KEY: jnp.zeros((1, 64, 64, 3), dtype=jnp.float32),
        HYBRID_WP_CP_KEY: jnp.zeros((1, cfg.vggt_feature_dim), dtype=jnp.float32),
    }


def _dummy_nogate_tokens(cfg: R2DreamerConfig, token_key: str):
    return {
        HYBRID_IMAGE_KEY: jnp.zeros((1, 64, 64, 3), dtype=jnp.float32),
        token_key: jnp.zeros(
            (1, cfg.vggt_token_count, cfg.vggt_token_dim),
            dtype=compute_jnp_dtype(cfg.compute_dtype),
        ),
    }


def _dummy_house_global_embedding(cfg: R2DreamerConfig):
    if not isinstance(cfg.obs_shape, Mapping):
        raise TypeError(f"{cfg.encoder_type} expects structured obs_shape")
    return {
        HYBRID_IMAGE_KEY: jnp.zeros(
            (1, *cfg.obs_shape[HYBRID_IMAGE_KEY]), dtype=jnp.float32
        ),
        GLOBAL_PATCH_TOKENS_KEY: jnp.zeros(
            (1, *cfg.obs_shape[GLOBAL_PATCH_TOKENS_KEY]), dtype=jnp.float32
        ),
    }


def _dummy_wp64(cfg: R2DreamerConfig):
    del cfg
    return {
        WORLD_POINTS_KEY: jnp.zeros((1, 64, 64, 3), dtype=jnp.float32),
        CAMERA_POSE_KEY: jnp.zeros((1, 9), dtype=jnp.float32),
    }


def _dummy_house_points(cfg: R2DreamerConfig, *, with_image: bool = False):
    if not isinstance(cfg.obs_shape, Mapping):
        raise TypeError(f"{cfg.encoder_type} expects structured obs_shape")
    dummy = {
        CAMERA_POSE_KEY: jnp.zeros((1, 9), dtype=jnp.float32),
        HOUSE_CONTEXT_KEY: jnp.zeros(
            (1, *cfg.obs_shape[HOUSE_CONTEXT_KEY]), dtype=jnp.float32
        ),
        HOUSE_CONTEXT_SIZE_KEY: jnp.zeros((), dtype=jnp.int32),
    }
    if with_image:
        dummy[HYBRID_IMAGE_KEY] = jnp.zeros((1, 64, 64, 3), dtype=jnp.float32)
    return dummy


# ---------------------------------------------------------------------------
# Adapter builders — delegate to the launcher encoder selections so the frozen
# extraction path is byte-for-byte the current one (adapters are an explicit
# non-goal of this refactor).
# ---------------------------------------------------------------------------


def _adapter_via_launcher(selection_name: str) -> Callable[[Any], Any]:
    def make(args: Any) -> Any:
        # Resolved at call time: the launcher selections live in the package
        # __init__, which is heavier than this module (VGGT extractor imports)
        # and is only needed when an adapter is actually built.
        launcher = import_module("src.r2dreamer.encoders")
        return getattr(launcher, selection_name).from_train_args(args).make_adapter()

    return make


def _cnn_adapter(args: Any = None) -> Any:
    del args  # CNN preparation needs no runtime args.
    module = import_module("src.r2dreamer.observation_preparation")
    return module.CNNObservationPreparation()


# ---------------------------------------------------------------------------
# The registry — one entry per encoder type
# ---------------------------------------------------------------------------


RECIPES: dict[str, EncoderRecipe] = {
    "cnn": EncoderRecipe(
        encoder_type="cnn",
        make_adapter=_cnn_adapter,
        build_module=_build_cnn_module,
        dummy_obs=_dummy_flat,
        rgb_key=HYBRID_IMAGE_KEY,
        composite=build_cnn_composite,
    ),
    "hybrid": EncoderRecipe(
        encoder_type="hybrid",
        make_adapter=_adapter_via_launcher("HybridEncoder"),
        build_module=_build_hybrid_module,
        dummy_obs=_dummy_hybrid,
        rgb_key=HYBRID_IMAGE_KEY,
        composite=build_hybrid_composite,
    ),
    "vggt": EncoderRecipe(
        encoder_type="vggt",
        make_adapter=_adapter_via_launcher("VGGTEncoder"),
        build_module=lambda cfg: _build_mlp_module(cfg, MLPEncoder),
        dummy_obs=_dummy_flat,
    ),
    "vggt_wp_cp_64": EncoderRecipe(
        encoder_type="vggt_wp_cp_64",
        make_adapter=_adapter_via_launcher("VGGTWPCP64Encoder"),
        build_module=lambda cfg: _build_mlp_module(cfg, MLPEncoder),
        dummy_obs=_dummy_flat,
    ),
    "vggt_aggregator_mlp": EncoderRecipe(
        encoder_type="vggt_aggregator_mlp",
        make_adapter=_adapter_via_launcher("VGGTAggregatorMLPEncoder"),
        build_module=lambda cfg: _build_mlp_module(cfg, VGGTAggregatorMLPEncoder),
        dummy_obs=_dummy_flat,
    ),
    "vggt_agg_raw": EncoderRecipe(
        encoder_type="vggt_agg_raw",
        make_adapter=_adapter_via_launcher("VGGTAggRawEncoder"),
        build_module=lambda cfg: _build_mlp_module(cfg, VGGTAggRawMLPEncoder),
        dummy_obs=_dummy_flat,
    ),
    "vggt_agg_token_transformer": EncoderRecipe(
        encoder_type="vggt_agg_token_transformer",
        make_adapter=_adapter_via_launcher("VGGTAggTokenTransformerEncoder"),
        build_module=_build_token_transformer_module,
        dummy_obs=_dummy_flat,
    ),
    "vggt_wp_dense_cnn": EncoderRecipe(
        encoder_type="vggt_wp_dense_cnn",
        make_adapter=_adapter_via_launcher("VGGTDenseWPEncoder"),
        build_module=_build_wp_conv_module,
        dummy_obs=_dummy_flat,
    ),
    "vggt_wp64_cnn_cp_mlp": EncoderRecipe(
        encoder_type="vggt_wp64_cnn_cp_mlp",
        make_adapter=_adapter_via_launcher("VGGTWP64CNNCPMLPEncoder"),
        build_module=_build_wp64_cnn_cp_mlp_module,
        dummy_obs=_dummy_wp64,
    ),
    "vggt_house_context": EncoderRecipe(
        encoder_type="vggt_house_context",
        make_adapter=_adapter_via_launcher("VGGTHouseContextEncoder"),
        build_module=_build_house_context_module,
        dummy_obs=_dummy_flat,
        rgb_key=HYBRID_IMAGE_KEY,
    ),
    "vggt_house_points_pose": EncoderRecipe(
        encoder_type="vggt_house_points_pose",
        make_adapter=_adapter_via_launcher("VGGTHousePointsPoseEncoder"),
        build_module=lambda cfg: _build_house_points_module(
            cfg, PointNetHousePointsCameraEncoder
        ),
        dummy_obs=_dummy_house_points,
    ),
    "vggt_hybrid_house_points_pose": EncoderRecipe(
        encoder_type="vggt_hybrid_house_points_pose",
        make_adapter=_adapter_via_launcher("VGGTHybridHousePointsPoseEncoder"),
        build_module=lambda cfg: _build_house_points_module(
            cfg, HybridHousePointsCameraEncoder
        ),
        dummy_obs=lambda cfg: _dummy_house_points(cfg, with_image=True),
    ),
    "gnn_house_points_pose": EncoderRecipe(
        encoder_type="gnn_house_points_pose",
        make_adapter=_adapter_via_launcher("GnnHousePointsPoseEncoder"),
        # The GNN Flax module classes arrive via cfg.encoder_module_cls from the
        # launcher spec; direct config construction without the override is an
        # error, exactly as under the legacy _resolve_encoder_cls dispatch.
        build_module=_build_gnn_house_points_module,
        dummy_obs=_dummy_house_points,
    ),
    "gnn_edge_house_points_pose": EncoderRecipe(
        encoder_type="gnn_edge_house_points_pose",
        make_adapter=_adapter_via_launcher("GnnEdgeHousePointsPoseEncoder"),
        build_module=_build_gnn_house_points_module,
        dummy_obs=_dummy_house_points,
    ),
    "vggt_house_full_tokens_nogate": EncoderRecipe(
        encoder_type="vggt_house_full_tokens_nogate",
        make_adapter=_adapter_via_launcher("VGGTHouseFullTokenNoGateEncoder"),
        build_module=_build_token_transformer_module,
        dummy_obs=lambda cfg: _dummy_nogate_tokens(cfg, FULL_TOKENS_KEY),
        rgb_key=HYBRID_IMAGE_KEY,
    ),
    "vggt_house_global_tokens_nogate": EncoderRecipe(
        encoder_type="vggt_house_global_tokens_nogate",
        make_adapter=_adapter_via_launcher("VGGTHouseGlobalTokenNoGateEncoder"),
        build_module=_build_token_transformer_module,
        dummy_obs=lambda cfg: _dummy_nogate_tokens(cfg, GLOBAL_TOKENS_KEY),
        rgb_key=HYBRID_IMAGE_KEY,
    ),
    "vggt_house_global_embedding": EncoderRecipe(
        encoder_type="vggt_house_global_embedding",
        make_adapter=_adapter_via_launcher("VGGTHouseGlobalEmbeddingEncoder"),
        build_module=_build_house_global_embedding_module,
        dummy_obs=_dummy_house_global_embedding,
        rgb_key=HYBRID_IMAGE_KEY,
    ),
}


def build_encoder_module(cfg: R2DreamerConfig) -> nn.Module:
    """Build the trainable encoder module for ``cfg.encoder_type``.

    The single cfg -> encoder-module entry point (replaces the factory
    dispatch).

    Raises:
        ValueError: For an unknown ``encoder_type`` or invalid config knobs.
    """
    recipe = RECIPES.get(cfg.encoder_type)
    if recipe is None:
        raise ValueError(f"unknown encoder_type {cfg.encoder_type!r}")
    return recipe.build_module(cfg)


def dummy_encoder_obs(cfg: R2DreamerConfig):
    """Build the batch-1 init dummy for ``cfg.encoder_type``.

    Raises:
        ValueError: For an unknown ``encoder_type``.
    """
    recipe = RECIPES.get(cfg.encoder_type)
    if recipe is None:
        raise ValueError(f"unknown encoder_type {cfg.encoder_type!r}")
    return recipe.dummy_obs(cfg)


# ---------------------------------------------------------------------------
# Startup consistency check
# ---------------------------------------------------------------------------


def infer_obs_spec(encoder_obs: Any) -> dict[str, tuple[int, ...]]:
    """Infer per-key event shapes from a single prepared frame (batch prefix 1).

    Args:
        encoder_obs: A prepared encoder observation — a dict of ``[1, *event]``
            arrays, or a bare ``[1, *event]`` array for single-key encoders.

    Returns:
        Mapping of observation key to event shape (leading batch axis dropped).
        A bare-array observation is reported under the single conventional
        image key.
    """
    if isinstance(encoder_obs, Mapping):
        return {k: tuple(v.shape[1:]) for k, v in encoder_obs.items()}
    return {HYBRID_IMAGE_KEY: tuple(encoder_obs.shape[1:])}


def check_branch_keys(composite: CompositeSpec, obs_keys: Any) -> None:
    """Fail fast when the composite's branch keys mismatch the inferred obs keys.

    Args:
        composite: The recipe's composite spec.
        obs_keys: Iterable of keys inferred from the first prepared frame
            (e.g. ``infer_obs_spec(...).keys()``).

    Raises:
        ValueError: If the sets differ — a recipe/adapter schema desync.
    """
    branch_keys = set(composite.branch_keys)
    inferred = set(obs_keys)
    if branch_keys != inferred:
        raise ValueError(
            "encoder branch/observation key mismatch: composite branches "
            f"{sorted(branch_keys)} != inferred obs keys {sorted(inferred)}"
        )
