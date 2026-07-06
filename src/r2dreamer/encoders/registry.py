"""Single source of truth for Dreamer Encoder Module dispatch.

Three call sites used to maintain independent ``encoder_type`` -> Flax module
mappings: ``agent.py`` (module construction + dummy-obs discovery),
``observation_preparation/module_factory.py`` (contract-driven construction),
and ``observation_preparation/contracts.py`` (constructor-kwargs derivation).
This module bundles all of that per encoder type into one
:class:`EncoderRegistryEntry` so the three call sites delegate instead of
re-implementing the mapping.

Dispatch is by Flax module *class* (via an MRO walk, see
``resolve_registry_entry``), not by ``encoder_type`` string, because several
``encoder_type`` values share one module class (e.g. ``"vggt"`` and
``"vggt_wp_cp_64"`` both build :class:`~src.r2dreamer.encoders.mlp.MLPEncoder`)
and some module classes are shared by string-unrelated subclasses (the GNN
house encoders subclass ``HousePointsCameraEncoder`` and must inherit its
kwargs/validation/diagnostics). A class-keyed, MRO-aware registry is the only
structure that captures both facts without duplicated branches.

Location and dependency direction: this module lives under ``encoders/``
(not ``launch/registries.py``) because it is purely about Flax module
classes/kwargs/diagnostics — a model-layer concern. ``launch/registries.py``
is a different, higher-level registry keyed by ``encoder_type`` string that
maps to launcher-side ``Encoder`` selections (observation adapters, VGGT
extractor wiring); those already resolve their own ``module_cls`` (often via
``observation_preparation.vggt.VGGT_DREAMER_SPECS``) and are consumed by
launch-time code, not by ``agent.py``. Putting the encoder-module registry in
``launch/`` would force ``agent.py`` (a model-layer module with no launcher
dependency today) to import from ``launch/``, which is the wrong direction —
`launch/` orchestrates training/eval runs built on top of the model layer, it
is not depended on by it. Keeping this registry in ``encoders/`` lets
``launch/registries.py`` import from it later if useful, without ever
reversing that edge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import jax.numpy as jnp

from src.r2dreamer.encoders.cnn import ConvEncoder
from src.r2dreamer.encoders.constants import AGG_REGISTER_TOKENS, HYBRID_RGB_DIM
from src.r2dreamer.encoders.mlp import (
    HouseGlobalEmbeddingEncoder,
    HousePointsCameraEncoder,
    HybridEncoder,
    MLPEncoder,
    VGGTAggRawMLPEncoder,
    VGGTAggregatorMLPEncoder,
    WP64CNNCPMLPEncoder,
)
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder
from src.r2dreamer.observation_keys import (
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
from src.shared.dtypes import compute_jnp_dtype


class EncoderModule(Protocol):
    """Structural type for a Dreamer Encoder Module (Flax module surface).

    Agent code type-hints against this Protocol instead of importing each
    concrete encoder class, matching the DIP-via-structural-subtyping
    convention used elsewhere in this codebase.
    """

    def init(self, rng: Any, *args: Any, **kwargs: Any) -> Any:
        """Initialize and return this module's parameter pytree."""
        ...

    def apply(self, variables: Any, *args: Any, **kwargs: Any) -> Any:
        """Run this module's forward pass against a parameter pytree."""
        ...


def _tuple_value(config: Any, name: str) -> tuple[int, ...]:
    return tuple(getattr(config, name))


def _contract_kwargs_from_config(config: Any) -> dict[str, Any]:
    """Read pre-resolved kwargs off a durable Encoder Input Contract snapshot.

    Args:
      config: Effective agent config (``R2DreamerConfig`` or equivalent).

    Returns:
      Normalized constructor kwargs, or ``{}`` when no snapshot is attached.
    """
    # Imported lazily to avoid a module-level cycle: contracts.py delegates
    # kwargs-building to this registry, so this registry cannot import
    # contracts.py at top level.
    from src.r2dreamer.observation_preparation.contracts import (
        normalize_encoder_module_kwargs,
    )

    snapshot = getattr(config, "encoder_input_contract", None)
    if snapshot is None:
        return {}
    return normalize_encoder_module_kwargs(snapshot.get("encoder_module_kwargs", {}))


# ---------------------------------------------------------------------------
# Per-entry kwargs builders
# ---------------------------------------------------------------------------


def _conv_kwargs(config: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "depth": int(config.encoder_depth),
        "kernel_size": int(config.encoder_kernel),
        "mults": _tuple_value(config, "encoder_mults"),
    }
    if getattr(config, "encoder_type", None) == "vggt_wp_dense_cnn":
        kwargs.update(
            {
                "input_kind": "world_points",
                "embed_dim": int(config.vggt_embed_dim),
            }
        )
    return kwargs


def _wp64_cnn_cp_mlp_kwargs(config: Any) -> dict[str, Any]:
    return {
        "embed_dim": int(config.vggt_embed_dim),
        "conv_depth": int(config.encoder_depth),
        "conv_kernel": int(config.encoder_kernel),
        "conv_mults": _tuple_value(config, "encoder_mults"),
        "cp_hidden": int(config.mlp_vggt_hidden),
        "cp_layers": int(config.mlp_vggt_layers),
    }


def _hybrid_kwargs(config: Any) -> dict[str, Any]:
    return {
        "cnn_depth": int(config.encoder_depth),
        "cnn_kernel": int(config.encoder_kernel),
        "cnn_mults": _tuple_value(config, "encoder_mults"),
        "vggt_embed_dim": int(config.vggt_embed_dim),
        "mlp_hidden": int(config.mlp_vggt_hidden),
        "mlp_layers": int(config.mlp_vggt_layers),
        "vggt_dim": int(config.vggt_feature_dim),
    }


def _house_points_camera_kwargs(config: Any) -> dict[str, Any]:
    return {
        "embed_dim": int(config.vggt_embed_dim),
        "camera_hidden": int(config.mlp_vggt_hidden),
        "camera_layers": int(config.mlp_vggt_layers),
        "point_hidden": int(config.mlp_vggt_hidden),
        "point_layers": int(config.mlp_vggt_layers),
    }


def _house_global_embedding_kwargs(config: Any) -> dict[str, Any]:
    # token_dim (1024) and num_patch_tokens (1369) use the module defaults
    # — they are fixed by the VGGT global-half token layout.
    return {
        "embed_dim": int(config.vggt_embed_dim),
        "reducer_hidden": int(config.mlp_vggt_hidden),
        "reducer_layers": int(config.mlp_vggt_layers),
        "camera_hidden": int(config.mlp_vggt_hidden),
        "camera_layers": int(config.mlp_vggt_layers),
    }


def _house_global_embedding_kwargs_with_explicit_tokens(config: Any) -> dict[str, Any]:
    # Historical divergence, preserved exactly: R2DreamerAgent's direct
    # construction path additionally derives token_dim/num_patch_tokens from
    # cfg.vggt_token_count/vggt_token_dim (fixed by the VGGT global-half token
    # layout: num_patch_tokens = vggt_token_count - (1 camera + register
    # tokens)), instead of relying on the module's defaults (1024/1369).
    # contracts.py's derivation never included these two keys. Prod sets
    # vggt_token_dim=1024 / vggt_token_count=1374 via agent_overrides; tests
    # inject small dims and rely on this explicit computation.
    num_patch_tokens = int(config.vggt_token_count) - (1 + AGG_REGISTER_TOKENS)
    return {
        **_house_global_embedding_kwargs(config),
        "token_dim": int(config.vggt_token_dim),
        "num_patch_tokens": num_patch_tokens,
    }


def _token_transformer_kwargs(config: Any) -> dict[str, Any]:
    common = {
        "embed_dim": int(config.vggt_embed_dim),
        "token_dim": int(config.vggt_token_dim),
        "num_tokens": int(config.vggt_token_count),
        "layers": int(config.vggt_token_transformer_layers),
        "heads": int(config.vggt_token_transformer_heads),
        "mlp_ratio": int(config.vggt_token_transformer_mlp_ratio),
        "dropout": float(config.vggt_token_transformer_dropout),
    }
    if getattr(config, "encoder_type", None) == "vggt_agg_token_transformer":
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
    if getattr(config, "encoder_type", None) == "vggt_house_global_tokens_nogate":
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
        "cnn_mults": _tuple_value(config, "encoder_mults"),
    }


def _token_transformer_kwargs_direct(config: Any) -> dict[str, Any]:
    # Historical divergences, preserved exactly: R2DreamerAgent's direct
    # construction path (no durable Encoder Input Contract snapshot attached)
    # has always threaded compute_dtype through to the token Transformer so
    # bfloat16 configs actually run attention/dense math in bfloat16, and (for
    # the vggt_agg_token_transformer branch specifically) never passed
    # `dropout`, relying on the module's 0.0 default. contracts.py's
    # derivation (module_factory/checkpointing/launch/train) never included
    # compute_dtype and always passes dropout explicitly for both branches;
    # these are existing inconsistencies this refactor preserves rather than
    # fixes.
    kwargs = dict(_token_transformer_kwargs(config))
    if getattr(config, "encoder_type", None) == "vggt_agg_token_transformer":
        kwargs.pop("dropout", None)
    kwargs["compute_dtype"] = compute_jnp_dtype(config.compute_dtype)
    return kwargs


def _mlp_kwargs(config: Any) -> dict[str, Any]:
    return {
        "embed_dim": int(config.vggt_embed_dim),
        "hidden": int(config.vggt_embed_dim),
        "num_layers": int(config.vggt_mlp_layers),
    }


# ---------------------------------------------------------------------------
# Per-entry config validators (no-op unless noted)
# ---------------------------------------------------------------------------


def _validate_no_mlp_layers_knob(config: Any, cls: type) -> None:
    # Fail loud instead of silently dropping the knob: conv encoders have no
    # MLP depth, so a non-default vggt_mlp_layers here is a misconfiguration.
    if config.vggt_mlp_layers != 1:
        raise ValueError(
            f"vggt_mlp_layers={config.vggt_mlp_layers} has no effect on "
            f"{cls.__name__} (a conv encoder, no MLP blocks). Only the 'vggt' and "
            f"'vggt_aggregator_mlp' encoders consume vggt_mlp_layers; leave it at 1 "
            f"for cnn / vggt_wp_dense_cnn."
        )


def _validate_hybrid_obs_shape(config: Any, cls: type) -> None:
    expected_shape = (HYBRID_RGB_DIM + config.vggt_feature_dim,)
    if not isinstance(config.obs_shape, tuple):
        raise ValueError(f"hybrid expects flat obs_shape, got {config.obs_shape}")
    if not (
        config.obs_shape == expected_shape
        and config.obs_shape[0] - config.vggt_feature_dim == HYBRID_RGB_DIM
    ):
        raise ValueError(
            "hybrid obs_shape/split mismatch: expected "
            f"{expected_shape} with vggt_feature_dim={config.vggt_feature_dim}, "
            f"got obs_shape={config.obs_shape}, "
            f"vggt_feature_dim={config.vggt_feature_dim}"
        )


def _no_validation(config: Any, cls: type) -> None:
    del config, cls


# ---------------------------------------------------------------------------
# Per-entry dummy-observation builders (agent init embed-size discovery)
# ---------------------------------------------------------------------------


def _dummy_obs_default(config: Any) -> Any:
    return jnp.zeros((1, *config.obs_shape))


def _dummy_obs_wp64_cnn_cp_mlp(config: Any) -> Any:
    del config
    return {
        WORLD_POINTS_KEY: jnp.zeros((1, 3, 64, 64), dtype=jnp.float32),
        CAMERA_POSE_KEY: jnp.zeros((1, 9), dtype=jnp.float32),
    }


def _dummy_obs_house_points_camera(config: Any) -> Any:
    if not isinstance(config.obs_shape, Mapping):
        raise TypeError(f"{config.encoder_type} expects structured obs_shape")
    return {
        CAMERA_POSE_KEY: jnp.zeros((1, 9), dtype=jnp.float32),
        HOUSE_CONTEXT_KEY: jnp.zeros(
            (1, *config.obs_shape[HOUSE_CONTEXT_KEY]), dtype=jnp.float32
        ),
        HOUSE_CONTEXT_SIZE_KEY: jnp.zeros((), dtype=jnp.int32),
    }


def _dummy_obs_house_global_embedding(config: Any) -> Any:
    if not isinstance(config.obs_shape, Mapping):
        raise TypeError(f"{config.encoder_type} expects structured obs_shape")
    return {
        HYBRID_IMAGE_KEY: jnp.zeros(
            (1, *config.obs_shape[HYBRID_IMAGE_KEY]), dtype=jnp.float32
        ),
        CAMERA_TOKEN_GLOBAL_KEY: jnp.zeros(
            (1, *config.obs_shape[CAMERA_TOKEN_GLOBAL_KEY]), dtype=jnp.float32
        ),
        GLOBAL_PATCH_TOKENS_KEY: jnp.zeros(
            (1, *config.obs_shape[GLOBAL_PATCH_TOKENS_KEY]), dtype=jnp.float32
        ),
    }


def _dummy_obs_token_transformer(config: Any) -> Any:
    if config.encoder_type == "vggt_house_global_tokens_nogate":
        return {
            HYBRID_IMAGE_KEY: jnp.zeros((1, 3, 64, 64), dtype=jnp.float32),
            GLOBAL_TOKENS_KEY: jnp.zeros(
                (1, config.vggt_token_count, config.vggt_token_dim),
                dtype=compute_jnp_dtype(config.compute_dtype),
            ),
        }
    if config.encoder_type == "vggt_house_full_tokens_nogate":
        return {
            HYBRID_IMAGE_KEY: jnp.zeros((1, 3, 64, 64), dtype=jnp.float32),
            FULL_TOKENS_KEY: jnp.zeros(
                (1, config.vggt_token_count, config.vggt_token_dim),
                dtype=compute_jnp_dtype(config.compute_dtype),
            ),
        }
    return _dummy_obs_default(config)


# ---------------------------------------------------------------------------
# Per-entry loss diagnostics hooks (Task 3: de-branch _loss_fn)
# ---------------------------------------------------------------------------


def _no_diagnostics(
    metrics: dict[str, Any],
    *,
    cfg: Any,
    params: dict[str, Any],
    forward: Any,
    B: int,
    T: int,
) -> None:
    """No-op diagnostics hook for encoders with nothing extra to log."""
    del metrics, cfg, params, forward, B, T


def _hybrid_contribution_diagnostics(
    metrics: dict[str, Any],
    *,
    cfg: Any,
    params: dict[str, Any],
    forward: Any,
    B: int,
    T: int,
) -> None:
    """Log CNN-vs-VGGT contribution shares for gated hybrid encoders.

    Reuses the already-computed fused embed instead of a second encoder
    forward: ``embed == concat([cnn_e, gate * vggt_mlp(...)])``, so the
    leading ``cnn_dim`` columns are the CNN branch and the rest are the gated
    VGGT branch. The raw gate scalar is read straight from params.

    Args:
      metrics: Metrics dict to update in place.
      cfg: Effective agent config.
      params: Full agent parameter pytree (only ``params["encoder"]`` is read).
      forward: The shared ``WorldModelForward`` for this train step.
      B: Batch size.
      T: Sequence length.
    """
    embed_flat = forward.embed.reshape(B * T, -1)
    cnn_dim = embed_flat.shape[-1] - cfg.vggt_embed_dim
    cnn_e = embed_flat[:, :cnn_dim]
    vggt_e = embed_flat[:, cnn_dim:]
    gate = params["encoder"]["params"]["gate"]
    cnn_l2 = jnp.sqrt(jnp.mean(jnp.sum(cnn_e**2, axis=-1)))
    vggt_l2 = jnp.sqrt(jnp.mean(jnp.sum(vggt_e**2, axis=-1)))
    denom = cnn_l2 + vggt_l2 + 1e-8
    metrics["hybrid/gate"] = gate
    metrics["hybrid/cnn_l2"] = cnn_l2
    metrics["hybrid/vggt_l2"] = vggt_l2
    metrics["hybrid/cnn_std"] = jnp.std(cnn_e)
    metrics["hybrid/vggt_std"] = jnp.std(vggt_e)
    metrics["hybrid/cnn_frac"] = cnn_l2 / denom
    metrics["hybrid/vggt_frac"] = vggt_l2 / denom


DiagnosticsHook = Callable[..., None]
KwargsBuilder = Callable[[Any], dict[str, Any]]
ConfigValidator = Callable[[Any, type], None]
DummyObsBuilder = Callable[[Any], Any]


@dataclass(frozen=True)
class EncoderRegistryEntry:
    """One encoder module's construction + validation + diagnostics bundle.

    Attributes:
      module_cls: The Flax module class this entry constructs.
      kwargs_from_config: Builds constructor kwargs from an effective config.
      validate_config: Raises on a misconfiguration specific to this module
        (e.g. an MLP-only knob set on a conv encoder). No-op by default.
      dummy_obs: Builds a batch-size-1 dummy observation for embed-size
        discovery during agent init. Defaults to a flat ``obs_shape`` zeros
        array.
      diagnostics: Optional per-step loss-diagnostics hook invoked from
        ``_loss_fn`` with ``(metrics, cfg=..., params=..., forward=...,
        B=..., T=...)``. No-op by default.
    """

    module_cls: type
    kwargs_from_config: KwargsBuilder
    validate_config: ConfigValidator = _no_validation
    dummy_obs: DummyObsBuilder = _dummy_obs_default
    diagnostics: DiagnosticsHook = _no_diagnostics
    direct_kwargs_from_config: KwargsBuilder | None = field(default=None)

    def resolved_kwargs_builder(self, *, direct: bool) -> KwargsBuilder:
        """Return the kwargs builder to use for this construction path.

        Args:
          direct: True for ``R2DreamerAgent``'s direct-construction path
            (legacy ``R2DreamerConfig`` with no durable Encoder Input
            Contract snapshot attached); False for the
            contract-snapshot-driven path used by ``module_factory``,
            checkpointing, and ``launch/train.py``.

        Returns:
          The kwargs builder to call with the effective config.
        """
        if direct and self.direct_kwargs_from_config is not None:
            return self.direct_kwargs_from_config
        return self.kwargs_from_config


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

# Keyed by Flax module class (not encoder_type string): several encoder_type
# strings share one module class ("vggt" / "vggt_wp_cp_64" -> MLPEncoder), and
# dispatch must also cover class hierarchies not enumerated here (the GNN house
# encoders subclass HousePointsCameraEncoder) — see resolve_registry_entry.
_REGISTRY: dict[type, EncoderRegistryEntry] = {
    ConvEncoder: EncoderRegistryEntry(
        module_cls=ConvEncoder,
        kwargs_from_config=_conv_kwargs,
        validate_config=_validate_no_mlp_layers_knob,
    ),
    WP64CNNCPMLPEncoder: EncoderRegistryEntry(
        module_cls=WP64CNNCPMLPEncoder,
        kwargs_from_config=_wp64_cnn_cp_mlp_kwargs,
        validate_config=_validate_no_mlp_layers_knob,
        dummy_obs=_dummy_obs_wp64_cnn_cp_mlp,
    ),
    HybridEncoder: EncoderRegistryEntry(
        module_cls=HybridEncoder,
        kwargs_from_config=_hybrid_kwargs,
        validate_config=_validate_hybrid_obs_shape,
        diagnostics=_hybrid_contribution_diagnostics,
    ),
    HousePointsCameraEncoder: EncoderRegistryEntry(
        module_cls=HousePointsCameraEncoder,
        kwargs_from_config=_house_points_camera_kwargs,
        dummy_obs=_dummy_obs_house_points_camera,
    ),
    HouseGlobalEmbeddingEncoder: EncoderRegistryEntry(
        module_cls=HouseGlobalEmbeddingEncoder,
        kwargs_from_config=_house_global_embedding_kwargs,
        direct_kwargs_from_config=_house_global_embedding_kwargs_with_explicit_tokens,
        dummy_obs=_dummy_obs_house_global_embedding,
    ),
    TokenTransformerEncoder: EncoderRegistryEntry(
        module_cls=TokenTransformerEncoder,
        kwargs_from_config=_token_transformer_kwargs,
        direct_kwargs_from_config=_token_transformer_kwargs_direct,
        dummy_obs=_dummy_obs_token_transformer,
    ),
}

# encoder_type string -> Flax module class, for callers (agent.py) that only
# have the documented type name and no EncoderSpec/contract yet.
_ENCODER_TYPE_TO_MODULE_CLS: dict[str, type] = {
    "cnn": ConvEncoder,
    "vggt": MLPEncoder,
    "vggt_wp_cp_64": MLPEncoder,  # same MLP module, finer WP grid (obs 12297)
    "vggt_aggregator_mlp": VGGTAggregatorMLPEncoder,
    "vggt_agg_raw": VGGTAggRawMLPEncoder,
    "vggt_agg_token_transformer": TokenTransformerEncoder,
    "vggt_wp_dense_cnn": ConvEncoder,
    "vggt_wp64_cnn_cp_mlp": WP64CNNCPMLPEncoder,
    "hybrid": HybridEncoder,
    "vggt_house_context": HybridEncoder,
    "vggt_house_points_pose": HousePointsCameraEncoder,
    "vggt_house_full_tokens_nogate": TokenTransformerEncoder,
    "vggt_house_global_tokens_nogate": TokenTransformerEncoder,
    "vggt_house_global_embedding": HouseGlobalEmbeddingEncoder,
}

# encoder_type strings whose Observation Preparation carries an RGB modality a
# ConvDecoder can reconstruct (decoder=True is only valid for these). Single
# source of truth for the three former hand-synced copies in
# agent_modules.py / observation_preparation/module_factory.py /
# decoder_targets.py.
RGB_BEARING_ENCODER_TYPES: frozenset[str] = frozenset(
    {
        "cnn",
        "hybrid",
        "vggt_house_context",
        "vggt_house_full_tokens_nogate",
        "vggt_house_global_tokens_nogate",
        "vggt_house_global_embedding",
    }
)


def encoder_type_has_rgb_target(encoder_type: str) -> bool:
    """Report whether ``encoder_type`` carries an RGB decoder target.

    Args:
      encoder_type: One of the documented ``R2DreamerConfig.encoder_type``
        values.

    Returns:
      True if the encoder's Observation Preparation exposes an RGB modality a
      ConvDecoder can reconstruct (i.e. ``decoder=True`` is valid for it).
    """
    return encoder_type in RGB_BEARING_ENCODER_TYPES


def resolve_module_cls_from_type(encoder_type: str) -> type:
    """Map a documented ``encoder_type`` string to its Flax module class.

    Args:
      encoder_type: One of the documented ``R2DreamerConfig.encoder_type``
        values.

    Returns:
      The Flax Encoder Module class for ``encoder_type``.

    Raises:
      ValueError: If ``encoder_type`` is not recognized.
    """
    cls = _ENCODER_TYPE_TO_MODULE_CLS.get(encoder_type)
    if cls is None:
        raise ValueError(f"unknown encoder_type {encoder_type!r}")
    return cls


def resolve_registry_entry(module_cls: type) -> EncoderRegistryEntry:
    """Resolve the registry entry governing ``module_cls``.

    Subclasses (e.g. the GNN variants of ``HousePointsCameraEncoder``)
    inherit their parent's constructor kwargs/validation/diagnostics, so
    dispatch walks the MRO and returns the first ancestor with a dedicated
    entry — most-derived-first, matching an ``issubclass`` chain. Classes
    with no dedicated ancestor (the plain MLP family) fall through to a
    generic MLP-kwargs entry built on demand for ``module_cls`` itself.

    Args:
      module_cls: The Encoder Module class to resolve.

    Returns:
      The bundled :class:`EncoderRegistryEntry` for ``module_cls``.
    """
    for base in module_cls.__mro__:
        entry = _REGISTRY.get(base)
        if entry is not None:
            return entry
    return EncoderRegistryEntry(
        module_cls=module_cls,
        kwargs_from_config=_mlp_kwargs,
    )


def make_encoder_kwargs(
    config: Any, module_cls: type, *, direct: bool = False
) -> dict[str, Any]:
    """Resolve Encoder Module constructor kwargs from an effective config.

    Args:
      config: Effective agent config (``R2DreamerConfig`` or equivalent).
      module_cls: The Encoder Module class to be constructed.
      direct: True for ``R2DreamerAgent``'s direct-construction path (see
        :meth:`EncoderRegistryEntry.resolved_kwargs_builder`); False (default)
        for the contract-snapshot-driven path.

    Returns:
      Constructor kwargs for ``module_cls``.
    """
    entry = resolve_registry_entry(module_cls)
    return entry.resolved_kwargs_builder(direct=direct)(config)


def validate_encoder_config(config: Any, module_cls: type) -> None:
    """Raise when ``config`` misconfigures ``module_cls``.

    Args:
      config: Effective agent config (``R2DreamerConfig`` or equivalent).
      module_cls: The Encoder Module class that would be constructed.

    Raises:
      ValueError: If the config is invalid for ``module_cls``.
    """
    resolve_registry_entry(module_cls).validate_config(config, module_cls)


def dummy_encoder_obs(config: Any, module_cls: type) -> Any:
    """Build a batch-size-1 dummy observation for embed-size discovery.

    Args:
      config: Effective agent config (``R2DreamerConfig`` or equivalent).
      module_cls: The Encoder Module class the dummy observation targets.

    Returns:
      A dummy observation (array or dict of arrays) shaped for one batch
      element, matching ``config.obs_shape``/``encoder_type``.
    """
    return resolve_registry_entry(module_cls).dummy_obs(config)


def encoder_loss_diagnostics(
    module_cls: type,
    metrics: dict[str, Any],
    *,
    cfg: Any,
    params: dict[str, Any],
    forward: Any,
    B: int,
    T: int,
) -> None:
    """Run ``module_cls``'s per-step loss diagnostics hook, if any.

    Args:
      module_cls: The Encoder Module class in use for this agent.
      metrics: Metrics dict to update in place.
      cfg: Effective agent config.
      params: Full agent parameter pytree.
      forward: The shared ``WorldModelForward`` for this train step.
      B: Batch size.
      T: Sequence length.
    """
    resolve_registry_entry(module_cls).diagnostics(
        metrics, cfg=cfg, params=params, forward=forward, B=B, T=T
    )


def make_encoder_module(config: Any, *, direct: bool = False) -> EncoderModule:
    """Instantiate a Dreamer Encoder Module from an effective config.

    Resolves the module class from ``config.encoder_module_cls`` when set
    (launcher-provided ``EncoderSpec``), else from ``config.encoder_type``.
    Constructor kwargs prefer a durable Encoder Input Contract snapshot
    (``config.encoder_input_contract``) when present, falling back to
    kwargs derived from the effective config.

    Args:
      config: Effective agent config (``R2DreamerConfig`` or equivalent).
      direct: True when called from ``R2DreamerAgent``'s direct-construction
        path (see :meth:`EncoderRegistryEntry.resolved_kwargs_builder`); only
        affects the config-derived kwargs fallback, not the contract-snapshot
        path.

    Returns:
      The constructed Encoder Module instance.

    Raises:
      ValueError: If the config is invalid for the resolved module, or if
        ``encoder_type`` is unrecognized and no explicit module class is set.
    """
    module_cls = config.encoder_module_cls
    if module_cls is None:
        module_cls = resolve_module_cls_from_type(config.encoder_type)
    validate_encoder_config(config, module_cls)
    kwargs = _contract_kwargs_from_config(config)
    if not kwargs:
        kwargs = make_encoder_kwargs(config, module_cls, direct=direct)
    return module_cls(**kwargs)
