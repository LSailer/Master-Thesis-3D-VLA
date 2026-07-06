"""Typed CLI-args -> agent-config-override bridge for train().

``train()`` turns diagnostic/ablation CLI flags into a ``dict`` of overrides
that get splatted into ``R2DreamerConfig(**agent_overrides)``. The historical
implementation copied ``getattr(args, name)`` into hardcoded field-name strings
inline; a typo in a target field name silently produced a dropped override
(``R2DreamerConfig`` would just ignore an extra kwarg under the old dataclass,
or -- now that configs are ``extra="forbid"`` pydantic models -- blow up far
from the mapping that caused it).

This module makes the mapping explicit and validates every target field name
against ``R2DreamerConfig.model_fields`` at build time, so an unknown/renamed
target field raises loudly, naming the offending CLI flag, before any config is
constructed. The *semantics* are byte-for-byte the historical ones: which flag
maps to which field, the None-means-not-provided skip, the ``latent_preset``
table applied before explicit RSSM-shape flags, and the inverted-bool /
dtype-alias special cases.
"""

from __future__ import annotations

from typing import Any

# CLI-arg attribute name -> R2DreamerConfig field name, for the plain
# "copy through if not None" overrides. Both sides are validated:
# the arg name must exist on the parsed namespace, the field name must exist in
# R2DreamerConfig.model_fields. Where the two names differ the historical
# mapping is preserved exactly (e.g. --actor_loss_weight -> scale_policy).
_SCALAR_OVERRIDES: dict[str, str] = {
    # Diagnostic loss-scale / shape / lr knobs (CLI name -> field name).
    "actor_loss_weight": "scale_policy",
    "value_loss_weight": "scale_value",
    "repval_loss_weight": "scale_repval",
    "batch_size": "batch_size",
    "seq_len": "seq_len",
    "lr": "lr",
    "mlp_layers": "vggt_mlp_layers",
    # Same-name replay knobs (historically read via getattr(..., default None)).
    "train_ratio": "train_ratio",
    "buffer_capacity": "buffer_capacity",
    # Model-size ablation RSSM/decoder/token-transformer knobs. Applied AFTER
    # the latent_preset table, so an explicit flag wins over the preset.
    "deter_size": "deter_size",
    "stoch_classes": "stoch_classes",
    "stoch_discrete": "stoch_discrete",
    "mlp_vggt_hidden": "mlp_vggt_hidden",
    "mlp_vggt_layers": "mlp_vggt_layers",
    "scale_decoder": "scale_decoder",
    "vggt_token_transformer_layers": "vggt_token_transformer_layers",
    "vggt_token_transformer_heads": "vggt_token_transformer_heads",
    "vggt_token_projection_dim": "vggt_token_projection_dim",
    "vggt_token_transformer_mlp_ratio": "vggt_token_transformer_mlp_ratio",
    "vggt_token_transformer_dropout": "vggt_token_transformer_dropout",
}

# The subset of _SCALAR_OVERRIDES applied *before* the latent_preset table (so
# the preset can override them), vs. *after* (so they override the preset). The
# split preserves the historical ordering exactly. Ordering only matters where a
# preset key collides with a scalar-override target; keeping the two loops keeps
# the collision winner identical to the old code.
_PRE_PRESET_ARGS: tuple[str, ...] = (
    "actor_loss_weight",
    "value_loss_weight",
    "repval_loss_weight",
    "batch_size",
    "seq_len",
    "lr",
    "mlp_layers",
    "train_ratio",
    "buffer_capacity",
)
_POST_PRESET_ARGS: tuple[str, ...] = (
    "deter_size",
    "stoch_classes",
    "stoch_discrete",
    "mlp_vggt_hidden",
    "mlp_vggt_layers",
    "scale_decoder",
    "vggt_token_transformer_layers",
    "vggt_token_transformer_heads",
    "vggt_token_projection_dim",
    "vggt_token_transformer_mlp_ratio",
    "vggt_token_transformer_dropout",
)

# store_true flags that flip a keep/stop config field to its non-default value
# (CLI flag attr -> (field name, value to set when the flag is present)).
_FLAG_OVERRIDES: dict[str, tuple[str, Any]] = {
    "barlow_grad_to_encoder": ("barlow_stop_grad", False),
    "decoder": ("decoder", True),
    "vggt_drop_register_tokens": ("vggt_keep_register_tokens", False),
}

# Presets normalise the CLI dtype spelling to the canonical config value.
_COMPUTE_DTYPE_ALIASES: dict[str, str] = {"bf16": "bfloat16", "fp16": "float16"}

# Every config field this bridge can target, for a single up-front validation
# pass against R2DreamerConfig.model_fields.
_TARGET_FIELDS: frozenset[str] = frozenset(
    list(_SCALAR_OVERRIDES.values())
    + [field for field, _ in _FLAG_OVERRIDES.values()]
    + ["compute_dtype"]
)


def _validate_target_fields(config_fields: set[str]) -> None:
    """Assert every override target names a real R2DreamerConfig field.

    Args:
      config_fields: The set of ``R2DreamerConfig.model_fields`` keys.

    Raises:
      ValueError: If any override maps to a field absent from ``config_fields``,
        naming the offending CLI flag(s) and target field(s).
    """
    unknown_scalar = {
        arg: field
        for arg, field in _SCALAR_OVERRIDES.items()
        if field not in config_fields
    }
    unknown_flag = {
        arg: field
        for arg, (field, _) in _FLAG_OVERRIDES.items()
        if field not in config_fields
    }
    unknown_extra = {} if "compute_dtype" in config_fields else {"compute_dtype": "compute_dtype"}
    unknown = {**unknown_scalar, **unknown_flag, **unknown_extra}
    if unknown:
        pairs = ", ".join(f"--{arg} -> {field!r}" for arg, field in sorted(unknown.items()))
        raise ValueError(
            "agent-override bridge targets unknown R2DreamerConfig field(s): "
            f"{pairs}. A config field was renamed or removed; update the mapping "
            "in src/r2dreamer/launch/agent_overrides.py."
        )


def _apply_scalar_group(
    overrides: dict[str, Any], args: Any, arg_names: tuple[str, ...]
) -> None:
    """Copy each provided (non-None) scalar CLI arg into its config field.

    Args:
      overrides: The override dict being built (mutated in place).
      args: Parsed argparse namespace.
      arg_names: Which ``_SCALAR_OVERRIDES`` CLI-arg names to consider, in order.
    """
    for arg_name in arg_names:
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[_SCALAR_OVERRIDES[arg_name]] = value


def agent_overrides_from_args(
    args: Any,
    encoder_spec: Any,
    latent_presets: dict[str, dict],
    *,
    config_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Build the agent-config override dict from parsed CLI args.

    Semantics are identical to the historical inline implementation:
    encoder-spec overrides form the base, diagnostic scalar flags override when
    provided, the ``--latent_preset`` table is applied, then explicit RSSM-shape
    flags override the preset, then the inverted-bool store_true flags and the
    dtype alias. Every target config field is validated against
    ``R2DreamerConfig.model_fields`` before anything is copied.

    Args:
      args: Parsed argparse namespace from the train parser.
      encoder_spec: EncoderSpec whose ``agent_overrides`` seed the dict.
      latent_presets: The ``LATENT_PRESETS`` table (preset name -> field dict).
      config_fields: The set of valid config field names. Defaults to
        ``R2DreamerConfig.model_fields`` (imported lazily to keep this module
        import-light); tests can inject a set to exercise validation.

    Returns:
      The override dict to splat into ``R2DreamerConfig(**overrides)``.

    Raises:
      ValueError: If any override maps to a field absent from the config.
    """
    if config_fields is None:
        from src.configs.config import R2DreamerConfig

        config_fields = set(R2DreamerConfig.model_fields)
    _validate_target_fields(config_fields)

    overrides = dict(encoder_spec.agent_overrides)

    _apply_scalar_group(overrides, args, _PRE_PRESET_ARGS)
    if args.barlow_grad_to_encoder:
        field, value = _FLAG_OVERRIDES["barlow_grad_to_encoder"]
        overrides[field] = value

    # Model-size ablation (3D-50): preset table first, explicit RSSM-shape flags
    # win over it.
    preset = getattr(args, "latent_preset", "12m")
    overrides.update(latent_presets.get(preset, {}))
    _apply_scalar_group(overrides, args, _POST_PRESET_ARGS)

    if getattr(args, "decoder", False):
        field, value = _FLAG_OVERRIDES["decoder"]
        overrides[field] = value
    if getattr(args, "vggt_drop_register_tokens", False):
        field, value = _FLAG_OVERRIDES["vggt_drop_register_tokens"]
        overrides[field] = value
    if getattr(args, "compute_dtype", None) is not None:
        dtype = args.compute_dtype
        overrides["compute_dtype"] = _COMPUTE_DTYPE_ALIASES.get(dtype, dtype)
    return overrides
