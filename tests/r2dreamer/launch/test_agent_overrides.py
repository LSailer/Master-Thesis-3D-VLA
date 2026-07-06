"""Unit tests for the CLI-args -> agent-config-override bridge.

Covers a valid override mapping through unchanged (including the renamed
``--actor_loss_weight -> scale_policy`` pair) and the ``_validate_target_fields``
guard that raises, naming the offending ``--flag -> 'field'`` pair, when a
target config field is missing.
"""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

from types import SimpleNamespace

import pytest

from src.r2dreamer.launch.agent_overrides import (
    _TARGET_FIELDS,
    _validate_target_fields,
    agent_overrides_from_args,
)

# Every field the bridge can target; a valid config-fields set must be a
# superset of this. Tests drop a single field from it to trigger validation.
_ALL_TARGET_FIELDS = set(_TARGET_FIELDS)


class _EncoderSpec:
    agent_overrides: dict = {}


def _args(**overrides):
    """Build a parsed-args stand-in with the fields the bridge reads."""
    base = dict(
        actor_loss_weight=None,
        value_loss_weight=None,
        repval_loss_weight=None,
        batch_size=None,
        seq_len=None,
        lr=None,
        mlp_layers=None,
        train_ratio=None,
        buffer_capacity=None,
        deter_size=None,
        stoch_classes=None,
        stoch_discrete=None,
        mlp_vggt_hidden=None,
        mlp_vggt_layers=None,
        scale_decoder=None,
        vggt_token_transformer_layers=None,
        vggt_token_transformer_heads=None,
        vggt_token_projection_dim=None,
        vggt_token_transformer_mlp_ratio=None,
        vggt_token_transformer_dropout=None,
        barlow_grad_to_encoder=False,
        decoder=False,
        vggt_drop_register_tokens=False,
        compute_dtype=None,
        latent_preset="12m",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_actor_loss_weight_maps_to_scale_policy():
    """The renamed --actor_loss_weight flag lands in the scale_policy field."""
    overrides = agent_overrides_from_args(
        _args(actor_loss_weight=0.25),
        _EncoderSpec(),
        latent_presets={},
        config_fields=set(_ALL_TARGET_FIELDS),
    )
    assert overrides["scale_policy"] == 0.25
    # An unprovided (None) flag is skipped, not copied as None.
    assert "scale_value" not in overrides


def test_none_flags_are_skipped():
    """Every flag left at None produces no override entry."""
    overrides = agent_overrides_from_args(
        _args(),
        _EncoderSpec(),
        latent_presets={},
        config_fields=set(_ALL_TARGET_FIELDS),
    )
    assert overrides == {}


def test_validate_target_fields_raises_naming_offender():
    """An unknown target field raises, naming the offending --flag -> 'field'."""
    # Drop scale_policy so --actor_loss_weight becomes the sole offender.
    config_fields = set(_ALL_TARGET_FIELDS) - {"scale_policy"}
    with pytest.raises(ValueError) as excinfo:
        _validate_target_fields(config_fields)
    message = str(excinfo.value)
    assert "--actor_loss_weight -> 'scale_policy'" in message


def test_agent_overrides_from_args_raises_on_missing_field():
    """The public entry point surfaces the same validation error."""
    config_fields = set(_ALL_TARGET_FIELDS) - {"scale_policy"}
    with pytest.raises(ValueError, match="--actor_loss_weight -> 'scale_policy'"):
        agent_overrides_from_args(
            _args(),
            _EncoderSpec(),
            latent_presets={},
            config_fields=config_fields,
        )
