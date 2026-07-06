"""Direct unit tests for the encoder-module registry.

Covers the class-keyed MRO dispatch, the direct-vs-snapshot kwargs divergence
for the token Transformer, the unknown-encoder_type error, and an import-level
consistency check that the three ``RGB_BEARING_ENCODER_TYPES`` call sites all
agree with the shared registry set (locking F1's consolidation).
"""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import pytest

from src.configs.config import R2DreamerConfig
from src.r2dreamer.encoders.mlp import HousePointsCameraEncoder
from src.r2dreamer.encoders.registry import (
    RGB_BEARING_ENCODER_TYPES,
    encoder_type_has_rgb_target,
    make_encoder_kwargs,
    resolve_module_cls_from_type,
    resolve_registry_entry,
)
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder


def test_resolve_registry_entry_walks_mro_to_parent():
    """A subclass with no dedicated entry resolves to its parent's entry."""

    class _SubHousePointsCameraEncoder(HousePointsCameraEncoder):
        pass

    entry = resolve_registry_entry(_SubHousePointsCameraEncoder)
    parent_entry = resolve_registry_entry(HousePointsCameraEncoder)
    assert entry is parent_entry
    assert entry.module_cls is HousePointsCameraEncoder


def test_token_transformer_direct_vs_snapshot_kwargs_diverge():
    """Direct path threads compute_dtype; the snapshot path omits it."""
    cfg = R2DreamerConfig(
        encoder_type="vggt_house_full_tokens_nogate", obs_shape=(3, 64, 64)
    )

    direct_kwargs = make_encoder_kwargs(cfg, TokenTransformerEncoder, direct=True)
    snapshot_kwargs = make_encoder_kwargs(cfg, TokenTransformerEncoder, direct=False)

    assert "compute_dtype" in direct_kwargs
    assert "compute_dtype" not in snapshot_kwargs
    # The two paths must otherwise agree on every shared key.
    for key, value in snapshot_kwargs.items():
        assert direct_kwargs[key] == value


def test_resolve_module_cls_from_type_unknown_raises():
    """An unrecognized encoder_type string raises ValueError naming it."""
    with pytest.raises(ValueError, match="unknown encoder_type 'not_a_real_encoder'"):
        resolve_module_cls_from_type("not_a_real_encoder")


def test_rgb_bearing_set_membership_and_helper():
    """The shared set has the expected members and the helper agrees with it."""
    assert RGB_BEARING_ENCODER_TYPES == frozenset(
        {
            "cnn",
            "hybrid",
            "vggt_house_context",
            "vggt_house_full_tokens_nogate",
            "vggt_house_global_tokens_nogate",
            "vggt_house_global_embedding",
        }
    )
    assert encoder_type_has_rgb_target("cnn")
    assert not encoder_type_has_rgb_target("vggt")


def test_all_call_sites_agree_with_shared_rgb_set():
    """module_factory / agent_modules / decoder_targets all use the shared set.

    Locks F1: the three former hand-synced copies are now imports of
    ``RGB_BEARING_ENCODER_TYPES``, so each module must expose the identical
    object.
    """
    from src.r2dreamer import agent_modules, decoder_targets
    from src.r2dreamer.observation_preparation import module_factory

    assert module_factory.RGB_BEARING_ENCODER_TYPES is RGB_BEARING_ENCODER_TYPES
    assert agent_modules.RGB_BEARING_ENCODER_TYPES is RGB_BEARING_ENCODER_TYPES
    # decoder_targets must NOT keep its own private copy of the set — it imports
    # the shared one lazily inside decoder_rgb_target. Guard against a duplicate
    # module-level literal being reintroduced.
    assert not hasattr(decoder_targets, "_DECODER_RGB_ENCODERS")
    assert not any(
        isinstance(getattr(decoder_targets, name, None), frozenset)
        and getattr(decoder_targets, name) == RGB_BEARING_ENCODER_TYPES
        and getattr(decoder_targets, name) is not RGB_BEARING_ENCODER_TYPES
        for name in dir(decoder_targets)
    )
