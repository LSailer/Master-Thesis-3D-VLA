"""Encoder recipes: one registry entry per encoder type (design card 6a).

An :class:`EncoderRecipe` couples the two halves that must agree on an
observation schema but are otherwise independent:

- ``make_adapter`` builds the *frozen* observation adapter (VGGT extraction,
  house-point accumulation, symlog) that the ``ExperienceCollector`` owns;
- ``build_composite`` builds the *trainable* :class:`CompositeSpec` the agent
  initializes and differentiates.

There is deliberately no declared obs-spec: it is inferred from the first
prepared frame at startup (``ExperienceCollector.reset()``), and the single
remaining consistency check is
``set(composite.branches) == set(inferred keys)`` — see
:func:`check_branch_keys`.

Migration status: ``cnn`` and ``hybrid`` are ported here. Unported encoder
types keep the old ``encoders/factory.py`` path until ported one by one
(HANDOFF.md step 5); ``build_composite`` reproduces the legacy module kwargs so
the ported types stay golden-run-identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from src.configs.config import R2DreamerConfig
from src.r2dreamer.encoders.cnn import ConvEncoder, make_rgb_conv_encoder
from src.r2dreamer.encoders.composite import BranchSpec, CompositeSpec
from src.r2dreamer.encoders import HybridEncoder as LauncherHybridEncoder
from src.r2dreamer.observation_keys import HYBRID_IMAGE_KEY, HYBRID_WP_CP_KEY
from src.r2dreamer.observation_preparation import (
    CNNObservationPreparation,
    normalize_encoder_module_kwargs,
)
from src.r2dreamer.world_model.heads import R2MLP
from src.shared.dtypes import compute_jnp_dtype


def _compute_dtype_kwargs(cfg: R2DreamerConfig) -> dict:
    """Return the ``compute_dtype`` overlay for the ``full_bf16`` gate.

    Mirrors ``factory._compute_dtype_kwargs`` but is defined here so recipes do
    not import factory (which would form a factory<->recipes cycle): only
    supplies ``compute_dtype`` when the gate is on, else each module keeps its
    own default.
    """
    if getattr(cfg, "full_bf16", False):
        return {"compute_dtype": compute_jnp_dtype(cfg.compute_dtype)}
    return {}


@dataclass(frozen=True)
class EncoderRecipe:
    """Everything needed to build one encoder type's adapter and composite.

    Attributes:
        encoder_type: Canonical encoder-type name (also the registry key).
        make_adapter: Builds the frozen observation adapter. Receives the
            parsed training args (or ``None`` for tests/tools that only need
            the composite).
        build_composite: Builds the trainable :class:`CompositeSpec` from the
            effective agent config. Kept a builder (not a literal) so branch
            modules pick up config knobs (depth, MLP width, ...) — which the
            golden run requires.
        rgb_key: Observation key carrying an RGB image, or ``None``. Replaces
            the global ``*_RGB_ENCODER_TYPES`` name lists (DELETIONS.md): a
            decoder target is available iff this is set.
    """

    encoder_type: str
    make_adapter: Callable[[Any], Any]
    build_composite: Callable[[R2DreamerConfig], CompositeSpec]
    rgb_key: str | None = None


# ---------------------------------------------------------------------------
# Composite builders (mirror the legacy factory / base.py kwargs exactly)
# ---------------------------------------------------------------------------


def _contract_module_kwargs(cfg: R2DreamerConfig) -> dict:
    """Return the durable Encoder Input Contract's module kwargs, if any.

    Mirrors ``factory._contract_encoder_kwargs``: a checkpoint's contract
    snapshot pins the exact kwargs the encoder was built with, and those win
    over the current config so an old checkpoint rebuilds the same module even
    when the config has drifted.
    """
    snapshot = getattr(cfg, "encoder_input_contract", None)
    if snapshot is None:
        return {}
    return normalize_encoder_module_kwargs(snapshot.get("encoder_module_kwargs", {}))


def build_cnn_composite(cfg: R2DreamerConfig) -> CompositeSpec:
    """Single ConvEncoder branch — byte-identical to the legacy ``cnn`` module.

    Honors the contract-pinned kwargs when present (checkpoint portability),
    else derives depth/kernel/mults from config plus the full_bf16 dtype overlay
    — exactly matching ``factory._make_conv_encoder``.
    """
    contract_kwargs = _contract_module_kwargs(cfg)
    conv_kwargs = contract_kwargs or {
        "depth": int(cfg.encoder_depth),
        "kernel_size": int(cfg.encoder_kernel),
        "mults": tuple(cfg.encoder_mults),
        **_compute_dtype_kwargs(cfg),
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
    branch here does too.
    """
    # Contract kwargs use the WMHybridEncoder parameter names; map them onto the
    # composite's CNN + MLP branches. Absent a contract, derive from config.
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


# ---------------------------------------------------------------------------
# Adapter builders
# ---------------------------------------------------------------------------
#
# During migration these delegate to the launcher encoder selections so the
# frozen extraction path is byte-for-byte the current one (golden-safe). The
# selection classes are deleted only in the later migration steps (DELETIONS.md),
# at which point this logic moves inline here.


def _cnn_adapter(args: Any = None) -> Any:
    del args  # CNN preparation needs no runtime args.
    return CNNObservationPreparation()


def _hybrid_adapter(args: Any) -> Any:
    return LauncherHybridEncoder.from_train_args(args).make_adapter()


RECIPES: dict[str, EncoderRecipe] = {
    "cnn": EncoderRecipe(
        encoder_type="cnn",
        make_adapter=_cnn_adapter,
        build_composite=build_cnn_composite,
        rgb_key=HYBRID_IMAGE_KEY,
    ),
    "hybrid": EncoderRecipe(
        encoder_type="hybrid",
        make_adapter=_hybrid_adapter,
        build_composite=build_hybrid_composite,
        rgb_key=HYBRID_IMAGE_KEY,
    ),
}


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
