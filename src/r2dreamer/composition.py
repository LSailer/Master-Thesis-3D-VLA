"""Learner composition from config: recipe lookup + encoder injection.

The launcher (``launch/train.py``) composes the learner from a live
collector's first prepared frame. Everything else — evaluation, checkpoint
loading, tests, profiling scripts — composes it from config alone via
:func:`make_learner`, using the recipe's init dummy in place of a real frame.
This module replaces the deleted ``R2DreamerAgent`` shim (DELETIONS.md): the
learner itself never constructs or validates encoders.
"""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from src.configs.config import R2DreamerConfig
from src.r2dreamer.encoders.composite import CompositeEncoder
from src.r2dreamer.encoders.recipes import (
    RECIPES,
    build_encoder_module,
    check_branch_keys,
    dummy_encoder_obs,
    infer_obs_spec,
)

from .learner import R2DLearner, load_policy_checkpoint
from .observation_preparation.contracts import recover_encoder_input_contract


def validate_decoder_support(config: R2DreamerConfig) -> None:
    """Fail loud when ``decoder=True`` is configured for a non-RGB encoder.

    Args:
        config: Effective agent config.

    Raises:
        ValueError: If the selected recipe carries no RGB modality
            (``rgb_key is None``) — the ConvDecoder would have no target.
    """
    if not config.decoder:
        return
    recipe = RECIPES.get(config.encoder_type)
    if recipe is not None and recipe.rgb_key is None:
        raise ValueError(
            "decoder=True requires an RGB-bearing encoder_type — the "
            "ConvDecoder reconstructs an RGB image, but "
            f"{config.encoder_type!r} carries no RGB modality to reconstruct."
        )


def make_learner(config: R2DreamerConfig, rng_key: jnp.ndarray) -> R2DLearner:
    """Build a learner from config alone (recipe module + init dummy).

    Args:
        config: Effective agent config.
        rng_key: Master init PRNG key (split internally by the learner).

    Returns:
        The composed :class:`R2DLearner`.

    Raises:
        ValueError: For an unknown encoder type, invalid encoder config, a
            composite branch/obs key mismatch, or an unsupported decoder ask.
    """
    validate_decoder_support(config)
    encoder = build_encoder_module(config)
    init_obs = dummy_encoder_obs(config)
    # Decision 5: the one startup check for composite (recipe) encoders —
    # the spec's branch keys must equal the keys of the prepared obs (here
    # the init dummy, which shares the prepared-frame schema).
    if isinstance(encoder, CompositeEncoder):
        check_branch_keys(encoder.spec, infer_obs_spec(init_obs).keys())
    return R2DLearner(config, rng_key, encoder=encoder, encoder_init_obs=init_obs)


def learner_from_checkpoint(
    path: str | Path,
    *,
    obs_shape: tuple[int, ...] | dict[str, tuple[int, ...]] | None = None,
    num_actions: int,
    seed: int,
    **config_kwargs: Any,
) -> R2DLearner:
    """Build a learner and load ``params`` + ``slow_critic_params`` from disk.

    Extra ``config_kwargs`` flow into :class:`R2DreamerConfig` so callers
    that need ``encoder_type`` / ``encoder_module_cls`` (e.g. evaluate)
    can pass them through. When the checkpoint contains a durable Encoder
    Input Contract snapshot, missing encoder config is recovered from it.
    The loaded checkpoint's ``step`` is stashed on the returned learner as
    ``checkpoint_step`` (``-1`` if absent).

    Args:
        path: Checkpoint path written by ``save_checkpoint``.
        obs_shape: Observation shape; optional when the checkpoint carries a
            contract snapshot to recover it from.
        num_actions: Discrete action-space size.
        seed: Init seed (params are overwritten from the checkpoint).
        **config_kwargs: Extra :class:`R2DreamerConfig` fields.

    Returns:
        The composed learner with checkpoint params applied.

    Raises:
        ValueError: On encoder type/shape mismatches against the checkpoint
            contract, or when ``obs_shape`` is missing with no snapshot.
        KeyError: When the checkpoint lacks required state keys.
    """
    ckpt = load_policy_checkpoint(path)
    contract_snapshot = ckpt.get("encoder_input_contract")
    if contract_snapshot is not None:
        contract = recover_encoder_input_contract(contract_snapshot)
        if obs_shape is None:
            obs_shape = contract.encoder_input.buffer_shape()
        requested_type = config_kwargs.get("encoder_type")
        if requested_type is not None and requested_type != contract.encoder_type:
            raise ValueError(
                "checkpoint encoder contract mismatch: requested "
                f"{requested_type!r}, checkpoint has {contract.encoder_type!r}"
            )
        requested_shape = obs_shape
        contract_shape = contract.encoder_input.buffer_shape()
        if requested_shape != contract_shape:
            raise ValueError(
                "checkpoint encoder shape mismatch: requested "
                f"{requested_shape!r}, checkpoint has {contract_shape!r}"
            )
        config_kwargs["encoder_type"] = contract.encoder_type
        config_kwargs["encoder_module_cls"] = contract.encoder_module_cls
        config_kwargs["encoder_input_contract"] = contract_snapshot
    if obs_shape is None:
        raise ValueError(
            "obs_shape must be provided when checkpoint has no Encoder Input "
            "Contract snapshot"
        )
    config = R2DreamerConfig(
        obs_shape=obs_shape,
        num_actions=num_actions,
        **config_kwargs,
    )
    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)
    learner = make_learner(config, init_key)
    learner.params = jax.tree.map(jnp.asarray, ckpt["params"])
    learner.slow_critic_params = jax.tree.map(
        jnp.asarray, ckpt["slow_critic_params"]
    )
    learner.checkpoint_step = int(ckpt.get("step", -1))
    return learner
