"""Temporary compatibility shim: ``R2DreamerAgent`` = recipe + ``R2DLearner``.

The learner internals moved to :mod:`src.r2dreamer.learner`; encoder
construction moved to the ``EncoderRecipe`` registry
(:mod:`src.r2dreamer.encoders.recipes`). This shim keeps the historical
one-argument construction path — ``R2DreamerAgent(cfg, rng)`` builds the
encoder from the registry and injects it — for tests, evaluation, and
checkpoint loading. It is deleted in the final migration step (DELETIONS.md)
once callers construct the learner via the composition root.
"""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from src.configs.config import R2DreamerConfig
from src.r2dreamer.encoders.composite import CompositeEncoder
from src.r2dreamer.encoders.recipes import (
    build_encoder_module,
    check_branch_keys,
    dummy_encoder_obs,
    infer_obs_spec,
)

from .learner import (
    ActState,
    R2DLearner,
    R2DTrainState,
    load_policy_checkpoint,
)
from .observation_preparation.contracts import recover_encoder_input_contract

# Re-exported legacy surface: callers historically imported these from agent.py.
__all__ = [
    "ActState",
    "R2DLearner",
    "R2DTrainState",
    "R2DreamerAgent",
    "load_policy_checkpoint",
]


class R2DreamerAgent(R2DLearner):
    """Legacy construction path: build the encoder from config, then learn.

    The learner proper receives the encoder injected (IDEA.md decision 1);
    this subclass only performs the recipe lookup + init-dummy construction
    that the launcher composition root does with a real prepared frame.
    """

    def __init__(self, config: R2DreamerConfig, rng_key: jnp.ndarray):
        encoder = build_encoder_module(config)
        init_obs = dummy_encoder_obs(config)
        # Decision 5: the one startup check for composite (recipe) encoders —
        # the spec's branch keys must equal the keys of the prepared obs
        # (here the init dummy, which shares the prepared-frame schema).
        if isinstance(encoder, CompositeEncoder):
            check_branch_keys(encoder.spec, infer_obs_spec(init_obs).keys())
        super().__init__(
            config, rng_key, encoder=encoder, encoder_init_obs=init_obs
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        obs_shape: tuple[int, ...] | dict[str, tuple[int, ...]] | None = None,
        num_actions: int,
        seed: int,
        **config_kwargs: Any,
    ) -> "R2DreamerAgent":
        """Build an agent and load ``params`` + ``slow_critic_params`` from disk.

        Extra ``config_kwargs`` flow into :class:`R2DreamerConfig` so callers
        that need ``encoder_type`` / ``encoder_module_cls`` (e.g. evaluate)
        can pass them through. When the checkpoint contains a durable Encoder
        Input Contract snapshot, missing encoder config is recovered from it.
        The loaded checkpoint's ``step`` is stashed on the returned agent as
        ``checkpoint_step`` (``-1`` if absent).
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
        agent = cls(config, init_key)
        agent.params = jax.tree.map(jnp.asarray, ckpt["params"])
        agent.slow_critic_params = jax.tree.map(jnp.asarray, ckpt["slow_critic_params"])
        agent.checkpoint_step = int(ckpt.get("step", -1))
        return agent
