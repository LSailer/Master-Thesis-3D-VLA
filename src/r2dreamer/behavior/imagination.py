"""Imagination rollout in latent space + lambda-return computation.

`_imagine` runs the actor inside the (frozen) RSSM for H steps, producing
detached imagined features and one-hot actions. `_lambda_return` computes the
GAE-style target used by both the imagination critic loss and the replay
value-learning bootstrap (`representation.repvalue`).
"""

import jax
import jax.numpy as jnp

from src.r2dreamer.learning_types import ImaginationRollout
from src.r2dreamer.value_targets import LambdaReturnInputs, lambda_return


def _lambda_return(*args):
    """Compatibility wrapper for the historical positional helper."""
    return lambda_return(LambdaReturnInputs(*args))


def _imagine(
    rssm_params,
    actor_params,
    rssm_mod,
    actor_mod,
    start_stoch,
    start_deter,
    horizon,
    rng_key,
):
    """Imagination rollout in latent space (no gradients).

    Matches PyTorch: imagination is fully detached (@torch.no_grad).
    The policy loss is computed separately by re-evaluating the actor
    on the detached imagined features.

    Args:
        rssm_params: RSSM parameters (frozen).
        actor_params: Actor parameters (frozen for imagination).
        rssm_mod: R2RSSM module.
        actor_mod: MLP module for actor.
        start_stoch: (N, C, K) starting stochastic state.
        start_deter: (N, D) starting deterministic state.
        horizon: number of steps to imagine.
        rng_key: PRNG key.

    Returns:
        `ImaginationRollout` with detached `feats` and clean one-hot `actions`.
    """
    frozen_rssm_params = jax.lax.stop_gradient(rssm_params)
    frozen_actor_params = jax.lax.stop_gradient(actor_params)

    stoch = start_stoch
    deter = start_deter
    feats = []
    actions = []

    for _ in range(horizon):
        feat = rssm_mod.apply(
            frozen_rssm_params, stoch, deter, method=rssm_mod.get_feat
        )

        # Frozen actor — no gradients during imagination
        logits = actor_mod.apply(frozen_actor_params, feat)
        rng_key, k = jax.random.split(rng_key)
        action = jax.nn.one_hot(
            jax.random.categorical(k, logits, axis=-1),
            logits.shape[-1],
        )

        feats.append(feat)
        actions.append(action)

        rng_key, k_img = jax.random.split(rng_key)
        stoch, deter = rssm_mod.apply(
            frozen_rssm_params,
            stoch,
            deter,
            action,
            method=rssm_mod.img_step,
            rngs={"sample": k_img},
        )

    return ImaginationRollout(
        feats=jnp.stack(feats, axis=1),
        actions=jnp.stack(actions, axis=1),
    )
