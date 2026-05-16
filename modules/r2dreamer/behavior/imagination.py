"""Imagination rollout in latent space + lambda-return computation.

`_imagine` runs the actor inside the (frozen) RSSM for H steps, producing
detached imagined features and one-hot actions. `_lambda_return` computes the
GAE-style target used by both the imagination critic loss and the replay
value-learning bootstrap (`representation.repvalue`).
"""

import jax
import jax.numpy as jnp


def _imagine(rssm_params, actor_params, rssm_mod, actor_mod,
             start_stoch, start_deter, horizon, rng_key):
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
        feats: (N, horizon, feat_size) — detached.
        actions: (N, horizon, num_actions) — clean one-hot, detached.
        None: legacy slot kept for tuple-arity back-compat.
    """
    frozen_rssm_params = jax.lax.stop_gradient(rssm_params)
    frozen_actor_params = jax.lax.stop_gradient(actor_params)

    stoch = start_stoch
    deter = start_deter
    feats = []
    actions = []

    for step in range(horizon):
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
            frozen_rssm_params, stoch, deter, action, method=rssm_mod.img_step,
            rngs={"sample": k_img},
        )

    return jnp.stack(feats, axis=1), jnp.stack(actions, axis=1), None


def _lambda_return(last, term, reward, value, boot, disc, lamb):
    """Compute lambda-returns (generalized advantage estimation target).

    All inputs: (..., T, 1).
    Returns: (..., T-1, 1).
    """
    live = (1.0 - term)[..., 1:, :] * disc
    cont = (1.0 - last)[..., 1:, :] * lamb
    interm = reward[..., 1:, :] + (1.0 - cont) * live * boot[..., 1:, :]
    T_minus_1 = live.shape[-2]

    def _scan_fn(carry, i):
        # i counts from 0 to T_minus_1 - 1, but we want reversed order
        idx = T_minus_1 - 1 - i
        val = interm[..., idx, :] + live[..., idx, :] * cont[..., idx, :] * carry
        return val, val

    init = boot[..., -1, :]
    _, outs = jax.lax.scan(
        _scan_fn, init, jnp.arange(T_minus_1)
    )
    # outs: (T_minus_1, ..., 1) — need to reverse and transpose
    outs = jnp.flip(outs, axis=0)
    ndim = outs.ndim
    axes = list(range(1, ndim - 1)) + [0, ndim - 1]
    outs = jnp.transpose(outs, axes)
    return outs
