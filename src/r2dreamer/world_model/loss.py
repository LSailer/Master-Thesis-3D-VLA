"""World-model loss: KL (dynamics + representation) + reward + continue heads.

The shared forward pass (encoder, RSSM observe, prior, get_feat) is computed
once in `agent._world_model_forward` and passed in via `WorldModelForward`.
This file owns only the loss math; it has no Flax modules of its own.
"""

import jax
import jax.numpy as jnp
import optax

from src.r2dreamer.decoder_targets import decoder_rgb_target
from src.r2dreamer.learning_types import LossResult, WorldModelForward


def kl_loss(post_logits, prior_logits, stoch_classes, stoch_discrete, kl_free):
    """DreamerV3-style KL losses with free nats.

    Args:
        post_logits: (N, C, K) posterior logits.
        prior_logits: (N, C, K) prior logits.
        stoch_classes, stoch_discrete: shape sanity (currently unused but kept
            for API stability with `tests/test_cross_framework.py`).
        kl_free: free-bits threshold (clip below).

    Returns:
        dyn_loss: (N,) — KL(sg(post) || prior), clipped to >= kl_free.
        rep_loss: (N,) — KL(post || sg(prior)), clipped to >= kl_free.
    """
    del stoch_classes, stoch_discrete  # only kept for signature back-compat
    post_log = jax.nn.log_softmax(post_logits, axis=-1)
    prior_log = jax.nn.log_softmax(prior_logits, axis=-1)
    post_probs = jnp.exp(post_log)

    def _kl(p, logp, logq):
        return jnp.sum(p * (logp - logq), axis=-1)  # (N, C)

    sg_post_probs = jax.lax.stop_gradient(post_probs)
    sg_post_log = jax.lax.stop_gradient(post_log)
    kl_dyn = jnp.sum(_kl(sg_post_probs, sg_post_log, prior_log), axis=-1)
    dyn_loss = jnp.maximum(kl_dyn, kl_free)

    sg_prior_log = jax.lax.stop_gradient(prior_log)
    kl_rep = jnp.sum(_kl(post_probs, post_log, sg_prior_log), axis=-1)
    rep_loss = jnp.maximum(kl_rep, kl_free)

    return dyn_loss, rep_loss


def world_model_loss(
    *, forward: WorldModelForward, params, batch, modules, cfg, twohot
) -> LossResult:
    """KL + reward + continue losses, plus latent diagnostics.

    Args:
        forward: shared `agent._world_model_forward` output.
        params: full agent params dict (only `reward`, `cont` are read here).
        batch: training batch (uses `rewards`, `is_episode_end`).
        modules: dict of Flax modules (uses `reward`, `cont`).
        cfg: R2DreamerConfig (uses `stoch_classes`, `stoch_discrete`, `kl_free`).
        twohot: R2TwoHotDist for the reward head.

    Returns:
        (losses, metrics) — losses has keys {dyn, rep, rew, con}; metrics
        contains latent entropy/KL diagnostics.
    """
    B, T = forward.embed.shape[0], forward.embed.shape[1]
    losses, metrics = {}, {}

    # ---- KL losses ----
    post_logits_flat = forward.post_logits.reshape(
        B * T, cfg.stoch_classes, cfg.stoch_discrete
    )
    prior_logits_flat = forward.prior_logits.reshape(
        B * T, cfg.stoch_classes, cfg.stoch_discrete
    )
    dyn_loss, rep_loss = kl_loss(
        post_logits_flat,
        prior_logits_flat,
        cfg.stoch_classes,
        cfg.stoch_discrete,
        cfg.kl_free,
    )
    losses["dyn"] = jnp.mean(dyn_loss)
    losses["rep"] = jnp.mean(rep_loss)

    # ---- Reward head ----
    feat_flat = forward.feat.reshape(B * T, -1)
    rew_logits = modules["reward"].apply(params["reward"], feat_flat).reshape(B, T, -1)
    losses["rew"] = jnp.mean(twohot.loss(rew_logits, batch["rewards"]))

    # ---- Continue head ----
    cont_logits = modules["cont"].apply(params["cont"], feat_flat).reshape(B, T, 1)
    cont_target = 1.0 - batch["is_episode_end"]
    losses["con"] = jnp.mean(
        optax.sigmoid_binary_cross_entropy(cont_logits[..., 0], cont_target)
    )

    # ---- Debug decoder probe (image reconstruction; only when cfg.decoder) ----
    # The decoder is a visualisation probe, not an agent objective: detach the
    # RSSM feature so reconstruction gradients train only decoder params. The
    # agent loss/metrics stay comparable to decoder-free runs; see agent.py for
    # the auxiliary opt loss that keeps the probe itself learning.
    if cfg.decoder:
        decoder_feat = jax.lax.stop_gradient(feat_flat)
        recon = modules["decoder"].apply(
            params["decoder"], decoder_feat
        )  # (BT,3,64,64)
        rgb_target = decoder_rgb_target(batch, cfg.encoder_type)
        losses["decoder"] = jnp.mean((recon - rgb_target) ** 2)
        metrics["decoder/recon_mse"] = losses["decoder"]

    # ---- Latent diagnostics (cheap; once per step) ----
    prior_probs = jax.nn.softmax(prior_logits_flat, axis=-1)
    post_probs = jax.nn.softmax(post_logits_flat, axis=-1)
    metrics["latent/prior_entropy"] = -jnp.mean(
        jnp.sum(prior_probs * jnp.log(prior_probs + 1e-8), axis=-1)
    )
    metrics["latent/posterior_entropy"] = -jnp.mean(
        jnp.sum(post_probs * jnp.log(post_probs + 1e-8), axis=-1)
    )

    return LossResult(losses=losses, metrics=metrics)
