"""World-model loss: KL (dynamics + representation) + reward + continue heads.

The shared forward pass (encoder, RSSM observe, prior, get_feat) is computed
once in `agent._world_model_forward` and passed in via the `forward` dict.
This file owns only the loss math; it has no Flax modules of its own.
"""

import jax
import jax.numpy as jnp
import optax


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
    post_probs = jax.nn.softmax(post_logits, axis=-1)
    prior_probs = jax.nn.softmax(prior_logits, axis=-1)

    post_log = jnp.log(post_probs + 1e-8)
    prior_log = jnp.log(prior_probs + 1e-8)

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


def world_model_loss(*, forward, params, batch, modules, cfg, twohot):
    """KL + reward + continue losses, plus latent diagnostics.

    Args:
        forward: dict from `agent._world_model_forward`. Must contain
            `embed` (B, T, E), `feat` (B, T, F), `post_logits` (B, T, C, K),
            `prior_logits` (B, T, C, K).
        params: full agent params dict (only `reward`, `cont` are read here).
        batch: training batch (uses `rewards`, `is_terminal`).
        modules: dict of Flax modules (uses `reward`, `cont`).
        cfg: R2DreamerConfig (uses `stoch_classes`, `stoch_discrete`, `kl_free`).
        twohot: R2TwoHotDist for the reward head.

    Returns:
        (losses, metrics) — losses has keys {dyn, rep, rew, con}; metrics
        contains latent entropy/KL diagnostics.
    """
    B, T = batch["obs"].shape[0], batch["obs"].shape[1]
    losses, metrics = {}, {}

    # ---- KL losses ----
    post_logits_flat = forward["post_logits"].reshape(
        B * T, cfg.stoch_classes, cfg.stoch_discrete)
    prior_logits_flat = forward["prior_logits"].reshape(
        B * T, cfg.stoch_classes, cfg.stoch_discrete)
    dyn_loss, rep_loss = kl_loss(
        post_logits_flat, prior_logits_flat,
        cfg.stoch_classes, cfg.stoch_discrete, cfg.kl_free,
    )
    losses["dyn"] = jnp.mean(dyn_loss)
    losses["rep"] = jnp.mean(rep_loss)

    # ---- Reward head ----
    feat_flat = forward["feat"].reshape(B * T, -1)
    rew_logits = modules["reward"].apply(params["reward"], feat_flat).reshape(B, T, -1)
    losses["rew"] = jnp.mean(twohot.loss(rew_logits, batch["rewards"]))

    # ---- Continue head ----
    cont_logits = modules["cont"].apply(params["cont"], feat_flat).reshape(B, T, 1)
    cont_target = 1.0 - batch["is_terminal"]
    losses["con"] = jnp.mean(
        optax.sigmoid_binary_cross_entropy(cont_logits[..., 0], cont_target)
    )

    # ---- Co-trained decoder (image reconstruction; only when cfg.decoder) ----
    # Off by default → no `decoder` loss key, so the total-loss sum and metrics
    # are identical to the decoder-free baseline. 3D-51 visual-verification head.
    if cfg.decoder:
        recon = modules["decoder"].apply(params["decoder"], feat_flat)  # (BT,3,64,64)
        if cfg.encoder_type.startswith("hybrid"):
            # Hybrid obs is [ rgb (rgb_dim) | wp_cp (vggt_feature_dim) ]; the RGB
            # slice is already normalised to [0, 1] by the adapter.
            rgb_dim = cfg.obs_shape[0] - cfg.vggt_feature_dim  # 16404 - 4116 = 12288
            rgb_target = batch["obs"].reshape(B * T, -1)[:, :rgb_dim].reshape(B * T, 3, 64, 64)
        else:
            # CNN path: obs is already (B, T, 3, 64, 64), normalised to [0, 1].
            rgb_target = batch["obs"].reshape(B * T, 3, 64, 64)
        losses["decoder"] = jnp.mean((recon - rgb_target) ** 2)
        metrics["decoder/recon_mse"] = losses["decoder"]

    # ---- Latent diagnostics (cheap; once per step) ----
    prior_probs = jax.nn.softmax(prior_logits_flat, axis=-1)
    post_probs = jax.nn.softmax(post_logits_flat, axis=-1)
    metrics["latent/prior_entropy"] = -jnp.mean(
        jnp.sum(prior_probs * jnp.log(prior_probs + 1e-8), axis=-1))
    metrics["latent/posterior_entropy"] = -jnp.mean(
        jnp.sum(post_probs * jnp.log(post_probs + 1e-8), axis=-1))

    return losses, metrics
