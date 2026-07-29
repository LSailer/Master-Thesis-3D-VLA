"""Behavior loss: imagination rollout, then actor (policy) + critic (value) losses.

Reward and continue heads are *frozen* here (their training signal lives in
`world_model.loss`); they're only evaluated to score the imagined trajectory.

Returns the imagination lambda-return `ret` so the composition root can:
  1) feed it as a bootstrap into `representation.repvalue`, and
  2) update the `ReturnEMA` outside `jax.grad`.
"""

from typing import cast

import jax
import jax.numpy as jnp

from src.r2dreamer.learning_types import BehaviorLossResult, WorldModelForward
from src.r2dreamer.value_targets import LambdaReturnInputs, lambda_return

from .imagination import imagine


def behavior_loss(
    *,
    forward,
    params,
    modules,
    cfg,
    twohot,
    slow_critic_params,
    ema_state,
    return_ema,
    rng_key,
    B,
    T,
) -> BehaviorLossResult:
    """Compute policy + value losses on an imagined rollout.

    Args:
        forward: shared world-model forward output.
        params: agent params (uses `rssm`, `actor`, `critic`, plus frozen
            reads of `reward`, `cont` for scoring).
        modules: Flax module dict.
        cfg: R2DreamerConfig.
        twohot: R2TwoHotDist for reward/critic logits.
        slow_critic_params: EMA-of-critic for the slow-target value loss.
        ema_state: ReturnEMA running state (read-only here).
        return_ema: ReturnEMA helper (provides `get_stats`).
        rng_key: PRNG key for action sampling during imagination.
        B, T: batch and time dims of the *replay* batch (rollout starts from B*T).

    Returns:
        losses: {"policy", "value"}.
        metrics: {} (imagination-side metrics could be added here).
        ret: (B*T, H-1, 1) lambda-return — needed by repval and ReturnEMA update.
    """
    cfg_horizon = cfg.imagination_horizon + 1

    # Detach starting state — imagination must not backprop into the world model.
    forward = cast(WorldModelForward, forward)
    start_stoch = jax.lax.stop_gradient(
        forward.post_stochs.reshape(B * T, cfg.stoch_classes, cfg.stoch_discrete)
    )
    start_deter = jax.lax.stop_gradient(
        forward.post_deters.reshape(B * T, cfg.deter_size)
    )

    rollout = imagine(
        params["rssm"],
        params["actor"],
        modules["rssm"],
        modules["actor"],
        start_stoch,
        start_deter,
        cfg_horizon,
        rng_key,
    )
    # Already-frozen inside imagine, but guard at the boundary too
    imag_feats = jax.lax.stop_gradient(rollout.feats)
    imag_actions = jax.lax.stop_gradient(rollout.actions)

    imag_feat_flat = imag_feats.reshape(B * T * cfg_horizon, -1)

    # Frozen reward/cont/critic heads scoring the imagined trajectory
    imag_rew_logits = modules["reward"].apply(
        jax.lax.stop_gradient(params["reward"]), imag_feat_flat
    )
    imag_reward = twohot.pred(imag_rew_logits).reshape(B * T, cfg_horizon, 1)

    imag_cont_logits = modules["cont"].apply(
        jax.lax.stop_gradient(params["cont"]), imag_feat_flat
    )
    imag_cont = jax.nn.sigmoid(imag_cont_logits).reshape(B * T, cfg_horizon, 1)

    imag_val_logits = modules["critic"].apply(
        jax.lax.stop_gradient(params["critic"]), imag_feat_flat
    )
    imag_value = twohot.pred(imag_val_logits).reshape(B * T, cfg_horizon, 1)

    imag_slow_logits = modules["critic"].apply(slow_critic_params, imag_feat_flat)
    imag_slow_value = twohot.pred(imag_slow_logits).reshape(B * T, cfg_horizon, 1)

    disc = 1.0 - 1.0 / cfg.horizon
    weight = jnp.cumprod(imag_cont * disc, axis=1)

    last = jnp.zeros_like(imag_cont)
    term = 1.0 - imag_cont
    ret = lambda_return(
        LambdaReturnInputs(
            last=last,
            term=term,
            reward=imag_reward,
            value=imag_value,
            boot=imag_value,
            disc=disc,
            lamb=cfg.lamb,
        )
    )  # (BT, H-1, 1)

    _, ret_scale = return_ema.get_stats(ema_state)
    adv = (ret - imag_value[:, :-1]) / ret_scale

    # ---- Actor loss: re-evaluate unfrozen actor on detached imagined feats ----
    actor_logits = (
        modules["actor"]
        .apply(params["actor"], imag_feat_flat)
        .reshape(B * T, cfg_horizon, cfg.num_actions)
    )

    # Unimix mixing — matches PyTorch OneHotDist
    probs = jax.nn.softmax(actor_logits, axis=-1)
    uniform = jnp.ones_like(probs) / cfg.num_actions
    probs = (1.0 - cfg.unimix_ratio) * probs + cfg.unimix_ratio * uniform
    log_probs = jnp.log(probs + 1e-8)

    logpi = jnp.sum(log_probs[:, :-1] * imag_actions[:, :-1], axis=-1, keepdims=True)
    entropy = -jnp.sum(probs[:, :-1] * log_probs[:, :-1], axis=-1, keepdims=True)

    losses = {}
    losses["policy"] = jnp.mean(
        jax.lax.stop_gradient(weight[:, :-1])
        * -(logpi * jax.lax.stop_gradient(adv) + cfg.act_entropy * entropy)
    )

    # ---- Critic loss on imagined features ----
    cri_logits_imag = (
        modules["critic"]
        .apply(params["critic"], imag_feat_flat)
        .reshape(B * T, cfg_horizon, cfg.twohot_bins)
    )

    tar_padded = jnp.concatenate(
        [ret, jnp.zeros_like(ret[:, -1:])], axis=1
    )  # (BT, H, 1)

    critic_loss_tar = twohot.loss(cri_logits_imag[:, :-1], tar_padded[:, :-1, 0])
    critic_loss_slow = twohot.loss(
        cri_logits_imag[:, :-1],
        jax.lax.stop_gradient(imag_slow_value[:, :-1, 0]),
    )
    losses["value"] = jnp.mean(
        jax.lax.stop_gradient(weight[:, :-1, 0]) * (critic_loss_tar + critic_loss_slow)
    )

    return BehaviorLossResult(losses=losses, metrics={}, imag_returns=ret)
