"""Replay-based value learning (repval).

Computes a critic loss on *replay* features (not imagined ones), using the
imagination return at step 0 as the bootstrap value. Crucially, the gradient
flows through `feat` back into the world model — that's the whole point of
repval as a representation-shaping signal.
"""

import jax
import jax.numpy as jnp

from src.r2dreamer.value_targets import LambdaReturnInputs, lambda_return


def repval_loss(
    *, feat, batch, params, modules, cfg, twohot, slow_critic_params, imag_ret, B, T
):
    """Replay value loss with world-model gradients.

    Args:
        feat: (B, T, F) — RSSM features WITH gradients to encoder/RSSM.
        batch: training batch (uses `is_episode_end`, `rewards`).
        params: agent params (uses `critic`).
        modules: Flax module dict (uses `critic`).
        cfg: R2DreamerConfig (uses `horizon`, `lamb`, `twohot_bins`).
        twohot: R2TwoHotDist for the critic.
        slow_critic_params: EMA-of-critic for the slow-target term.
        imag_ret: (B*T, H-1, 1) imagination lambda-return; `imag_ret[:, 0]`
            is the bootstrap value at each replay step.
        B, T: batch and time dims.

    Returns:
        scalar repval loss.
    """
    replay_episode_end = batch.is_episode_end  # (B, T)
    replay_reward = batch.rewards  # (B, T)

    # Bootstrap from the imagined return at step 0
    boot = imag_ret[:, 0].reshape(B, T, 1)

    feat_flat = feat.reshape(B * T, -1)

    # Frozen critic for the lambda-return target
    replay_val_logits = (
        modules["critic"]
        .apply(jax.lax.stop_gradient(params["critic"]), feat_flat)
        .reshape(B, T, cfg.twohot_bins)
    )
    replay_value = twohot.pred(replay_val_logits)  # (B, T, 1)

    replay_slow_logits = (
        modules["critic"]
        .apply(slow_critic_params, feat_flat)
        .reshape(B, T, cfg.twohot_bins)
    )
    replay_slow_value = twohot.pred(replay_slow_logits)  # (B, T, 1)

    disc = 1.0 - 1.0 / cfg.horizon
    replay_ret = lambda_return(
        LambdaReturnInputs(
            last=replay_episode_end[..., None],
            term=replay_episode_end[..., None],
            reward=replay_reward[..., None],
            value=replay_value,
            boot=boot,
            disc=disc,
            lamb=cfg.lamb,
        )
    )  # (B, T-1, 1)
    ret_padded = jnp.concatenate(
        [replay_ret, jnp.zeros_like(replay_ret[:, -1:])], axis=1
    )

    # Critic on replay features WITH gradients to the world model
    repval_logits = (
        modules["critic"]
        .apply(params["critic"], feat_flat)
        .reshape(B, T, cfg.twohot_bins)
    )

    repval_weight = 1.0 - replay_episode_end  # (B, T)
    loss_tar = twohot.loss(
        repval_logits[:, :-1],
        jax.lax.stop_gradient(ret_padded[:, :-1, 0]),
    )
    loss_slow = twohot.loss(
        repval_logits[:, :-1],
        jax.lax.stop_gradient(replay_slow_value[:, :-1, 0]),
    )
    return jnp.mean(repval_weight[:, :-1] * (loss_tar + loss_slow))
