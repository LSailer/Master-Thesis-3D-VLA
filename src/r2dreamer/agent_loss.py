"""Total-objective composition for R2DreamerAgent.

Combines the three sub-package losses (world-model, behavior, representation)
into one scalar objective plus a flat metrics dict. Extracted from
``agent.py``'s ``_loss_fn`` so the agent's training orchestration stays
readable; all encoder-identity-specific diagnostics (previously an
``if cfg.encoder_type in (...)`` branch here) are now resolved through the
encoder registry (``encoders/registry.py``) so adding a new encoder's
diagnostics never requires editing this module.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.configs.config import R2DreamerConfig
from src.r2dreamer.behavior.loss import behavior_loss
from src.r2dreamer.decoder_targets import replay_batch_shape
from src.r2dreamer.encoders.registry import encoder_loss_diagnostics
from src.r2dreamer.learning_types import AgentLossAux
from src.r2dreamer.representation.loss import representation_loss
from src.r2dreamer.world_model.loss import world_model_loss


def weighted_total_loss(cfg: R2DreamerConfig, losses: dict) -> jnp.ndarray:
    """Agent objective, excluding the optional debug decoder probe.

    Args:
      cfg: Effective agent config supplying the per-term ``scale_*`` weights.
      losses: Per-term loss dict with keys ``dyn``, ``rep``, ``barlow``,
        ``rew``, ``con``, ``policy``, ``value``, ``repval``.

    Returns:
      The scalar weighted sum of all terms.
    """
    return (
        cfg.scale_dyn * losses["dyn"]
        + cfg.scale_rep * losses["rep"]
        + cfg.scale_barlow * losses["barlow"]
        + cfg.scale_rew * losses["rew"]
        + cfg.scale_con * losses["con"]
        + cfg.scale_policy * losses["policy"]
        + cfg.scale_value * losses["value"]
        + cfg.scale_repval * losses["repval"]
    )


def _add_loss_metrics(metrics: dict, losses: dict) -> None:
    for k, v in losses.items():
        metrics[f"loss/{k}"] = v


def _add_encoder_l2_metric(metrics: dict, params: dict) -> None:
    # Encoder L2 — Protocol D diagnostic for whether Barlow's gradient
    # toggle is actually moving the encoder weights.
    enc_sq = jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(jnp.square(x)),
        params["encoder"],
        0.0,
    )
    metrics["params/encoder_l2"] = jnp.sqrt(enc_sq)


def compose_agent_loss(
    *,
    cfg: R2DreamerConfig,
    modules: dict,
    twohot,
    return_ema,
    world_model_forward,
    params,
    slow_critic_params,
    ema_state,
    batch,
    rng_key,
) -> tuple[jnp.ndarray, AgentLossAux]:
    """Compose the world-model, behavior, and representation losses.

    Args:
      cfg: Effective agent config.
      modules: Name-keyed Flax module bundle (``AgentModules.modules``).
      twohot: The shared ``R2TwoHotDist`` used by reward/critic heads.
      return_ema: The agent's ``ReturnEMA`` tracker (stats only; the actual
        state update happens outside ``jax.grad`` in the train step).
      world_model_forward: Callable computing the shared encoder/RSSM
        forward pass, i.e. ``agent._world_model_forward``.
      params: Current parameter pytree.
      slow_critic_params: EMA'd critic params for bootstrap targets.
      ema_state: Current ``ReturnEMA`` state.
      batch: Replay batch.
      rng_key: PRNG key for this loss evaluation.

    Returns:
      ``(total_loss, aux)`` — ``aux`` carries metrics and the imagination
      returns used for the post-step ``ReturnEMA`` update.
    """
    B, T = replay_batch_shape(batch)

    rng_key, k_fwd = jax.random.split(rng_key)
    forward = world_model_forward(params, batch, k_fwd)

    wm_result = world_model_loss(
        forward=forward,
        params=params,
        batch=batch,
        modules=modules,
        cfg=cfg,
        twohot=twohot,
    )

    rng_key, k_behavior = jax.random.split(rng_key)
    behavior_result = behavior_loss(
        forward=forward,
        params=params,
        modules=modules,
        cfg=cfg,
        twohot=twohot,
        slow_critic_params=slow_critic_params,
        ema_state=ema_state,
        return_ema=return_ema,
        rng_key=k_behavior,
        B=B,
        T=T,
    )

    rep_result = representation_loss(
        forward=forward,
        batch=batch,
        params=params,
        modules=modules,
        cfg=cfg,
        twohot=twohot,
        slow_critic_params=slow_critic_params,
        imag_ret=behavior_result.imag_returns,
        B=B,
        T=T,
    )

    losses = {
        **wm_result.losses,
        **behavior_result.losses,
        **rep_result.losses,
    }
    agent_loss = weighted_total_loss(cfg, losses)
    # The decoder is a stop-gradient visualisation probe. Add its detached
    # reconstruction loss only to the optimiser objective so the decoder
    # learns to read the current latent, while the agent/RSSM/encoder see
    # exactly the same objective as decoder-free runs.
    total_loss = agent_loss
    if cfg.decoder:
        total_loss = total_loss + cfg.scale_decoder * losses["decoder"]

    # ---- Metrics ----
    metrics = {
        **wm_result.metrics,
        **behavior_result.metrics,
        **rep_result.metrics,
    }
    _add_loss_metrics(metrics, losses)
    _add_encoder_l2_metric(metrics, params)

    # ---- Per-encoder loss diagnostics (registry-resolved; no-op unless the
    # active encoder module registers a hook, e.g. hybrid gate contribution) ----
    encoder_loss_diagnostics(
        type(modules["encoder"]),
        metrics,
        cfg=cfg,
        params=params,
        forward=forward,
        B=B,
        T=T,
    )

    aux = AgentLossAux(
        metrics=metrics,
        imag_returns=behavior_result.imag_returns.reshape(-1),
        agent_loss=agent_loss,
    )
    return total_loss, aux
