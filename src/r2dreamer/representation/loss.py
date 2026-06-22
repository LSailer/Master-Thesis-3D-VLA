"""Representation loss = Barlow Twins + repval, composed for the agent."""

from typing import cast

from src.r2dreamer.learning_types import LossResult, WorldModelForward

from .barlow import barlow_loss
from .repvalue import repval_loss


def representation_loss(
    *, forward, batch, params, modules, cfg, twohot, slow_critic_params, imag_ret, B, T
) -> LossResult:
    """Combined Barlow + repval. Both consume the shared forward dict.

    Returns:
        losses: {"barlow", "repval"}.
        metrics: {} (extend here for representation diagnostics).
    """
    losses = {}
    forward = cast(WorldModelForward, forward)
    losses["barlow"] = barlow_loss(
        feat=forward.feat,
        embed=forward.embed,
        params=params,
        modules=modules,
        cfg=cfg,
        B=B,
        T=T,
    )
    losses["repval"] = repval_loss(
        feat=forward.feat,
        batch=batch,
        params=params,
        modules=modules,
        cfg=cfg,
        twohot=twohot,
        slow_critic_params=slow_critic_params,
        imag_ret=imag_ret,
        B=B,
        T=T,
    )
    return LossResult(losses=losses, metrics={})
