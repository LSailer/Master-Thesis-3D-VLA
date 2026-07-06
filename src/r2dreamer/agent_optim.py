"""Optimizer construction for R2DreamerAgent.

Isolated from ``agent.py`` so the agent's ``__init__`` doesn't hand-roll the
LaProp + warmup wiring inline. Adaptive Gradient Clipping (``agc``) is applied
per train-step (it needs live gradients/params), so it stays a plain function
call in the train step rather than part of this construction.
"""

from __future__ import annotations

import optax

from src.configs.config import R2DreamerConfig
from src.shared.optim import laprop


def make_optimizer(config: R2DreamerConfig) -> optax.GradientTransformation:
    """Build the agent's LaProp optimizer with linear warmup.

    Args:
      config: Effective agent config supplying ``lr``/``beta1``/``beta2``/
        ``eps``/``warmup_steps``.

    Returns:
      An Optax ``GradientTransformation`` implementing LaProp.
    """
    return laprop(
        lr=config.lr,
        b1=config.beta1,
        b2=config.beta2,
        eps=config.eps,
        warmup=config.warmup_steps,
    )
