"""Typed pytrees shared by R2Dreamer learning code."""

from __future__ import annotations

from typing import Any, NamedTuple


class WorldModelForward(NamedTuple):
    """Shared encoder/RSSM forward pass consumed by all loss components."""

    embed: Any
    post_stochs: Any
    post_deters: Any
    post_logits: Any
    prior_logits: Any
    feat: Any


class LossResult(NamedTuple):
    """Loss component output before weighting and metric logging."""

    losses: dict[str, Any]
    metrics: dict[str, Any]


class BehaviorLossResult(NamedTuple):
    """Behavior loss output plus imagined returns needed by repval/EMA."""

    losses: dict[str, Any]
    metrics: dict[str, Any]
    imag_returns: Any


class AgentLossAux(NamedTuple):
    """Auxiliary values returned from the total agent objective."""

    metrics: dict[str, Any]
    imag_returns: Any
    agent_loss: Any


class ImaginationRollout(NamedTuple):
    """Detached latent imagination rollout."""

    feats: Any
    actions: Any
