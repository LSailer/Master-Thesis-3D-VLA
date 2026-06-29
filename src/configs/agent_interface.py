"""R2Dreamer environment/agent interface configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class R2DreamerInterfaceConfig:
    """Static interface facts required to build an ``R2DreamerAgent``.

    Args:
        obs_shape: Encoder Module input shape without the live batch axis or
            replay ``(B, T)`` prefix. Lazy initialization may infer this from the
            first prepared encoder observation and then store it here.
        num_actions: Number of discrete actions exposed by the environment.
            This is not inferable from observations and determines RSSM action
            input width and actor logits.

    ``max_episode_steps`` is intentionally absent: episode caps and reward
    shaping belong to environment/trainer configuration, not to the agent's
    neural architecture/interface.
    """

    obs_shape: tuple[int, ...] | Mapping[str, tuple[int, ...]]
    num_actions: int
