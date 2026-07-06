"""Tests for random-agent observation-action plumbing."""

from typing import cast

import numpy as np

from src.baselines.random_agent import RandomAgent
from src.environments.habitat import HabitatObjectNavEnv
from src.environments.observation import ObservationFrame


class _DummyEnv:
    """Environment stub that records the action used for each step."""

    def __init__(self) -> None:
        self.actions: list[int] = []

    def reset(self) -> ObservationFrame:
        """Return a reset observation without a previous action."""
        return ObservationFrame(
            image=np.zeros((3, 4, 4), dtype=np.uint8),
            is_first=True,
        )

    def step(self, action: int) -> ObservationFrame:
        """Return an observation carrying the action that produced it."""
        self.actions.append(int(action))
        return ObservationFrame(
            image=np.zeros((3, 4, 4), dtype=np.uint8),
            is_first=False,
            previous_action=int(action),
        )


def test_random_agent_returns_observation_with_previous_action() -> None:
    """RandomAgent.act returns ObservationFrame, not an action wrapper."""
    env = _DummyEnv()
    agent = RandomAgent(cast(HabitatObjectNavEnv, env), num_actions=4, seed=0)

    obs = agent.act()

    assert isinstance(obs, ObservationFrame)
    assert obs.previous_action == env.actions[-1]


def test_reset_observation_defaults_to_no_previous_action() -> None:
    """Reset-like observations have no action predecessor by default."""
    obs = ObservationFrame(
        image=np.zeros((3, 4, 4), dtype=np.uint8),
        is_first=True,
    )

    assert obs.previous_action is None
