"""End-to-end tests for ``HabitatObjectNavEnv`` (habitat-sim + HM3D data required).

Run on a node with Habitat-Sim EGL/GPU access and HM3D data installed:

    RUN_HABITAT_E2E=1 uv run pytest tests/environments/test_habitat_e2e.py -v
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from src.environments.habitat import ACTIONS, build_habitat_env

HAS_HABITAT = importlib.util.find_spec("habitat_sim") is not None
RUN_HABITAT_E2E = os.environ.get("RUN_HABITAT_E2E") == "1"

pytestmark = [
    pytest.mark.habitat_sim,
    pytest.mark.integration,
    pytest.mark.skipif(
        not HAS_HABITAT,
        reason="habitat-sim not installed — run: uv sync --extra habitat",
    ),
    pytest.mark.skipif(
        not RUN_HABITAT_E2E,
        reason="requires RUN_HABITAT_E2E=1; loads Habitat-Sim + HM3D scenes",
    ),
]


@pytest.fixture(name="habitat_env")
def fixture_habitat_env():
    """Create a Habitat ObjectNav environment for e2e checks."""
    try:
        env = build_habitat_env(
            (3, 64, 64),
            max_episode_steps=50,
            split="val_mini",
        )
    except (FileNotFoundError, OSError, RuntimeError, AssertionError) as exc:
        pytest.skip(f"Habitat dataset/scene unavailable: {exc}")
    yield env
    env.close()


def test_reset_returns_chw_frame(habitat_env):
    """Reset returns a typed CHW RGB frame."""
    obs = habitat_env.reset()

    assert obs.is_first is True
    assert obs.is_episode_end is False
    assert obs.image.shape == (3, 64, 64)
    assert obs.image.dtype == np.uint8
    assert obs.scene_id is not None
    assert obs.episode_id is not None
    assert obs.step == 0


def test_step_updates_frame(habitat_env):
    """Step returns the next typed frame."""
    habitat_env.reset()

    obs = habitat_env.step(1)  # MOVE_FORWARD

    assert obs.is_first is False
    assert obs.step == 1


def test_random_rollout_stays_consistent(habitat_env):
    """Random actions keep frame shape and step metadata valid."""
    obs = habitat_env.reset()
    rng = np.random.default_rng(0)

    for _ in range(10):
        if obs.done:
            obs = habitat_env.reset()
        action = int(rng.integers(0, len(ACTIONS)))
        obs = habitat_env.step(action)

        assert obs.image.shape == (3, 64, 64)
        assert obs.step is not None
        assert obs.step >= 1
