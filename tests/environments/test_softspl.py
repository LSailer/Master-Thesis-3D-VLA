"""Unit tests for SoftSPL + DTG in HabitatObjectNavEnv.step().

Mocks habitat sim so the math can be exercised without a GPU. Sibling
test_spl.py is the GPU-only integration test that validates SPL end-to-end.
"""

import numpy as np
import pytest

from src.shared.configs import DreamerConfig
from src.environments.habitat import (
    GOAL_RADIUS,
    HabitatObjectNavEnv,
)


class _FakeAgentState:
    def __init__(self, position):
        self.position = position


class _FakeSim:
    def __init__(self, position):
        self._position = position

    def get_agent_state(self):
        return _FakeAgentState(self._position)


class _FakeHabitatEnv:
    def __init__(self, distance, position=(0.0, 0.0, 0.0)):
        self._distance = distance
        self.sim = _FakeSim(list(position))

    def step(self, action):
        return {"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}

    def get_metrics(self):
        return {"distance_to_goal": self._distance}


def _make_env(distance: float, *, start_geodesic: float, path_length: float,
              prev_position=(0.0, 0.0, 0.0), max_steps: int = 100) -> HabitatObjectNavEnv:
    env = object.__new__(HabitatObjectNavEnv)
    env._cfg = DreamerConfig(
        obs_shape=(3, 1, 1),
        max_episode_steps=max_steps,
        reward_type="sparse",
        step_penalty=0.0,
        success_bonus=1.0,
    )
    env._env = _FakeHabitatEnv(distance, position=prev_position)
    env._last_obs = {"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}
    env._prev_dist = start_geodesic
    env._start_geodesic = start_geodesic
    # Step count chosen so the upcoming step() lands at max_steps -> done=True
    env._step_count = max_steps - 1
    env._path_length = path_length
    env._prev_position = np.array(prev_position)
    env._collisions = 0
    env._forward_steps = 0
    return env


def test_softspl_on_success_uses_residual_distance():
    """On success SoftSPL = (1 - d_final/d_init) * length_ratio — never quite SPL.

    d_final inside GOAL_RADIUS triggers success=1 but softspl reflects the
    fact that the agent stopped 0.1m from the goal, not exactly on it.
    """
    env = _make_env(distance=0.1, start_geodesic=5.0, path_length=5.0)
    obs = env.step(1)
    assert obs["success"] == 1.0
    assert obs["spl"] == pytest.approx(1.0)
    # progress = 1 - 0.1/5.0 = 0.98 ; length_ratio = 1.0 -> softspl 0.98
    assert obs["softspl"] == pytest.approx(0.98)
    assert obs["dtg"] == pytest.approx(0.1)


def test_softspl_partial_progress_no_success():
    """Failed episode that made progress: SoftSPL > 0, SPL = 0."""
    # d_init = 10.0, d_final = 3.0 -> progress = 0.7
    # path_length = 10.0, shortest = 10.0 -> length_ratio = 1.0
    # softspl = 0.7
    env = _make_env(distance=3.0, start_geodesic=10.0, path_length=10.0)
    obs = env.step(1)
    assert obs["success"] == 0.0
    assert obs["spl"] == 0.0
    assert obs["softspl"] == pytest.approx(0.7)
    assert obs["dtg"] == pytest.approx(3.0)


def test_softspl_clipped_when_agent_moves_away():
    """If d_final > d_init, the progress term is clipped to 0."""
    env = _make_env(distance=12.0, start_geodesic=10.0, path_length=15.0)
    obs = env.step(1)
    assert obs["success"] == 0.0
    assert obs["softspl"] == 0.0
    assert obs["dtg"] == pytest.approx(12.0)


def test_softspl_length_ratio_penalises_long_paths():
    """A long wandering path reduces SoftSPL even when progress is full."""
    # progress = 1 - 1/10 = 0.9
    # path_length = 30, shortest = 10 -> length_ratio = 10/30 = 1/3
    # softspl = 0.9 / 3 = 0.3
    env = _make_env(distance=1.0, start_geodesic=10.0, path_length=30.0)
    obs = env.step(1)
    assert obs["softspl"] == pytest.approx(0.3)


def test_softspl_zero_when_start_geodesic_zero():
    """Defensive: if start_geodesic is 0 (degenerate), SoftSPL must be 0."""
    env = _make_env(distance=0.5, start_geodesic=0.0, path_length=0.0)
    obs = env.step(1)
    assert obs["softspl"] == 0.0


def test_softspl_zero_mid_episode():
    """SoftSPL/DTG are 0 mid-episode (only meaningful at done)."""
    env = _make_env(distance=5.0, start_geodesic=10.0, path_length=2.0, max_steps=100)
    env._step_count = 0  # step() takes us to 1, far below max
    obs = env.step(1)
    assert obs["done"] is False
    assert obs["softspl"] == 0.0
    assert obs["dtg"] == 0.0


def test_softspl_stop_action_timeout_uses_prev_dist():
    """STOP at max_steps still surfaces SoftSPL/DTG using _prev_dist."""
    env = _make_env(distance=999.0, start_geodesic=10.0, path_length=8.0)
    env._prev_dist = 4.0  # last known distance before STOP
    obs = env.step(0)
    assert obs["done"] is True
    # progress = 1 - 4/10 = 0.6, length_ratio = 10/10 = 1.0 -> softspl 0.6
    assert obs["softspl"] == pytest.approx(0.6)
    assert obs["dtg"] == pytest.approx(4.0)


def test_softspl_in_unit_interval():
    """Property: softspl ∈ [0, 1] for all reasonable inputs."""
    for d_init in (1.0, 5.0, 20.0):
        for d_final in (0.0, 0.5, 1.0, d_init, d_init * 2):
            for path in (d_init, d_init * 1.5, d_init * 3):
                env = _make_env(
                    distance=d_final, start_geodesic=d_init, path_length=path,
                )
                obs = env.step(1)
                assert 0.0 <= obs["softspl"] <= 1.0, (
                    f"d_init={d_init} d_final={d_final} path={path} "
                    f"softspl={obs['softspl']}"
                )
