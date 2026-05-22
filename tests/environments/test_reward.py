"""Unit tests for configurable reward parameters.

Tests reward computation logic without Habitat — uses a mock env wrapper
with only the fields _compute_reward() depends on.
"""

import pytest
import numpy as np
from src.shared.configs import DreamerConfig
from src.environments.habitat import (
    GOAL_RADIUS,
    HabitatObjectNavEnv,
    _is_success_distance,
)


class _RewardTestEnv:
    """Minimal stand-in to test _compute_reward logic in isolation."""

    def __init__(self, cfg: DreamerConfig):
        self._cfg = cfg
        self._prev_dist = 0.0

    def _compute_reward(self, dist: float) -> float:
        return HabitatObjectNavEnv._compute_reward(self, dist)


class _FakeAgentState:
    def __init__(self, position):
        self.position = position


class _FakeSim:
    def __init__(self, position):
        self._position = position

    def get_agent_state(self):
        return _FakeAgentState(self._position)


class _FakeHabitatEnv:
    def __init__(self, distance):
        self._distance = distance
        self.sim = _FakeSim([0.1, 0.0, 0.0])

    def step(self, action):
        assert action == 1
        return {"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}

    def get_metrics(self):
        return {"distance_to_goal": self._distance}


def _make_step_test_env(distance: float) -> HabitatObjectNavEnv:
    env = object.__new__(HabitatObjectNavEnv)
    env._cfg = DreamerConfig(
        obs_shape=(3, 1, 1),
        max_episode_steps=10,
        reward_type="geodesic_delta",
        step_penalty=0.0,
        success_bonus=1.0,
    )
    env._env = _FakeHabitatEnv(distance)
    env._last_obs = {"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}
    env._prev_dist = 1.0
    env._start_geodesic = 1.0
    env._step_count = 0
    env._path_length = 0.0
    env._prev_position = [0.0, 0.0, 0.0]
    env._collisions = 0
    env._forward_steps = 0
    return env


def test_config_defaults():
    cfg = DreamerConfig()
    assert cfg.step_penalty == -0.01
    assert cfg.success_bonus == 10.0


def test_geodesic_delta_closer():
    """Moving closer gives positive delta + step penalty."""
    cfg = DreamerConfig(reward_type="geodesic_delta")
    env = _RewardTestEnv(cfg)
    env._prev_dist = 5.0
    reward = env._compute_reward(4.75)  # 0.25m closer
    assert reward == pytest.approx(0.25 + cfg.step_penalty)


def test_geodesic_delta_further():
    """Moving further gives negative delta + step penalty."""
    cfg = DreamerConfig(reward_type="geodesic_delta")
    env = _RewardTestEnv(cfg)
    env._prev_dist = 5.0
    reward = env._compute_reward(5.25)  # 0.25m further
    assert reward == pytest.approx(-0.25 + cfg.step_penalty)


def test_geodesic_delta_no_movement():
    """No movement gives zero delta + step penalty."""
    cfg = DreamerConfig(reward_type="geodesic_delta")
    env = _RewardTestEnv(cfg)
    env._prev_dist = 5.0
    reward = env._compute_reward(5.0)
    assert reward == pytest.approx(cfg.step_penalty)


def test_geodesic_delta_success():
    """Reaching goal gives delta + success_bonus + step penalty."""
    cfg = DreamerConfig(reward_type="geodesic_delta", success_bonus=10.0)
    env = _RewardTestEnv(cfg)
    env._prev_dist = 0.3
    dist = 0.1  # within GOAL_RADIUS
    reward = env._compute_reward(dist)
    expected = (0.3 - 0.1) + 10.0 + cfg.step_penalty
    assert reward == pytest.approx(expected)


def test_custom_success_bonus():
    """Custom success_bonus is used instead of hardcoded 10.0."""
    cfg = DreamerConfig(reward_type="geodesic_delta", success_bonus=5.0)
    env = _RewardTestEnv(cfg)
    env._prev_dist = 0.3
    dist = 0.1
    reward = env._compute_reward(dist)
    expected = (0.3 - 0.1) + 5.0 + cfg.step_penalty
    assert reward == pytest.approx(expected)


def test_custom_step_penalty():
    """Custom step_penalty value is applied."""
    cfg = DreamerConfig(reward_type="geodesic_delta", step_penalty=-0.05)
    env = _RewardTestEnv(cfg)
    env._prev_dist = 5.0
    reward = env._compute_reward(5.0)  # no movement
    assert reward == pytest.approx(-0.05)


def test_zero_step_penalty():
    """step_penalty=0 recovers original behavior (no penalty)."""
    cfg = DreamerConfig(reward_type="geodesic_delta", step_penalty=0.0)
    env = _RewardTestEnv(cfg)
    env._prev_dist = 5.0
    reward = env._compute_reward(4.75)
    assert reward == pytest.approx(0.25)


def test_sparse_reward_uses_success_bonus():
    """Sparse mode uses configurable success_bonus."""
    cfg = DreamerConfig(reward_type="sparse", success_bonus=15.0)
    env = _RewardTestEnv(cfg)
    reward = env._compute_reward(0.1)  # within GOAL_RADIUS
    assert reward == pytest.approx(15.0 + cfg.step_penalty)


def test_sparse_reward_failure():
    """Sparse mode returns only step penalty on failure."""
    cfg = DreamerConfig(reward_type="sparse", success_bonus=10.0)
    env = _RewardTestEnv(cfg)
    reward = env._compute_reward(5.0)  # far from goal
    assert reward == pytest.approx(cfg.step_penalty)


@pytest.mark.parametrize(
    ("dist", "expected"),
    [
        (GOAL_RADIUS, False),
        (GOAL_RADIUS - 1e-6, True),
        (GOAL_RADIUS + 1e-6, False),
    ],
)
def test_success_boundary_is_strictly_inside_goal_radius(dist, expected):
    assert _is_success_distance(dist) is expected


@pytest.mark.parametrize("dist", [None, float("nan"), float("inf"), -0.01])
def test_invalid_goal_distances_raise_clear_error(dist):
    cfg = DreamerConfig(reward_type="geodesic_delta")
    env = _RewardTestEnv(cfg)
    with pytest.raises(ValueError, match="distance_to_goal"):
        env._compute_reward(dist)


def test_invalid_distance_does_not_update_prev_dist():
    cfg = DreamerConfig(reward_type="geodesic_delta")
    env = _RewardTestEnv(cfg)
    env._prev_dist = 5.0
    with pytest.raises(ValueError):
        env._compute_reward(float("nan"))
    assert env._prev_dist == 5.0


def test_unknown_reward_type_raises_explicit_error():
    cfg = DreamerConfig(reward_type="dense-but-misspelled")
    env = _RewardTestEnv(cfg)
    with pytest.raises(ValueError, match="Unknown reward_type"):
        env._compute_reward(5.0)


def test_step_marks_success_distance_as_done():
    env = _make_step_test_env(GOAL_RADIUS - 1e-6)
    obs = env.step(1)
    assert obs["success"] == 1.0
    assert obs["done"] is True
    assert obs["spl"] > 0.0


def test_step_does_not_end_episode_at_exact_goal_radius():
    env = _make_step_test_env(GOAL_RADIUS)
    obs = env.step(1)
    assert obs["success"] == 0.0
    assert obs["done"] is False
    assert obs["spl"] == 0.0
