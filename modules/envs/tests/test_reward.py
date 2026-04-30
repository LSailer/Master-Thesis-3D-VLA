"""Unit tests for configurable reward parameters.

Tests reward computation logic without Habitat — uses a mock env wrapper
with only the fields _compute_reward() depends on.
"""

import pytest
from modules.shared.configs import DreamerConfig
from modules.envs.habitat import HabitatObjectNavEnv


class _RewardTestEnv:
    """Minimal stand-in to test _compute_reward logic in isolation."""

    def __init__(self, cfg: DreamerConfig):
        self._cfg = cfg
        self._prev_dist = 0.0

    def _compute_reward(self, dist: float) -> float:
        return HabitatObjectNavEnv._compute_reward(self, dist)


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
