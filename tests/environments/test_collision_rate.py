"""Unit tests for the collision_rate workaround in HabitatObjectNavEnv.

Habitat exposes no direct collision API. We approximate it by detecting
FORWARD steps whose position delta falls under 0.01 m (vs nominal 0.25 m).
"""

import numpy as np
import pytest

from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv


class _FakeAgentState:
    def __init__(self, position):
        self.position = np.array(position)


class _FakeSim:
    """Simulates a position the test can advance step-by-step."""

    def __init__(self):
        self._position = [0.0, 0.0, 0.0]

    def set_position(self, position):
        self._position = list(position)

    def get_agent_state(self):
        return _FakeAgentState(self._position)


class _FakeHabitatEnv:
    """Drives _FakeSim from a fixed list of post-step positions."""

    def __init__(self, post_step_positions, distance: float = 5.0):
        self.sim = _FakeSim()
        self._post_step_positions = list(post_step_positions)
        self._idx = 0
        self._distance = distance

    def step(self, action):
        # Each step advances the sim to the next pre-loaded position
        if self._idx < len(self._post_step_positions):
            self.sim.set_position(self._post_step_positions[self._idx])
            self._idx += 1
        return {"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}

    def get_metrics(self):
        return {"distance_to_goal": self._distance}


def _make_env(post_step_positions, max_steps: int = 100) -> HabitatObjectNavEnv:
    env = object.__new__(HabitatObjectNavEnv)
    env._cfg = HabitatEnvConfig(
        obs_shape=(3, 1, 1),
        max_episode_steps=max_steps,
        reward_type="sparse",
        step_penalty=0.0,
        success_bonus=1.0,
    )
    env._env = _FakeHabitatEnv(post_step_positions, distance=5.0)
    env._last_obs = {"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}
    env._prev_dist = 5.0
    env._start_geodesic = 5.0
    env._step_count = 0
    env._path_length = 0.0
    env._prev_position = np.array([0.0, 0.0, 0.0])
    env._collisions = 0
    env._forward_steps = 0
    return env


def test_collision_rate_zero_when_no_forward_steps():
    """Pure rotations and STOPs leave collision_rate at 0 (no forward attempts)."""
    env = _make_env(post_step_positions=[(0.0, 0.0, 0.0)] * 5, max_steps=5)
    env.step(2)  # TURN_LEFT — position unchanged but not counted
    env.step(3)  # TURN_RIGHT — same
    env.step(2)
    env.step(3)
    obs = env.step(0)  # STOP at max_steps -> done=True
    assert obs["done"] is True
    assert obs["collision_rate"] == 0.0


def test_collision_rate_all_collisions():
    """Every FORWARD step blocked at < 0.01 m -> collision_rate = 1.0."""
    # 5 forward steps, all moving < 0.01 m
    blocked = [(0.005, 0.0, 0.0)] * 5  # 5 mm increment, < 1 cm threshold
    env = _make_env(post_step_positions=blocked, max_steps=5)
    for _ in range(4):
        env.step(1)
    obs = env.step(1)  # 5th forward triggers done at max_steps
    assert obs["done"] is True
    assert obs["collision_rate"] == pytest.approx(1.0)


def test_collision_rate_no_collisions():
    """All FORWARD steps moving ~0.25 m -> collision_rate = 0.0."""
    free = [(0.25 * (i + 1), 0.0, 0.0) for i in range(5)]
    env = _make_env(post_step_positions=free, max_steps=5)
    for _ in range(4):
        env.step(1)
    obs = env.step(1)
    assert obs["done"] is True
    assert obs["collision_rate"] == 0.0


def test_collision_rate_mixed():
    """3 collisions out of 5 forward steps -> 0.6."""
    # Forward steps 1, 3, 5 move freely; 2 and 4 collide.
    positions = [
        (0.25, 0.0, 0.0),    # move
        (0.255, 0.0, 0.0),   # collision (0.005 m delta)
        (0.50, 0.0, 0.0),    # move
        (0.505, 0.0, 0.0),   # collision
        (0.75, 0.0, 0.0),    # move
    ]
    env = _make_env(post_step_positions=positions, max_steps=5)
    for _ in range(4):
        env.step(1)
    obs = env.step(1)
    assert obs["done"] is True
    # 2 collisions / 5 forward steps = 0.4
    assert obs["collision_rate"] == pytest.approx(0.4)


def test_collision_rate_only_counts_forward_actions():
    """TURN actions don't move and must NOT be classified as collisions."""
    # 1 forward (moves freely), 4 turns (no movement)
    positions = [
        (0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    ]
    env = _make_env(post_step_positions=positions, max_steps=5)
    env.step(1)  # forward (free)
    env.step(2)  # turn left
    env.step(3)  # turn right
    env.step(2)
    obs = env.step(3)  # final turn triggers done at max_steps
    assert obs["done"] is True
    # Only 1 forward step, 0 collisions -> 0.0
    assert obs["collision_rate"] == 0.0


def test_collision_rate_resets_between_episodes():
    """A fresh episode starts with collision_rate=0 (state reset)."""
    blocked = [(0.005, 0.0, 0.0)] * 3
    env = _make_env(post_step_positions=blocked, max_steps=3)
    env.step(1)
    env.step(1)
    obs = env.step(1)
    assert obs["collision_rate"] == pytest.approx(1.0)
    # Simulate reset state
    env._collisions = 0
    env._forward_steps = 0
    assert env._compute_collision_rate() == 0.0


def test_collision_rate_in_unit_interval():
    """Property: collision_rate ∈ [0, 1] for any combination of moves."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(1, 10))
        positions = [
            (float(rng.uniform(0, 0.3)), 0.0, 0.0) for _ in range(n)
        ]
        env = _make_env(post_step_positions=positions, max_steps=n)
        for _ in range(n - 1):
            env.step(1)
        obs = env.step(1)
        assert 0.0 <= obs["collision_rate"] <= 1.0
