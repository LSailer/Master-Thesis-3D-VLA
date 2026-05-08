"""Integration tests for StepDriver — the acting-step seam.

These tests exercise the contract that used to be split across the
trainer's four-place reset coordination dance: env reset, adapter reset
(extractor cache flush), and RSSM acting-state zero must all fire
atomically when the previous step ended an episode.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from modules.r2dreamer.adapters import ObsAdapter
from modules.r2dreamer.agent import R2DreamerAgent
from modules.r2dreamer.config import R2DreamerConfig
from modules.r2dreamer.step_driver import StepDriver, Transition


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Configurable fake env. Returns done=True on the steps in ``done_steps``.

    Tracks reset/step calls so tests can assert on them.
    """

    def __init__(self, done_steps: list[int] | None = None):
        self._done_steps = set(done_steps or [])
        self._step_count = 0
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self) -> dict:
        self.reset_calls += 1
        self._step_count = 0
        return {
            "image": np.zeros((3, 64, 64), dtype=np.uint8),
            "is_first": True,
            "reward": 0.0,
            "done": False,
            "success": 0.0,
        }

    def step(self, action: int) -> dict:
        self.step_calls += 1
        self._step_count += 1
        is_done = self._step_count in self._done_steps
        return {
            "image": np.full((3, 64, 64), action % 256, dtype=np.uint8),
            "is_first": False,
            "reward": 1.0 if is_done else 0.1,
            "done": is_done,
            "success": 1.0 if is_done else 0.0,
        }

    def close(self) -> None:
        pass


class _CountingAdapter(ObsAdapter):
    """ObsAdapter subclass that tracks reset() calls."""

    def __init__(self):
        super().__init__()
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


@pytest.fixture
def cfg():
    return R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=4)


@pytest.fixture
def agent(cfg):
    return R2DreamerAgent(cfg, jax.random.PRNGKey(42))


def _make_driver(agent, done_steps=None):
    env = _FakeEnv(done_steps=done_steps)
    adapter = _CountingAdapter()
    driver = StepDriver(env=env, agent=agent, obs_adapter=adapter)
    return driver, env, adapter


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestFirstCallIsAFreshStart:
    """Constructor leaves previous_done=True; first step() resets everything."""

    def test_fires_env_reset_adapter_reset_and_zeroes_state(self, agent):
        driver, env, adapter = _make_driver(agent)
        # Pre-conditions: nothing called yet; state already zeroed by ctor.
        assert env.reset_calls == 0
        assert adapter.reset_calls == 0

        # Tickle some non-zero state to prove _reset_episode zeroes it.
        driver._stoch[:] = 0.5
        driver._deter[:] = 0.5
        driver._prev_action[:] = 0.5

        driver.step(jax.random.PRNGKey(0))

        assert env.reset_calls == 1
        assert adapter.reset_calls == 1
        # State zeroed at boundary (then policy_step writes new values for the
        # *next* call's prev state — but prev_action[0,action]=1 was set after).
        # Easier check: stoch was zeroed BEFORE policy_step ran. Asserting
        # post-state would conflate boundary-reset with policy-step output.
        # Instead: assert the agent saw zeroed state by checking that
        # peek_state() != the manually-tickled values.
        snap = driver.peek_state()
        assert not np.allclose(snap["stoch"], 0.5)
        assert not np.allclose(snap["deter"], 0.5)


class TestMidEpisodeStepsDoNotReset:
    """While done=False, step() must call env.step() (not reset) and not
    re-reset the adapter or zero RSSM state."""

    def test_no_reset_for_three_consecutive_non_terminal_steps(self, agent):
        driver, env, adapter = _make_driver(agent, done_steps=[])  # never done
        rng = jax.random.PRNGKey(1)

        for _ in range(3):
            rng, k = jax.random.split(rng)
            t, ended = driver.step(k)
            assert ended is False
            assert t.done is False

        # Exactly one reset (the initial one); three step() calls total.
        assert env.reset_calls == 1
        assert env.step_calls == 3
        assert adapter.reset_calls == 1


class TestEpisodeBoundaryFiresAtomicReset:
    """A step that returns done=True records done=True in the transition;
    the *next* step fires env.reset, adapter.reset, and zeros RSSM state."""

    def test_boundary_then_fresh_start(self, agent):
        driver, env, adapter = _make_driver(agent, done_steps=[2])
        rng = jax.random.PRNGKey(2)

        rng, k = jax.random.split(rng)
        t1, ended1 = driver.step(k)
        assert ended1 is False
        assert t1.done is False

        # Step 2: env returns done=True. Transition records it. NOT yet reset.
        rng, k = jax.random.split(rng)
        t2, ended2 = driver.step(k)
        assert ended2 is True
        assert t2.done is True
        assert t2.terminal is True
        assert env.reset_calls == 1  # still only the initial reset
        assert adapter.reset_calls == 1

        # Step 3: previous_done=True triggers the cascade.
        rng, k = jax.random.split(rng)
        t3, ended3 = driver.step(k)
        assert ended3 is False
        assert env.reset_calls == 2  # now reset for the next episode
        assert adapter.reset_calls == 2

        # State was zeroed at the boundary; policy_step then wrote new values
        # for the action just taken. Just assert prev_action is one-hot
        # for *this* action, not whatever was held before.
        snap = driver.peek_state()
        assert snap["prev_action"].shape == (1, agent.cfg.num_actions)
        assert np.isclose(snap["prev_action"].sum(), 1.0)


class TestSeamWinsOverBuggyEnv:
    """If env returns is_first=False on the first obs of a new episode, the
    seam still triggers the full reset cascade because previous_done was True.
    This is the paranoia test for Fork B."""

    class _BuggyEnv(_FakeEnv):
        def reset(self) -> dict:
            obs = super().reset()
            obs["is_first"] = False  # buggy
            return obs

    def test_reset_cascade_fires_despite_env_lying(self, agent):
        env = self._BuggyEnv(done_steps=[1])
        adapter = _CountingAdapter()
        driver = StepDriver(env=env, agent=agent, obs_adapter=adapter)
        rng = jax.random.PRNGKey(3)

        rng, k = jax.random.split(rng)
        _, ended = driver.step(k)
        assert ended is True
        assert env.reset_calls == 1
        assert adapter.reset_calls == 1

        # Tickle state so we can detect zeroing.
        driver._stoch[:] = 0.7
        driver._deter[:] = 0.7

        rng, k = jax.random.split(rng)
        driver.step(k)
        assert env.reset_calls == 2
        assert adapter.reset_calls == 2
        snap = driver.peek_state()
        assert not np.allclose(snap["stoch"], 0.7)
        assert not np.allclose(snap["deter"], 0.7)


class TestRandomAndAgentPoliciesShareTheSeam:
    """Same StepDriver, two policies (random for prefill, agent for training).
    Reset behaviour is identical; only the action stream differs."""

    def test_random_mode_does_not_invoke_agent(self, agent):
        driver, env, adapter = _make_driver(agent, done_steps=[2])
        rng = jax.random.PRNGKey(4)

        # Three random steps with one episode boundary in the middle.
        actions = []
        for _ in range(3):
            rng, k = jax.random.split(rng)
            t, _ = driver.step(k, policy="random")
            actions.append(t.action)

        # Random policy should produce ints in [0, num_actions).
        for a in actions:
            assert 0 <= a < agent.cfg.num_actions

        # Reset cascade fired the same way as in the agent-policy test.
        assert env.reset_calls == 2  # initial + post-boundary
        assert adapter.reset_calls == 2

    def test_unknown_policy_raises(self, agent):
        driver, _, _ = _make_driver(agent)
        with pytest.raises(ValueError, match="Unknown policy"):
            driver.step(jax.random.PRNGKey(0), policy="banana")  # type: ignore[arg-type]
