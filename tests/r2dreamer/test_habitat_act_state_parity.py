"""Real Habitat parity for the functional R2Dreamer acting carry.

The jitted ``act`` takes ``self`` as a static argument, so nothing may ride on
the agent object between steps: replaying a step from the same carry, key and
observation must reproduce it exactly, and ``is_first`` must reset the carry.
Both are asserted against a real Habitat observation stream, because the CPU
tests feed synthetic fields.

Run on a GPU node with Habitat-Sim/HM3D available:

    RUN_HABITAT_ACT_STATE_PARITY=1 uv run pytest \
      tests/r2dreamer/test_habitat_act_state_parity.py -q
"""

from __future__ import annotations

import importlib.util
import os

import jax
import numpy as np
import pytest

from src.adapters import ADAPTERS
from src.adapters.contract import encoder_obs_from_fields
from src.configs.config import R2DreamerConfig
from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv
from src.r2dreamer.agent import R2DreamerAgent

HAS_HABITAT = importlib.util.find_spec("habitat_sim") is not None
RUN_PARITY = os.environ.get("RUN_HABITAT_ACT_STATE_PARITY") == "1"

pytestmark = [
    pytest.mark.habitat_sim,
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_HABITAT, reason="habitat-sim not installed"),
    pytest.mark.skipif(
        not RUN_PARITY,
        reason="requires RUN_HABITAT_ACT_STATE_PARITY=1; loads Habitat-Sim + HM3D",
    ),
]


def _small_agent_config() -> R2DreamerConfig:
    return R2DreamerConfig(
        adapter="rgb",
        num_actions=4,
        deter_size=32,
        hidden_size=16,
        stoch_classes=4,
        stoch_discrete=4,
        blocks=4,
        encoder_depth=4,
        encoder_kernel=3,
        encoder_mults=(1, 1, 1, 1),
        mlp_units=16,
        mlp_layers_reward=1,
        mlp_layers_cont=1,
        mlp_layers_actor=1,
        mlp_layers_critic=1,
        twohot_bins=21,
        imagination_horizon=2,
        horizon=10,
        lr=1e-3,
        warmup_steps=0,
    )


def _tree_allclose(left, right, *, atol=1e-6) -> bool:
    leaves = zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    return all(np.allclose(np.asarray(a), np.asarray(b), atol=atol) for a, b in leaves)


def test_real_habitat_act_is_pure_and_resets_on_is_first():
    env = None
    try:
        try:
            env = HabitatObjectNavEnv(
                HabitatEnvConfig(
                    obs_shape=(64, 64, 3),
                    max_episode_steps=20,
                    mode="train",
                )
            )
        except (FileNotFoundError, OSError, RuntimeError, AssertionError) as exc:
            pytest.skip(f"Habitat dataset/scene unavailable: {exc}")

        observe = ADAPTERS["rgb"]()
        cfg = _small_agent_config()
        obs = env.reset()
        fields = observe(obs)
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(7), fields=fields)
        state = agent.initial_act_state()
        encoder_obs = encoder_obs_from_fields(fields)
        is_first = obs.is_first
        rng = jax.random.PRNGKey(11)
        forced_reset_checked = False

        for step in range(6):
            # Fold reset parity into the real stream: step 0 is env reset, step 3
            # forces a synthetic episode boundary.
            compare_is_first = is_first
            if step == 3:
                compare_is_first = True

            rng, act_key = jax.random.split(rng)
            stale = state
            action, state = agent.act(
                agent.params, encoder_obs, compare_is_first, stale, act_key, False
            )
            replay_action, replay_state = agent.act(
                agent.params, encoder_obs, compare_is_first, stale, act_key, False
            )

            # Purity: no hidden state on the agent between two identical calls.
            assert int(replay_action) == int(action)
            assert _tree_allclose(replay_state, state)

            if compare_is_first:
                # is_first must drop the carry, whatever it held before.
                fresh_action, fresh_state = agent.act(
                    agent.params,
                    encoder_obs,
                    False,
                    agent.initial_act_state(),
                    act_key,
                    False,
                )
                assert int(fresh_action) == int(action)
                assert _tree_allclose(fresh_state, state)
                forced_reset_checked = True

            obs = env.step(int(action))
            if obs.done:
                break
            encoder_obs = encoder_obs_from_fields(observe(obs))
            is_first = obs.is_first

        assert forced_reset_checked
    finally:
        if env is not None:
            env.close()
