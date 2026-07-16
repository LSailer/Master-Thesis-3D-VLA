"""Real Habitat parity for mutable and functional R2Dreamer acting state.

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

from src.configs.config import R2DreamerConfig
from src.environments.habitat import build_habitat_env
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.encoders import CNNEncoder

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
        obs_shape=(3, 64, 64),
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


def test_real_habitat_act_with_state_matches_mutable_acting():
    env = None
    try:
        try:
            env = build_habitat_env((3, 64, 64), max_episode_steps=20, mode="val_mini")
        except (FileNotFoundError, OSError, RuntimeError, AssertionError) as exc:
            pytest.skip(f"Habitat dataset/scene unavailable: {exc}")

        adapter = CNNEncoder().make_adapter()
        cfg = _small_agent_config()
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(7))
        state = agent.initial_act_state()
        obs = env.reset()
        prepared = adapter.prepare_env_step(obs)
        encoder_obs = prepared.encoder_obs
        is_first = prepared.is_first
        rng = jax.random.PRNGKey(11)
        forced_reset_checked = False

        for step in range(6):
            # Fold reset parity into the real stream: step 0 is env reset, step 3
            # forces a synthetic episode boundary through both APIs.
            compare_is_first = is_first
            if step == 3:
                compare_is_first = True
                forced_reset_checked = True

            rng, act_key = jax.random.split(rng)
            mutable_action = agent.act(
                encoder_obs, compare_is_first, act_key, training=False
            )
            state_action, state = agent.act_with_state(
                encoder_obs, compare_is_first, state, act_key, training=False
            )

            assert state_action == mutable_action
            assert _tree_allclose(state, agent.snapshot_act_state())

            obs = env.step(mutable_action)
            if obs.done:
                break
            prepared = adapter.prepare_env_step(obs)
            encoder_obs = prepared.encoder_obs
            is_first = prepared.is_first

        assert forced_reset_checked
    finally:
        if env is not None:
            env.close()
