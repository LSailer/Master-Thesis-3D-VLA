"""Smoke test: Habitat env + DreamerV3 agent end-to-end."""

import numpy as np
import pytest

habitat = pytest.importorskip("habitat", reason="habitat not installed")


from dreamerv3.configs import DreamerConfig
from dreamerv3.env_habitat import HabitatObjectNavEnv
from dreamerv3.replay_buffer import ReplayBuffer


@pytest.fixture
def small_config():
    return DreamerConfig(
        obs_shape=(3, 64, 64),
        split="val_mini",
        max_episode_steps=50,
        buffer_capacity=1000,
    )


def test_env_reset_and_step(small_config):
    """Reset env, take 5 random steps, verify shapes and types."""
    env = HabitatObjectNavEnv(small_config)
    try:
        obs = env.reset()
        assert obs["image"].shape == (3, 64, 64)
        assert obs["image"].dtype == np.uint8
        assert obs["is_first"] is True

        transitions = []
        for _ in range(5):
            action = np.random.randint(0, small_config.num_actions)
            obs = env.step(action)
            assert obs["image"].shape == (3, 64, 64)
            assert isinstance(obs["reward"], float)
            assert isinstance(obs["done"], bool)
            transitions.append((obs, action))
            print(f"  action={action}  reward={obs['reward']:.3f}  done={obs['done']}")
    finally:
        env.close()


def test_agent_acts_on_env_obs(small_config):
    """Create env + agent + buffer, run a few steps end-to-end."""
    import jax

    from dreamerv3.agent import DreamerAgent

    env = HabitatObjectNavEnv(small_config)
    rng = jax.random.PRNGKey(0)
    agent = DreamerAgent(small_config, rng)
    buf = ReplayBuffer(small_config)

    try:
        obs = env.reset()
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs, act_key)
        assert 0 <= action < small_config.num_actions

        for _ in range(5):
            obs = env.step(action)
            buf.add(obs["image"], action, obs["reward"], obs["done"])
            rng, act_key = jax.random.split(rng)
            action = agent.act(obs, act_key)
            assert 0 <= action < small_config.num_actions

        assert buf.size == 5
        print(f"  buffer size={buf.size}, last action={action}")
    finally:
        env.close()
