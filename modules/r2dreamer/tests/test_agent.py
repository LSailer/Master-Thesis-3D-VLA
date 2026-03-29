"""Smoke tests for R2DreamerAgent."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modules.r2dreamer.config import R2DreamerConfig
from modules.r2dreamer.agent import R2DreamerAgent


@pytest.fixture
def cfg():
    return R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=17)


@pytest.fixture
def agent(cfg):
    return R2DreamerAgent(cfg, jax.random.PRNGKey(42))


def make_batch(cfg, B=4, T=16):
    return {
        "obs": jnp.array(np.random.rand(B, T, *cfg.obs_shape).astype(np.float32)),
        "actions": jnp.array(np.eye(cfg.num_actions, dtype=np.float32)[
            np.random.randint(0, cfg.num_actions, (B, T))]),
        "rewards": jnp.array(np.random.randn(B, T).astype(np.float32)),
        "is_first": jnp.zeros((B, T)),
        "is_last": jnp.zeros((B, T)),
        "is_terminal": jnp.zeros((B, T)),
    }


class TestR2DreamerAgent:
    def test_init(self, agent):
        assert agent is not None

    def test_act(self, agent, cfg):
        obs = {"image": np.random.randint(0, 256, cfg.obs_shape, dtype=np.uint8), "is_first": True}
        action = agent.act(obs, jax.random.PRNGKey(0))
        assert 0 <= action < cfg.num_actions

    def test_train_step_produces_metrics(self, agent, cfg):
        batch = make_batch(cfg)
        metrics = agent.train_step(batch, jax.random.PRNGKey(1))
        assert "loss/barlow" in metrics
        assert "loss/dyn" in metrics
        assert "loss/rew" in metrics
        assert "loss/policy" in metrics
        assert "loss/value" in metrics
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} = {v}"

    def test_train_step_does_not_diverge(self, agent, cfg):
        batch = make_batch(cfg)
        rng = jax.random.PRNGKey(2)
        losses = []
        for _ in range(3):
            rng, k = jax.random.split(rng)
            m = agent.train_step(batch, k)
            losses.append(m["total_loss"])
        # Should not be NaN or explode
        assert all(np.isfinite(l) for l in losses)
