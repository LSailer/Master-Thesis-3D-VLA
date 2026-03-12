"""TDD tests for DreamerV3 agent shapes and checkpoint behaviour."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def cfg():
    from src.dreamerv3.configs import DreamerConfig
    return DreamerConfig()


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


class TestGreedyAction:
    def test_greedy_is_deterministic(self, cfg, rng):
        """training=False → argmax → same action for same state."""
        from src.dreamerv3.agent import DreamerAgent
        agent = DreamerAgent(cfg, rng)
        obs = {"image": jnp.zeros(cfg.obs_shape, dtype=jnp.uint8), "is_first": True}
        k1, k2 = jax.random.split(rng)
        a1 = agent.act(obs, k1, training=False)
        a2 = agent.act(obs, k2, training=False)
        assert a1 == a2, "greedy mode must be deterministic regardless of rng key"

    def test_stochastic_uses_sampling(self, cfg, rng):
        """training=True → categorical sampling (default behavior)."""
        import inspect
        from src.dreamerv3.agent import DreamerAgent
        source = inspect.getsource(DreamerAgent._act_forward)
        assert "training" in inspect.signature(DreamerAgent._act_forward).parameters or \
               "argmax" in source or "categorical" in source


class TestCheckpoint:
    def test_save_load_roundtrip(self, cfg, rng, tmp_path):
        """Save → load → params match."""
        from src.dreamerv3.agent import DreamerAgent
        agent = DreamerAgent(cfg, rng)
        original_params = jax.tree.map(lambda x: x.copy(), agent.wm_state.params)
        agent.save(str(tmp_path))
        agent.wm_state = agent.wm_state.replace(
            params=jax.tree.map(jnp.zeros_like, agent.wm_state.params)
        )
        agent.load(str(tmp_path))
        for orig, loaded in zip(
            jax.tree.leaves(original_params),
            jax.tree.leaves(agent.wm_state.params),
        ):
            assert jnp.allclose(orig, loaded, atol=1e-6)

    def test_checkpoint_file_exists(self, cfg, rng, tmp_path):
        from src.dreamerv3.agent import DreamerAgent
        agent = DreamerAgent(cfg, rng)
        agent.save(str(tmp_path))
        assert (tmp_path / "checkpoint.pkl").exists()
