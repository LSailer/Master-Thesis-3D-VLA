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


class TestEvalConfig:
    def test_eval_fields_exist(self, cfg):
        assert hasattr(cfg, "eval_every")
        assert hasattr(cfg, "eval_episodes")
        assert isinstance(cfg.eval_every, int)
        assert isinstance(cfg.eval_episodes, int)

    def test_eval_defaults(self, cfg):
        assert cfg.eval_every == 100_000
        assert cfg.eval_episodes == 10


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
