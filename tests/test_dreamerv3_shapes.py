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


class TestImaginationMetrics:
    def test_imag_return_in_metrics(self, cfg, rng):
        """train_step must return imag_return metric."""
        from src.dreamerv3.agent import DreamerAgent
        agent = DreamerAgent(cfg, rng)
        B, T = cfg.batch_size, cfg.seq_len
        batch = {
            "obs": jnp.zeros((B, T, *cfg.obs_shape)),
            "actions": jnp.zeros((B, T), dtype=jnp.int32),
            "rewards": jnp.zeros((B, T)),
            "dones": jnp.zeros((B, T)),
            "is_first": jnp.zeros((B, T)),
        }
        metrics = agent.train_step(batch, rng)
        assert "imag_return" in metrics, "imag_return must be in train_step metrics"
        assert isinstance(metrics["imag_return"], float)


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
