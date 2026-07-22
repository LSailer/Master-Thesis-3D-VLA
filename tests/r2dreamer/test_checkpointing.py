"""Tests for src/r2dreamer/checkpointing.py — checkpoint round-trip + snapshot."""

import json
import os
import tempfile

import jax
import numpy as np
import pytest

from src.configs.config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.checkpointing import (
    config_snapshot,
    load_checkpoint,
    save_checkpoint,
)
from src.r2dreamer.observation_preparation import CNNObservationPreparation


class TestCheckpoint:
    """save_checkpoint and load_checkpoint round-trip agent state."""

    @pytest.fixture
    def agent(self):
        cfg = R2DreamerConfig(obs_shape=(64, 64, 3), num_actions=4)
        rng = jax.random.PRNGKey(42)
        return R2DreamerAgent(cfg, rng)

    def test_roundtrip_preserves_params(self, agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=100, output_dir=tmpdir)
            assert os.path.exists(path)

            data = load_checkpoint(path)
            assert data["step"] == 100
            # Params should match
            for key in agent.params:
                orig = agent.params[key]
                loaded = data["params"][key]
                jax.tree.map(
                    lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6),
                    orig, loaded,
                )

    def test_roundtrip_preserves_ema_state(self, agent):
        """The old _save_checkpoint missed ema_state — verify it's saved now."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=50, output_dir=tmpdir)
            data = load_checkpoint(path)
            assert "ema_state" in data
            np.testing.assert_allclose(
                data["ema_state"], np.array(agent.ema_state), atol=1e-6
            )

    def test_roundtrip_preserves_slow_critic(self, agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=10, output_dir=tmpdir)
            data = load_checkpoint(path)
            jax.tree.map(
                lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6),
                agent.slow_critic_params, data["slow_critic_params"],
            )

    def test_checkpoint_persists_serializable_encoder_input_contract(self):
        cfg = R2DreamerConfig(obs_shape=(64, 64, 3), num_actions=4)
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(42))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=10, output_dir=tmpdir)
            data = load_checkpoint(path)

        snapshot = data["encoder_input_contract"]
        assert snapshot["encoder_module"] == "src.r2dreamer.encoders.cnn.ConvEncoder"
        assert snapshot["encoder_module_kwargs"] == {
            "depth": 16,
            "kernel_size": 5,
            "mults": (2, 3, 4, 4),
        }
        json.dumps(snapshot)

    def test_agent_from_checkpoint_recovers_encoder_contract_when_shape_omitted(self):
        cfg = R2DreamerConfig(
            obs_shape=(64, 64, 3),
            num_actions=4,
            encoder_input_contract=CNNObservationPreparation().contract.to_snapshot(),
        )
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(42))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=10, output_dir=tmpdir)
            recovered = R2DreamerAgent.from_checkpoint(
                path, num_actions=4, seed=0,
            )

        assert recovered.cfg.obs_shape == (64, 64, 3)
        assert recovered.cfg.encoder_type == "cnn"
        assert recovered.cfg.encoder_input_contract["encoder_type"] == "cnn"

    def test_agent_instantiates_encoder_module_from_contract_kwargs(self):
        contract = CNNObservationPreparation().contract.to_snapshot()
        contract["encoder_module_kwargs"] = {
            "depth": 8,
            "kernel_size": 3,
            "mults": (2, 2),
        }
        cfg = R2DreamerConfig(
            obs_shape=(64, 64, 3),
            num_actions=4,
            encoder_depth=16,
            encoder_kernel=5,
            encoder_mults=(2, 3, 4, 4),
            encoder_input_contract=contract,
        )

        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(42))

        assert agent.encoder_mod.depth == 8
        assert agent.encoder_mod.kernel_size == 3
        assert agent.encoder_mod.mults == (2, 2)


class TestConfigSnapshot:
    def test_config_snapshot_uses_serializable_encoder_contract_and_module_name(self):
        cfg = R2DreamerConfig(
            obs_shape=(64, 64, 3),
            num_actions=4,
            encoder_module_cls=CNNObservationPreparation().contract.encoder_module_cls,
            encoder_input_contract=CNNObservationPreparation().contract.to_snapshot(),
        )

        snapshot = config_snapshot(cfg)

        assert snapshot["encoder_module"] == "src.r2dreamer.encoders.cnn.ConvEncoder"
        assert "encoder_module_cls" not in snapshot
        assert snapshot["encoder_input_contract"]["encoder_type"] == "cnn"
        assert snapshot["encoder_input_contract"]["encoder_module_kwargs"] == {
            "depth": 16,
            "kernel_size": 5,
            "mults": (2, 3, 4, 4),
        }
        json.dumps(snapshot)

    def test_config_snapshot_derives_default_cnn_contract(self):
        snapshot = config_snapshot(R2DreamerConfig(obs_shape=(64, 64, 3), num_actions=4))

        assert snapshot["encoder_input_contract"]["encoder_type"] == "cnn"
        json.dumps(snapshot)
