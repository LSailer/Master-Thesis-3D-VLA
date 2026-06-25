"""Tests for src/r2dreamer/trainer.py — convert_batch and checkpoint."""

import json
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.environments.observation import ObservationFrame
from src.r2dreamer.config import R2DreamerConfig, TrainerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.adapters import ObsAdapter
from src.r2dreamer.observation_preparation import (
    CNNObservationPreparation,
    PreparedObservation,
)
from src.r2dreamer.trainer import (
    Trainer,
    config_snapshot,
    convert_batch,
    load_checkpoint,
    save_checkpoint,
)


def test_trainer_config_defaults_to_scalars_only_no_validation_or_video():
    cfg = TrainerConfig(output_dir="/tmp/r2dreamer-test")

    assert cfg.val_every == 0
    assert cfg.video_log_every == 0
    assert cfg.val_video_episodes == 0
    assert cfg.video_log_episodes == 0


class _DummyEnv:
    """Minimal env stub — Trainer.__init__ does not call any of these methods."""

    def reset(self) -> ObservationFrame:
        return ObservationFrame(
            image=np.zeros((3, 64, 64), dtype=np.uint8),
            is_first=True,
        )

    def step(self, action: int) -> ObservationFrame:
        return ObservationFrame(
            image=np.zeros((3, 64, 64), dtype=np.uint8),
            is_first=False,
            previous_action=int(action),
        )

    def close(self) -> None:
        pass


class _TinyCNNEnv:
    """Small deterministic env for a full CNN Trainer pipeline smoke test."""

    def __init__(self):
        self.t = 0
        self.closed = False

    def reset(self) -> ObservationFrame:
        self.t = 0
        return ObservationFrame(
            image=np.zeros((3, 64, 64), dtype=np.uint8),
            is_first=True,
        )

    def step(self, action: int) -> ObservationFrame:
        self.t += 1
        done = self.t >= 4
        return ObservationFrame(
            image=np.full((3, 64, 64), self.t, dtype=np.uint8),
            is_first=False,
            previous_action=int(action),
            reward=1.0,
            done=done,
        )

    def close(self) -> None:
        self.closed = True


class _MappingObsAdapter(ObsAdapter):
    def __init__(self):
        super().__init__(
            buffer_dtype={"image": "uint8", "wp_cp": "float32"},
            buffer_shape={"image": (3, 64, 64), "wp_cp": (4116,)},
            normalize_on_sample={"image": False, "wp_cp": False},
            agent_obs_shape=(16404,),
        )

    def transform(self, env_obs: ObservationFrame) -> tuple[dict[str, np.ndarray], dict]:
        return {
            "image": env_obs.image,
            "wp_cp": np.ones((4116,), dtype=np.float32),
        }, {"image": env_obs.image, "is_first": env_obs.is_first}


class _PrepareOnlyAdapter(ObsAdapter):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def prepare_env_step(self, env_obs: ObservationFrame, packer) -> PreparedObservation:
        self.calls += 1
        step_obs = {"image": env_obs.image, "is_first": True}
        return PreparedObservation(
            replay_obs=env_obs.image,
            encoder_obs=packer.from_step(step_obs),
            is_first=True,
        )

    def transform(self, env_obs: ObservationFrame):
        raise AssertionError("trainer should route through prepare_env_step")


def _tiny_cnn_cfg(tmp_path):
    return R2DreamerConfig(
        encoder_type="cnn",
        obs_shape=(3, 64, 64),
        num_actions=4,
        buffer_capacity=64,
        batch_size=1,
        seq_len=2,
        train_ratio=2,
        deter_size=32,
        hidden_size=16,
        stoch_classes=4,
        stoch_discrete=4,
        blocks=4,
        encoder_depth=4,
        encoder_kernel=3,
        encoder_mults=(1, 1),
        mlp_units=16,
        mlp_layers_reward=1,
        mlp_layers_cont=1,
        mlp_layers_actor=1,
        mlp_layers_critic=1,
        twohot_bins=21,
        imagination_horizon=2,
        horizon=20,
        lr=1e-3,
        warmup_steps=0,
        logdir=str(tmp_path),
    )


def _tree_any_changed(before, after, *, atol=1e-7):
    return any(
        not np.allclose(np.asarray(a), np.asarray(b), atol=atol)
        for a, b in zip(before, jax.tree.leaves(after))
    )


class TestConvertBatch:
    """convert_batch turns replay buffer output into agent-ready batches."""

    @pytest.fixture
    def replay_batch(self):
        B, T, A = 4, 8, 6
        return {
            "obs": jnp.ones((B, T, 3, 4, 4)),
            "actions": jnp.array(np.random.randint(0, A, (B, T)), dtype=jnp.int32),
            "rewards": jnp.ones((B, T)),
            "is_episode_end": jnp.zeros((B, T)),
            "is_first": jnp.zeros((B, T)),
        }

    def test_actions_become_onehot(self, replay_batch):
        replay_batch["actions"] = jnp.array([[1, 2, 3, 4, 5, 0, 1, 2]] * 4)
        out = convert_batch(replay_batch, num_actions=6)
        assert out["actions"].shape == (4, 8, 6)
        assert out["actions"].dtype == jnp.float32
        assert jnp.allclose(out["actions"][:, 0].sum(axis=-1), 0.0)
        assert jnp.allclose(out["actions"][:, 1:].sum(axis=-1), 1.0)
        np.testing.assert_allclose(np.asarray(out["actions"][0, 1]), np.eye(6)[1])

    def test_episode_end_is_shifted_to_training_alignment(self, replay_batch):
        replay_batch["is_episode_end"] = jnp.ones((4, 8))
        out = convert_batch(replay_batch, num_actions=6)
        assert "is_episode_end" in out
        assert jnp.allclose(out["is_episode_end"][:, 0], 0.0)
        assert jnp.allclose(out["is_episode_end"][:, 1:], 1.0)

    def test_obs_and_rewards_pass_through(self, replay_batch):
        out = convert_batch(replay_batch, num_actions=6)
        assert jnp.allclose(out["obs"], replay_batch["obs"])
        assert jnp.allclose(out["rewards"][:, 0], 0.0)
        assert jnp.allclose(out["rewards"][:, 1:], replay_batch["rewards"][:, :-1])
        assert jnp.allclose(out["is_first"], replay_batch["is_first"])

    def test_output_keys(self, replay_batch):
        out = convert_batch(replay_batch, num_actions=6)
        assert set(out.keys()) == {
            "obs",
            "actions",
            "rewards",
            "is_first",
            "is_episode_end",
        }


class TestCheckpoint:
    """save_checkpoint and load_checkpoint round-trip agent state."""

    @pytest.fixture
    def agent(self):
        cfg = R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=4)
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
        cfg = R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=4)
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(42))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=10, output_dir=tmpdir)
            data = load_checkpoint(path)

        snapshot = data["encoder_input_contract"]
        assert snapshot["encoder_module"] == "src.r2dreamer.world_model.encoders.ConvEncoder"
        assert snapshot["encoder_module_kwargs"] == {
            "depth": 16,
            "kernel_size": 5,
            "mults": (2, 3, 4, 4),
        }
        json.dumps(snapshot)

    def test_agent_from_checkpoint_recovers_encoder_contract_when_shape_omitted(self):
        cfg = R2DreamerConfig(
            obs_shape=(3, 64, 64),
            num_actions=4,
            encoder_input_contract=CNNObservationPreparation().contract.to_snapshot(),
        )
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(42))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=10, output_dir=tmpdir)
            recovered = R2DreamerAgent.from_checkpoint(
                path, num_actions=4, seed=0,
            )

        assert recovered.cfg.obs_shape == (3, 64, 64)
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
            obs_shape=(3, 64, 64),
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
            obs_shape=(3, 64, 64),
            num_actions=4,
            encoder_module_cls=CNNObservationPreparation().contract.encoder_module_cls,
            encoder_input_contract=CNNObservationPreparation().contract.to_snapshot(),
        )

        snapshot = config_snapshot(cfg)

        assert snapshot["encoder_module"] == "src.r2dreamer.world_model.encoders.ConvEncoder"
        assert "encoder_module_cls" not in snapshot
        assert snapshot["encoder_input_contract"]["encoder_type"] == "cnn"
        assert snapshot["encoder_input_contract"]["encoder_module_kwargs"] == {
            "depth": 16,
            "kernel_size": 5,
            "mults": (2, 3, 4, 4),
        }
        json.dumps(snapshot)

    def test_config_snapshot_derives_default_cnn_contract(self):
        snapshot = config_snapshot(R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=4))

        assert snapshot["encoder_input_contract"]["encoder_type"] == "cnn"
        json.dumps(snapshot)


class TestResume:
    """Trainer with resume_from restores agent state and offsets the step counter."""

    @pytest.fixture
    def cfg(self):
        return R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=4)

    @pytest.fixture
    def saved_agent(self, cfg, tmp_path):
        """Build an agent, save its checkpoint, return (agent, ckpt_path, step)."""
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))
        step = 12345
        ckpt_path = save_checkpoint(agent, step=step, output_dir=str(tmp_path))
        return agent, ckpt_path, step

    def test_resume_restores_params_and_step(self, cfg, saved_agent, tmp_path):
        original, ckpt_path, step = saved_agent

        # Build a fresh agent with a different init seed so its weights differ.
        fresh = R2DreamerAgent(cfg, jax.random.PRNGKey(99))
        before = [np.asarray(x) for x in jax.tree.leaves(fresh.params)]
        target = [np.asarray(x) for x in jax.tree.leaves(original.params)]
        assert not all(np.allclose(a, b) for a, b in zip(before, target))

        tcfg = TrainerConfig(
            output_dir=str(tmp_path / "logdir"),
            total_steps=step + 1,
            wandb_project=None,
            resume_from=ckpt_path,
        )
        trainer = Trainer(
            agent=fresh, env=_DummyEnv(), agent_config=cfg, trainer_config=tcfg,
        )

        assert trainer._resume_step == step
        after_params = [np.asarray(x) for x in jax.tree.leaves(fresh.params)]
        for a, b in zip(after_params, target):
            np.testing.assert_allclose(a, b, atol=1e-6)
        for a, b in zip(jax.tree.leaves(fresh.slow_critic_params),
                        jax.tree.leaves(original.slow_critic_params)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(fresh.ema_state), np.asarray(original.ema_state), atol=1e-6,
        )

    def test_no_resume_keeps_step_zero(self, cfg, tmp_path):
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(7))
        tcfg = TrainerConfig(
            output_dir=str(tmp_path / "logdir"),
            total_steps=1,
            wandb_project=None,
        )
        trainer = Trainer(
            agent=agent, env=_DummyEnv(), agent_config=cfg, trainer_config=tcfg,
        )
        assert trainer._resume_step == 0

    def test_missing_resume_path_raises(self, cfg, tmp_path):
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(7))
        tcfg = TrainerConfig(
            output_dir=str(tmp_path / "logdir"),
            total_steps=1,
            wandb_project=None,
            resume_from=str(tmp_path / "nope.pkl"),
        )
        with pytest.raises(FileNotFoundError):
            Trainer(
                agent=agent, env=_DummyEnv(), agent_config=cfg, trainer_config=tcfg,
            )


class TestTrainerObservationPreparation:
    def test_reset_train_episode_uses_prepare_env_step_when_available(self, tmp_path):
        cfg = R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=4, buffer_capacity=8)
        tcfg = TrainerConfig(
            output_dir=str(tmp_path / "logdir"),
            total_steps=1,
            wandb_project=None,
        )
        obs_adapter = _PrepareOnlyAdapter()
        trainer = Trainer(
            agent=object(),
            env=_DummyEnv(),
            agent_config=cfg,
            trainer_config=tcfg,
            obs_adapter=obs_adapter,
        )

        _, buffer_obs, encoder_obs, is_first = trainer._reset_train_episode()

        assert obs_adapter.calls == 1
        assert buffer_obs.shape == (3, 64, 64)
        assert encoder_obs.shape == (1, 3, 64, 64)
        assert is_first is True


class TestTrainerFullPipeline:
    def test_cnn_observation_preparation_runs_through_training_pipeline(self, tmp_path):
        cfg = _tiny_cnn_cfg(tmp_path)
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))
        env = _TinyCNNEnv()
        trainer = Trainer(
            agent=agent,
            env=env,
            agent_config=cfg,
            trainer_config=TrainerConfig(
                output_dir=str(tmp_path / "run"),
                total_steps=4,
                prefill_steps=4,
                log_every=1,
                checkpoint_every=100,
                wandb_project=None,
                val_every=0,
            ),
            obs_adapter=CNNObservationPreparation(),
        )
        before = [np.asarray(x).copy() for x in jax.tree.leaves(agent.params)]

        trainer.run()

        assert env.closed is True
        assert trainer.buffer.size > 0
        assert _tree_any_changed(before, agent.params)


class TestTrainerMappingReplay:
    def test_trainer_builds_and_records_mapping_obs_buffer(self, tmp_path):
        cfg = R2DreamerConfig(obs_shape=(16404,), num_actions=4, buffer_capacity=8)
        tcfg = TrainerConfig(
            output_dir=str(tmp_path / "logdir"),
            total_steps=1,
            wandb_project=None,
        )
        trainer = Trainer(
            agent=object(),
            env=_DummyEnv(),
            agent_config=cfg,
            trainer_config=tcfg,
            obs_adapter=_MappingObsAdapter(),
        )

        buffer_obs, _ = trainer.obs_adapter.transform(_DummyEnv().reset())
        trainer._record_train_transition(
            buffer_obs=buffer_obs,
            action=1,
            next_obs=ObservationFrame(
                image=np.zeros((3, 64, 64), dtype=np.uint8),
                is_first=False,
                previous_action=1,
                reward=1.0,
            ),
        )

        assert trainer.buffer.size == 1
        batch = trainer.buffer.sample(batch_size=1, seq_len=1)
        assert set(batch["obs"]) == {"image", "wp_cp"}
        assert batch["obs"]["image"].shape == (1, 1, 3, 64, 64)
        assert batch["obs"]["image"].dtype == jnp.uint8
        assert batch["obs"]["wp_cp"].shape == (1, 1, 4116)
        assert batch["obs"]["wp_cp"].dtype == jnp.float32
