"""Tests for src/r2dreamer/trainer.py — convert_batch and checkpoint."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.adapters import ObsAdapter
from src.r2dreamer.trainer import (
    Trainer,
    TrainerConfig,
    convert_batch,
    load_checkpoint,
    save_checkpoint,
)


class _DummyEnv:
    """Minimal env stub — Trainer.__init__ does not call any of these methods."""

    def reset(self) -> dict:
        return {"image": np.zeros((3, 64, 64), dtype=np.uint8), "is_first": True}

    def step(self, action: int) -> dict:
        return {
            "image": np.zeros((3, 64, 64), dtype=np.uint8),
            "reward": 0.0,
            "done": False,
            "success": 0.0,
        }

    def close(self) -> None:
        pass


class _MappingObsAdapter(ObsAdapter):
    def __init__(self):
        super().__init__(
            buffer_dtype={"image": "uint8", "wp_cp": "float32"},
            buffer_shape={"image": (3, 64, 64), "wp_cp": (4116,)},
            normalize_on_sample={"image": False, "wp_cp": False},
            agent_obs_shape=(16404,),
        )

    def transform(self, obs_dict: dict) -> tuple[dict[str, np.ndarray], dict]:
        return {
            "image": obs_dict["image"],
            "wp_cp": np.ones((4116,), dtype=np.float32),
        }, obs_dict


def test_trainer_config_has_no_in_run_val_or_video_controls():
    tcfg = TrainerConfig()

    assert not hasattr(tcfg, "val_every")
    assert not hasattr(tcfg, "val_episodes")
    assert not hasattr(tcfg, "val_video_episodes")
    assert not hasattr(tcfg, "val_max_episode_steps")
    assert not hasattr(tcfg, "video_log_every")
    assert not hasattr(tcfg, "video_log_episodes")


class TestConvertBatch:
    """convert_batch turns replay buffer output into agent-ready batches."""

    @pytest.fixture
    def replay_batch(self):
        B, T, A = 4, 8, 6
        return {
            "obs": jnp.ones((B, T, 3, 4, 4)),
            "actions": jnp.array(np.random.randint(0, A, (B, T)), dtype=jnp.int32),
            "rewards": jnp.ones((B, T)),
            "dones": jnp.zeros((B, T)),
            "terminals": jnp.zeros((B, T)),
            "is_first": jnp.zeros((B, T)),
        }

    def test_actions_become_onehot(self, replay_batch):
        out = convert_batch(replay_batch, num_actions=6)
        assert out["actions"].shape == (4, 8, 6)
        assert out["actions"].dtype == jnp.float32
        # Each row should sum to 1 (one-hot)
        assert jnp.allclose(out["actions"].sum(axis=-1), 1.0)

    def test_dones_renamed_to_is_last(self, replay_batch):
        replay_batch["dones"] = jnp.ones((4, 8))
        out = convert_batch(replay_batch, num_actions=6)
        assert "is_last" in out
        assert "dones" not in out
        assert jnp.allclose(out["is_last"], 1.0)

    def test_terminals_renamed_to_is_terminal(self, replay_batch):
        replay_batch["terminals"] = jnp.ones((4, 8))
        out = convert_batch(replay_batch, num_actions=6)
        assert "is_terminal" in out
        assert "terminals" not in out
        assert jnp.allclose(out["is_terminal"], 1.0)

    def test_obs_and_rewards_pass_through(self, replay_batch):
        out = convert_batch(replay_batch, num_actions=6)
        assert jnp.allclose(out["obs"], replay_batch["obs"])
        assert jnp.allclose(out["rewards"], replay_batch["rewards"])
        assert jnp.allclose(out["is_first"], replay_batch["is_first"])

    def test_output_keys(self, replay_batch):
        out = convert_batch(replay_batch, num_actions=6)
        assert set(out.keys()) == {"obs", "actions", "rewards", "is_first", "is_last", "is_terminal"}


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
            next_obs={"reward": 1.0, "done": False, "success": 0.0},
        )

        assert trainer.buffer.size == 1
        batch = trainer.buffer.sample(batch_size=1, seq_len=1)
        assert set(batch["obs"]) == {"image", "wp_cp"}
        assert batch["obs"]["image"].shape == (1, 1, 3, 64, 64)
        assert batch["obs"]["image"].dtype == jnp.uint8
        assert batch["obs"]["wp_cp"].shape == (1, 1, 4116)
        assert batch["obs"]["wp_cp"].dtype == jnp.float32
