
"""Tests for src/r2dreamer/trainer.py — replay_batch_to_arrays and checkpoint."""

import json
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.buffer.replay_buffer import ReplayTransition
from src.environments.observation import ObservationFrame
from src.configs.config import R2DreamerConfig, TrainerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.adapters import ObsAdapter
from src.r2dreamer.observation_preparation import (
    CNNObservationPreparation,
    PreparedObservation,
)
from src.r2dreamer.trainer import (
    Trainer,
    config_snapshot,
    load_checkpoint,
    replay_batch_to_arrays,
    save_checkpoint,
)


def test_trainer_config_defaults_to_scalars_only_no_validation_or_video():
    cfg = TrainerConfig(output_dir="/tmp/r2dreamer-test")

    assert cfg.total_steps == 10_000_000
    assert cfg.seed == 0
    assert cfg.val_every == 0
    assert cfg.video_log_every == 0
    assert cfg.val_video_episodes == 0
    assert cfg.video_log_episodes == 0


class _DummyEnv:
    """Minimal env stub — Trainer.__init__ does not call any of these methods."""

    def reset(self) -> ObservationFrame:
        return ObservationFrame(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            is_first=True,
        )

    def step(self, action: int) -> ObservationFrame:
        return ObservationFrame(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
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
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            is_first=True,
        )

    def step(self, action: int) -> ObservationFrame:
        self.t += 1
        done = self.t >= 4
        return ObservationFrame(
            image=np.full((64, 64, 3), self.t, dtype=np.uint8),
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
            buffer_shape={"image": (64, 64, 3), "wp_cp": (4116,)},
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

    def prepare_env_step(self, env_obs: ObservationFrame) -> PreparedObservation:
        self.calls += 1
        return PreparedObservation(
            replay_obs=env_obs.image,
            encoder_obs=env_obs.image[None],
            is_first=True,
        )

    def transform(self, env_obs: ObservationFrame):
        raise AssertionError("trainer should route through prepare_env_step")


def _tiny_cnn_cfg(tmp_path):
    return R2DreamerConfig(
        encoder_type="cnn",
        obs_shape=(64, 64, 3),
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


class TestReplayBatchToArrays:
    """Replay transition windows become raw training-aligned arrays."""

    def test_marks_is_first_after_episode_end(self):
        batch = replay_batch_to_arrays(
            [
                [
                    ReplayTransition(
                        obs=np.array([0.0], dtype=np.float32),
                        action=0,
                        reward=0.0,
                        is_first=False,
                        is_episode_end=False,
                    ),
                    ReplayTransition(
                        obs=np.array([1.0], dtype=np.float32),
                        action=1,
                        reward=1.0,
                        is_first=False,
                        is_episode_end=True,
                    ),
                    ReplayTransition(
                        obs=np.array([2.0], dtype=np.float32),
                        action=2,
                        reward=2.0,
                        is_first=False,
                        is_episode_end=False,
                    ),
                ]
            ]
        )

        np.testing.assert_array_equal(
            np.asarray(batch["is_first"]), np.array([[True, False, True]])
        )


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


class TestResume:
    """Trainer with resume_from restores agent state and offsets the step counter."""

    @pytest.fixture
    def cfg(self):
        return R2DreamerConfig(obs_shape=(64, 64, 3), num_actions=4)

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
        cfg = R2DreamerConfig(obs_shape=(64, 64, 3), num_actions=4, buffer_capacity=8)
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
        assert buffer_obs.shape == (64, 64, 3)
        assert encoder_obs.shape == (1, 64, 64, 3)
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
                image=np.zeros((64, 64, 3), dtype=np.uint8),
                is_first=False,
                previous_action=1,
                reward=1.0,
            ),
        )

        assert trainer.buffer.size == 1
        batch = replay_batch_to_arrays(trainer.buffer.sample(batch_size=1, seq_len=1))
        obs_batch = batch["obs"]
        assert isinstance(obs_batch, dict)
        assert set(obs_batch) == {"image", "wp_cp"}
        assert obs_batch["image"].shape == (1, 1, 64, 64, 3)
        assert obs_batch["image"].dtype == jnp.uint8
        assert obs_batch["wp_cp"].shape == (1, 1, 4116)
        assert obs_batch["wp_cp"].dtype == jnp.float32


class _AgentSpy:
    """Records ``act``/``train_step`` calls on a real agent, delegating to it.

    Wraps rather than replaces, so the genuine ``R2DreamerAgent`` stays under
    test and its real signatures are exercised — only the call record is added.
    ``_train_loop`` asserts on *interactions* (how many updates, with which
    ``materialize`` flag), which the agent cannot report on its own.

    Attributes:
        actions: Every action the real policy returned, in call order.
        materialize_flags: The ``materialize`` value seen by each train_step.
    """

    def __init__(self, agent: R2DreamerAgent):
        self.actions: list[int] = []
        self.materialize_flags: list[bool] = []
        self._real_act = agent.act
        self._real_train_step = agent.train_step
        agent.act = self._act
        agent.train_step = self._train_step

    def _act(self, encoder_obs, is_first, rng_key, training: bool = True) -> int:
        action = self._real_act(encoder_obs, is_first, rng_key, training=training)
        self.actions.append(int(action))
        return action

    def _train_step(self, batch, rng_key, **kwargs) -> dict:
        # Mirror the real keyword-only default (agent.train_step: materialize=True).
        self.materialize_flags.append(bool(kwargs.get("materialize", True)))
        return self._real_train_step(batch, rng_key, **kwargs)

    @property
    def act_calls(self) -> int:
        return len(self.actions)

    @property
    def train_steps(self) -> int:
        return len(self.materialize_flags)


class _ScriptedEnv:
    """Env whose episodes end every ``done_every`` steps (never, if None)."""

    def __init__(self, done_every: int | None = None):
        self._done_every = done_every
        self.t = 0
        self.reset_calls = 0
        self.closed = False

    def reset(self) -> ObservationFrame:
        self.reset_calls += 1
        self.t = 0
        return ObservationFrame(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            is_first=True,
        )

    def step(self, action: int) -> ObservationFrame:
        self.t += 1
        done = self._done_every is not None and self.t % self._done_every == 0
        return ObservationFrame(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            is_first=False,
            previous_action=int(action),
            reward=1.0,
            done=done,
        )

    def close(self) -> None:
        self.closed = True


class _RecordingWriter:
    """csv.writer stand-in that keeps the rows in memory."""

    def __init__(self):
        self.rows: list[list] = []

    def writerow(self, row) -> None:
        self.rows.append(list(row))


class _NullFile:
    def flush(self) -> None:
        pass


@pytest.fixture
def build_agent(tmp_path):
    """Factory for a real R2DreamerAgent (tiny CNN) plus a call-recording spy.

    Args:
        tmp_path: pytest tmp_path, used for the config's logdir.

    Returns:
        A callable taking agent-config overrides (``batch_size``, ``seq_len``,
        ``train_ratio``) and returning ``(cfg, agent, spy)``.
    """

    def _build(batch_size: int = 1, seq_len: int = 2, train_ratio: int = 2):
        cfg = _tiny_cnn_cfg(tmp_path)
        cfg.batch_size = batch_size
        cfg.seq_len = seq_len
        cfg.train_ratio = train_ratio
        agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))
        return cfg, agent, _AgentSpy(agent)

    return _build


def _loop_trainer(
    tmp_path,
    cfg,
    agent,
    *,
    env=None,
    episode_metrics_fn=None,
    val_env=None,
    **tcfg_kwargs,
) -> Trainer:
    """Build a Trainer wired for driving ``_train_loop`` directly.

    Args:
        tmp_path: pytest tmp_path for the run's output dir.
        cfg: Agent config (supplies batch_size / seq_len / train_ratio).
        agent: The real agent under test.
        env: Env stub; defaults to a never-ending ``_ScriptedEnv``. Habitat is
            Linux-only, so the env is the one collaborator that must be scripted.
        episode_metrics_fn: Optional episode-end metrics callback.
        val_env: Optional val env; enables the validation branch.
        **tcfg_kwargs: Forwarded to TrainerConfig.

    Returns:
        A Trainer with W&B disabled and the default ObsAdapter.
    """
    tcfg = TrainerConfig(
        output_dir=str(tmp_path / "run"),
        wandb_project=None,
        **tcfg_kwargs,
    )
    return Trainer(
        agent=agent,
        env=env if env is not None else _ScriptedEnv(),
        agent_config=cfg,
        trainer_config=tcfg,
        episode_metrics_fn=episode_metrics_fn,
        val_env=val_env,
    )


def _run_loop(trainer: Trainer) -> _RecordingWriter:
    writer = _RecordingWriter()
    trainer._train_loop(jax.random.PRNGKey(0), writer, _NullFile())
    return writer


class TestTrainLoopTrainCredit:
    """train_credit accounting: train_ratio / (batch_size * seq_len) per step."""

    def test_trains_once_per_env_step_when_credit_rate_is_one(
        self, tmp_path, build_agent
    ):
        # batch_steps = 1 * 2 = 2; train_ratio 2 => +1.0 credit per env step.
        cfg, agent, spy = build_agent(train_ratio=2)
        trainer = _loop_trainer(tmp_path, cfg, agent, total_steps=6)

        _run_loop(trainer)

        # Step 0 leaves buffer.size == 1 < 2, so the gate opens at step 1.
        assert spy.train_steps == 5
        assert spy.act_calls == 6

    def test_train_ratio_scales_updates_per_env_step(self, tmp_path, build_agent):
        # train_ratio 4 over batch_steps 2 => +2.0 credit per env step.
        cfg, agent, spy = build_agent(train_ratio=4)
        trainer = _loop_trainer(tmp_path, cfg, agent, total_steps=6)

        _run_loop(trainer)

        assert spy.train_steps == 10

    def test_fractional_credit_accumulates_across_steps(self, tmp_path, build_agent):
        # train_ratio 1 over batch_steps 2 => +0.5 per step: train every 2nd.
        cfg, agent, spy = build_agent(train_ratio=1)
        trainer = _loop_trainer(tmp_path, cfg, agent, total_steps=6)

        _run_loop(trainer)

        assert spy.train_steps == 2

    def test_no_training_until_buffer_holds_one_batch(self, tmp_path, build_agent):
        # batch_steps = 1 * 8 = 8 > total_steps, so the gate never opens.
        cfg, agent, spy = build_agent(seq_len=8, train_ratio=64)
        trainer = _loop_trainer(tmp_path, cfg, agent, total_steps=4)

        _run_loop(trainer)

        assert spy.train_steps == 0
        assert spy.act_calls == 4


class TestTrainLoopCadences:
    """Checkpoint / logging / validation fire on their configured cadence."""

    def test_checkpoints_on_cadence_using_one_based_step(
        self, tmp_path, build_agent, monkeypatch
    ):
        saved: list[int] = []
        monkeypatch.setattr(
            "src.r2dreamer.trainer.save_checkpoint",
            lambda agent, step, output_dir: saved.append(step),
        )
        cfg, agent, _ = build_agent()
        trainer = _loop_trainer(tmp_path, cfg, agent, total_steps=6, checkpoint_every=2)

        _run_loop(trainer)

        assert saved == [2, 4, 6]

    def test_materialize_is_true_only_on_log_steps(self, tmp_path, build_agent):
        cfg, agent, spy = build_agent(train_ratio=2)
        trainer = _loop_trainer(tmp_path, cfg, agent, total_steps=6, log_every=2)

        _run_loop(trainer)

        # Gate opens at step 1, one update per step for steps 1..5;
        # will_log = step % 2 == 0 => True at steps 2 and 4.
        assert spy.materialize_flags == [False, True, False, True, False]

    def test_val_loop_runs_on_cadence_when_val_env_present(self, tmp_path, build_agent):
        called: list[int] = []
        cfg, agent, _ = build_agent()
        trainer = _loop_trainer(
            tmp_path, cfg, agent, total_steps=4, val_every=2, val_env=_ScriptedEnv()
        )
        trainer._run_val_loop = lambda key, step, writer, f: called.append(step)

        _run_loop(trainer)

        # (step + 1) % val_every == 0 => steps 1 and 3.
        assert called == [1, 3]

    def test_val_loop_skipped_when_no_val_env(self, tmp_path, build_agent):
        called: list[int] = []
        cfg, agent, _ = build_agent()
        trainer = _loop_trainer(
            tmp_path, cfg, agent, total_steps=4, val_every=2, val_env=None
        )
        trainer._run_val_loop = lambda key, step, writer, f: called.append(step)

        _run_loop(trainer)

        assert not called


class TestTrainLoopEpisodeHandoff:
    """Episode-end resets the env and the per-episode accumulators."""

    def test_episode_end_resets_env_and_reward_accumulator(self, tmp_path, build_agent):
        cfg, agent, _ = build_agent()
        env = _ScriptedEnv(done_every=2)
        trainer = _loop_trainer(tmp_path, cfg, agent, env=env, total_steps=4)

        writer = _run_loop(trainer)

        # One reset at loop entry plus one per finished episode.
        assert env.reset_calls == 3
        rewards = [row[2] for row in writer.rows if row[1] == "episode/reward"]
        # reward=1.0 per step over 2-step episodes: the accumulator must reset.
        assert rewards == [2.0, 2.0]

    def test_action_counts_are_per_episode_and_indexed_by_action(
        self, tmp_path, build_agent
    ):
        seen: list[np.ndarray] = []

        def metrics_fn(last_obs, episode_reward, episode_steps, action_counts):
            seen.append(action_counts.copy())
            return {"episode/reward": episode_reward}

        cfg, agent, spy = build_agent()
        env = _ScriptedEnv(done_every=3)
        trainer = _loop_trainer(
            tmp_path,
            cfg,
            agent,
            env=env,
            total_steps=6,
            episode_metrics_fn=metrics_fn,
        )

        _run_loop(trainer)

        assert len(seen) == 2
        # The real policy picks the actions; the spy records what it picked, so
        # each episode's counts must be the histogram of that episode's actions.
        for episode, counts in enumerate(seen):
            actions = spy.actions[episode * 3 : (episode + 1) * 3]
            expected = np.bincount(actions, minlength=cfg.num_actions)
            assert counts.tolist() == expected.tolist()
            assert counts.sum() == 3

    def test_episode_steps_reset_between_episodes(self, tmp_path, build_agent):
        seen: list[int] = []

        def metrics_fn(last_obs, episode_reward, episode_steps, action_counts):
            seen.append(episode_steps)
            return {"episode/reward": episode_reward}

        cfg, agent, _ = build_agent()
        env = _ScriptedEnv(done_every=3)
        trainer = _loop_trainer(
            tmp_path, cfg, agent, env=env, total_steps=6, episode_metrics_fn=metrics_fn
        )

        _run_loop(trainer)

        assert seen == [3, 3]
