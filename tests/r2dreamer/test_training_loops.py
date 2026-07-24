"""Tests for src/r2dreamer/launch/loops.py — the training loop functions.

Ports the old Trainer tests (ADR 0006): the loops are plain functions taking
protocol-typed collaborators, so they are driven directly with a scripted env
inside a real ``ExperienceCollector``, a real tiny agent, and a fake logger.
"""

import jax
import numpy as np
import pytest

from src.buffer.replay_arrays import replay_batch_to_arrays
from src.buffer.replay_buffer import ReplayBuffer
from src.configs.config import R2DreamerConfig, TrainerConfig
from src.environments.observation import ObservationFrame
from src.r2dreamer.composition import make_learner
from src.r2dreamer.learner import R2DLearner
from src.r2dreamer.adapters import ObsAdapter
from src.r2dreamer.checkpointing import save_checkpoint
from src.r2dreamer.experience import ExperienceCollector
from src.r2dreamer.launch.loops import apply_resume, run_training, train_loop
from src.r2dreamer.observation_preparation import (
    CNNObservationPreparation,
    PreparedObservation,
)
from src.shared.dtypes import compute_jnp_dtype


def test_trainer_config_defaults_to_scalars_only_no_validation_or_video():
    cfg = TrainerConfig(output_dir="/tmp/r2dreamer-test")

    assert cfg.total_steps == 10_000_000
    assert cfg.seed == 0
    assert cfg.val_every == 0
    assert cfg.video_log_every == 0
    assert cfg.val_video_episodes == 0
    assert cfg.video_log_episodes == 0


class _TinyCNNEnv:
    """Small deterministic env for a full-pipeline smoke test."""

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
        raise AssertionError("collector should route through prepare_env_step")


class _FakeLogger:
    """RunLoggerLike stand-in recording metric rows in memory."""

    def __init__(self):
        self.rows: list[list] = []
        self.videos: list[tuple[str, int]] = []
        self.wandb_active = False

    def start_timing(self, start_step: int) -> None:
        pass

    def log_episode(self, episode, step: int) -> None:
        for k, v in episode.metrics.items():
            self.rows.append([step, k, v])

    def log_video(self, key: str, frames, step: int) -> None:
        self.videos.append((key, step))

    def log_train_metrics(self, metrics, step: int) -> None:
        for k, v in metrics.items():
            self.rows.append([step, k, v])

    def log_reconstructions(self, target, recon, step: int) -> None:
        pass

    def log_val_metrics(self, metrics, step: int, elapsed: float) -> None:
        for k, v in metrics.items():
            self.rows.append([step, k, v])

    def write_row(self, step: int, key: str, value) -> None:
        self.rows.append([step, key, value])


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


def _make_collector(cfg, *, env=None, adapter=None, episode_metrics_fn=None):
    return ExperienceCollector(
        env=env if env is not None else _ScriptedEnv(),
        adapter=adapter if adapter is not None else ObsAdapter(),
        num_actions=cfg.num_actions,
        buffer=ReplayBuffer(
            capacity=cfg.buffer_capacity,
            num_actions=cfg.num_actions,
            float_dtype=compute_jnp_dtype(cfg.compute_dtype),
        ),
        episode_metrics_fn=episode_metrics_fn,
    )


class _AgentSpy:
    """Records ``act``/``train_step`` calls on a real agent, delegating to it.

    Wraps rather than replaces, so the genuine ``R2DLearner`` stays under
    test and its real signatures are exercised — only the call record is added.
    ``train_loop`` asserts on *interactions* (how many updates, with which
    ``materialize`` flag), which the agent cannot report on its own.

    Attributes:
        actions: Every action the real policy returned, in call order.
        materialize_flags: The ``materialize`` value seen by each train_step.
    """

    def __init__(self, agent: R2DLearner):
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


@pytest.fixture
def build_agent(tmp_path):
    """Factory for a real R2DLearner (tiny CNN) plus a call-recording spy.

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
        agent = make_learner(cfg, jax.random.PRNGKey(0))
        return cfg, agent, _AgentSpy(agent)

    return _build


def _run_train_loop(
    tmp_path,
    cfg,
    agent,
    *,
    env=None,
    episode_metrics_fn=None,
    val_experience=None,
    **tcfg_kwargs,
):
    """Drive ``train_loop`` directly with a scripted env and a fake logger.

    Args:
        tmp_path: pytest tmp_path for the run's output dir.
        cfg: Agent config (supplies batch_size / seq_len / train_ratio).
        agent: The real agent under test.
        env: Env stub; defaults to a never-ending ``_ScriptedEnv``. Habitat is
            Linux-only, so the env is the one collaborator that must be scripted.
        episode_metrics_fn: Optional episode-end metrics callback.
        val_experience: Optional val collector; enables the validation branch.
        **tcfg_kwargs: Forwarded to TrainerConfig.

    Returns:
        The fake logger (with recorded rows) and the train collector.
    """
    tcfg = TrainerConfig(
        output_dir=str(tmp_path / "run"),
        wandb_project=None,
        **tcfg_kwargs,
    )
    collector = _make_collector(cfg, env=env, episode_metrics_fn=episode_metrics_fn)
    logger = _FakeLogger()
    train_loop(
        agent,
        collector,
        cfg,
        tcfg,
        logger,
        jax.random.PRNGKey(0),
        val_experience=val_experience,
    )
    return logger, collector


def _val_collector(cfg):
    return ExperienceCollector(
        env=_ScriptedEnv(),
        adapter=ObsAdapter(),
        num_actions=cfg.num_actions,
        buffer=None,
        auto_reset=False,
    )


class TestApplyResume:
    """apply_resume restores agent state and returns the checkpoint step."""

    @pytest.fixture
    def cfg(self):
        return R2DreamerConfig(obs_shape=(64, 64, 3), num_actions=4)

    def test_resume_restores_params_and_step(self, cfg, tmp_path):
        original = make_learner(cfg, jax.random.PRNGKey(0))
        step = 12345
        ckpt_path = save_checkpoint(original, step=step, output_dir=str(tmp_path))

        # Build a fresh agent with a different init seed so its weights differ.
        fresh = make_learner(cfg, jax.random.PRNGKey(99))
        before = [np.asarray(x) for x in jax.tree.leaves(fresh.params)]
        target = [np.asarray(x) for x in jax.tree.leaves(original.params)]
        assert not all(np.allclose(a, b) for a, b in zip(before, target))

        resume_step = apply_resume(fresh, ckpt_path)

        assert resume_step == step
        after_params = [np.asarray(x) for x in jax.tree.leaves(fresh.params)]
        for a, b in zip(after_params, target):
            np.testing.assert_allclose(a, b, atol=1e-6)
        for a, b in zip(jax.tree.leaves(fresh.slow_critic_params),
                        jax.tree.leaves(original.slow_critic_params)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(fresh.ema_state), np.asarray(original.ema_state), atol=1e-6,
        )

    def test_missing_resume_path_raises(self, cfg, tmp_path):
        agent = make_learner(cfg, jax.random.PRNGKey(7))
        with pytest.raises(FileNotFoundError):
            apply_resume(agent, str(tmp_path / "nope.pkl"))


class TestCollectorWiring:
    def test_collector_routes_through_prepare_env_step(self, tmp_path):
        cfg = R2DreamerConfig(obs_shape=(64, 64, 3), num_actions=4, buffer_capacity=8)
        adapter = _PrepareOnlyAdapter()
        collector = _make_collector(cfg, adapter=adapter)

        agent_step = collector.reset()

        assert adapter.calls == 1
        assert agent_step.encoder_obs.shape == (1, 64, 64, 3)
        assert agent_step.is_first is True

    def test_collector_records_mapping_obs_through_replay_arrays(self, tmp_path):
        cfg = R2DreamerConfig(obs_shape=(16404,), num_actions=4, buffer_capacity=8)
        collector = _make_collector(cfg, adapter=_MappingObsAdapter())

        collector.reset()
        collector.step(1)

        assert collector.buffer_size == 1
        batch = replay_batch_to_arrays(
            collector.buffer.sample(batch_size=1, seq_len=1)
        )
        obs_batch = batch["obs"]
        assert isinstance(obs_batch, dict)
        assert set(obs_batch) == {"image", "wp_cp"}
        assert obs_batch["image"].shape == (1, 1, 64, 64, 3)
        assert obs_batch["image"].dtype == np.uint8
        assert obs_batch["wp_cp"].shape == (1, 1, 4116)
        assert obs_batch["wp_cp"].dtype == np.float32


class TestFullPipeline:
    def test_cnn_observation_preparation_runs_through_training_pipeline(self, tmp_path):
        cfg = _tiny_cnn_cfg(tmp_path)
        agent = make_learner(cfg, jax.random.PRNGKey(0))
        env = _TinyCNNEnv()
        collector = ExperienceCollector(
            env=env,
            adapter=CNNObservationPreparation(),
            num_actions=cfg.num_actions,
            buffer=ReplayBuffer(
                capacity=cfg.buffer_capacity,
                num_actions=cfg.num_actions,
                float_dtype=compute_jnp_dtype(cfg.compute_dtype),
            ),
        )
        tcfg = TrainerConfig(
            output_dir=str(tmp_path / "run"),
            total_steps=4,
            prefill_steps=4,
            log_every=1,
            checkpoint_every=100,
            wandb_project=None,
            val_every=0,
        )
        before = [np.asarray(x).copy() for x in jax.tree.leaves(agent.params)]

        run_training(agent, collector, cfg, tcfg)

        assert env.closed is True
        assert collector.buffer_size > 0
        assert _tree_any_changed(before, agent.params)
        assert (tmp_path / "run" / "metrics.csv").exists()
        assert (tmp_path / "run" / "MANIFEST.json").exists()


class TestTrainLoopTrainCredit:
    """train_credit accounting: train_ratio / (batch_size * seq_len) per step."""

    def test_trains_once_per_env_step_when_credit_rate_is_one(
        self, tmp_path, build_agent
    ):
        # batch_steps = 1 * 2 = 2; train_ratio 2 => +1.0 credit per env step.
        cfg, agent, spy = build_agent(train_ratio=2)

        _run_train_loop(tmp_path, cfg, agent, total_steps=6)

        # Step 0 leaves buffer.size == 1 < 2, so the gate opens at step 1.
        assert spy.train_steps == 5
        assert spy.act_calls == 6

    def test_train_ratio_scales_updates_per_env_step(self, tmp_path, build_agent):
        # train_ratio 4 over batch_steps 2 => +2.0 credit per env step.
        cfg, agent, spy = build_agent(train_ratio=4)

        _run_train_loop(tmp_path, cfg, agent, total_steps=6)

        assert spy.train_steps == 10

    def test_fractional_credit_accumulates_across_steps(self, tmp_path, build_agent):
        # train_ratio 1 over batch_steps 2 => +0.5 per step: train every 2nd.
        cfg, agent, spy = build_agent(train_ratio=1)

        _run_train_loop(tmp_path, cfg, agent, total_steps=6)

        assert spy.train_steps == 2

    def test_no_training_until_buffer_holds_one_batch(self, tmp_path, build_agent):
        # batch_steps = 1 * 8 = 8 > total_steps, so the gate never opens.
        cfg, agent, spy = build_agent(seq_len=8, train_ratio=64)

        _run_train_loop(tmp_path, cfg, agent, total_steps=4)

        assert spy.train_steps == 0
        assert spy.act_calls == 4


class TestTrainLoopCadences:
    """Checkpoint / logging / validation fire on their configured cadence."""

    def test_checkpoints_on_cadence_using_one_based_step(
        self, tmp_path, build_agent, monkeypatch
    ):
        saved: list[int] = []
        monkeypatch.setattr(
            "src.r2dreamer.launch.loops.save_checkpoint",
            lambda agent, step, output_dir: saved.append(step),
        )
        cfg, agent, _ = build_agent()

        _run_train_loop(tmp_path, cfg, agent, total_steps=6, checkpoint_every=2)

        assert saved == [2, 4, 6]

    def test_materialize_is_true_only_on_log_steps(self, tmp_path, build_agent):
        cfg, agent, spy = build_agent(train_ratio=2)

        _run_train_loop(tmp_path, cfg, agent, total_steps=6, log_every=2)

        # Gate opens at step 1, one update per step for steps 1..5;
        # will_log = step % 2 == 0 => True at steps 2 and 4.
        assert spy.materialize_flags == [False, True, False, True, False]

    def test_val_loop_runs_on_cadence_when_val_experience_present(
        self, tmp_path, build_agent, monkeypatch
    ):
        called: list[int] = []
        monkeypatch.setattr(
            "src.r2dreamer.launch.loops.val_loop",
            lambda agent, val_exp, tcfg, logger, key, step: called.append(step),
        )
        cfg, agent, _ = build_agent()

        _run_train_loop(
            tmp_path,
            cfg,
            agent,
            total_steps=4,
            val_every=2,
            val_experience=_val_collector(cfg),
        )

        # (step + 1) % val_every == 0 => steps 1 and 3.
        assert called == [1, 3]

    def test_val_loop_skipped_when_no_val_experience(
        self, tmp_path, build_agent, monkeypatch
    ):
        called: list[int] = []
        monkeypatch.setattr(
            "src.r2dreamer.launch.loops.val_loop",
            lambda agent, val_exp, tcfg, logger, key, step: called.append(step),
        )
        cfg, agent, _ = build_agent()

        _run_train_loop(
            tmp_path, cfg, agent, total_steps=4, val_every=2, val_experience=None
        )

        assert not called


class TestTrainLoopEpisodeHandoff:
    """Episode-end resets the env and the per-episode accumulators."""

    def test_episode_end_resets_env_and_reward_accumulator(self, tmp_path, build_agent):
        cfg, agent, _ = build_agent()
        env = _ScriptedEnv(done_every=2)

        logger, _ = _run_train_loop(tmp_path, cfg, agent, env=env, total_steps=4)

        # One reset at loop entry plus one per finished episode.
        assert env.reset_calls == 3
        rewards = [row[2] for row in logger.rows if row[1] == "episode/reward"]
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

        _run_train_loop(
            tmp_path,
            cfg,
            agent,
            env=env,
            total_steps=6,
            episode_metrics_fn=metrics_fn,
        )

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

        _run_train_loop(
            tmp_path, cfg, agent, env=env, total_steps=6, episode_metrics_fn=metrics_fn
        )

        assert seen == [3, 3]
