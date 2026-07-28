"""Tests for src/r2dreamer/launch/loops.py — the training loop functions.

Ports the old Trainer tests (ADR 0006): the loops are plain functions taking
protocol-typed collaborators, so they are driven directly with a scripted env
inside a real ``ExperienceCollector``, a real tiny agent, and a fake logger.

The collector takes a plain adapter callable and the agent is built from that
adapter's routed fields, so these tests exercise the same composition
``src.main.train`` performs — only the env is scripted (Habitat is Linux-only and
slow to start).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.adapters.contract import AdapterField, AdapterOutput, Encoder
from src.adapters.rgb import RgbAdapter
from src.buffer.replay_arrays import replay_batch_to_arrays
from src.buffer.replay_buffer import ReplayBuffer
from src.configs.config import R2DreamerConfig, TrainerConfig
from src.environments.observation import ObservationFrame
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.checkpointing import save_checkpoint
from src.r2dreamer.experience import ExperienceCollector
from src.r2dreamer.launch.loops import apply_resume, run_training, train_loop
from src.shared.dtypes import compute_jnp_dtype


def test_trainer_config_defaults_to_scalars_only_no_validation_or_video():
    cfg = TrainerConfig(output_dir="/tmp/r2dreamer-test")

    assert cfg.total_steps == 10_000_000
    assert cfg.seed == 0
    assert cfg.val_every == 0
    assert cfg.video_log_every == 0
    assert cfg.val_video_episodes == 0
    assert cfg.video_log_episodes == 0


def _rgb_fields() -> AdapterOutput:
    """The routed fields of the RGB baseline, as one adapter call produces them."""
    return RgbAdapter()(
        ObservationFrame(image=np.zeros((64, 64, 3), dtype=np.uint8), is_first=True)
    )


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


class _MultiFieldAdapter:
    """Routes two fields, so replay stores a structure rather than one array."""

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        return [
            AdapterField(
                key="image",
                encoder=Encoder.CONV,
                buffer=True,
                value=jnp.asarray(frame.image),
                decoder_target=True,
            ),
            AdapterField(
                key="camera_pose",
                encoder=Encoder.MLP,
                buffer=True,
                value=jnp.ones((9,), dtype=jnp.float32),
            ),
        ]


class _CountingAdapter(RgbAdapter):
    """The RGB adapter with a call counter, to prove the collector routes frames."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        self.calls += 1
        return super().__call__(frame)


class _FakeLogger:
    """RunLoggerLike stand-in recording metric rows in memory."""

    def __init__(self):
        self.rows: list[list] = []
        self.train_metric_calls: list[tuple[int, dict]] = []
        self.videos: list[tuple[str, int]] = []
        self.adapter_summaries: list[tuple[dict, int]] = []
        self.wandb_active = False

    def start_timing(self, start_step: int) -> None:
        pass

    def log_episode(self, episode, step: int) -> None:
        for k, v in episode.metrics.items():
            self.rows.append([step, k, v])

    def log_video(self, key: str, frames, step: int) -> None:
        self.videos.append((key, step))

    def log_train_metrics(self, metrics, step: int) -> None:
        self.train_metric_calls.append((step, dict(metrics)))
        for k, v in metrics.items():
            self.rows.append([step, k, v])

    def log_reconstructions(self, target, recon, step: int) -> None:
        pass

    def log_val_metrics(self, metrics, step: int, elapsed: float) -> None:
        for k, v in metrics.items():
            self.rows.append([step, k, v])

    def log_adapter_summary(self, stats: dict, final_step: int) -> None:
        self.adapter_summaries.append((dict(stats), final_step))

    def write_row(self, step: int, key: str, value) -> None:
        self.rows.append([step, key, value])


def _tiny_cnn_cfg(tmp_path):
    return R2DreamerConfig(
        adapter="rgb",
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


def _tiny_agent(cfg, seed: int = 0) -> R2DreamerAgent:
    return R2DreamerAgent(cfg, jax.random.PRNGKey(seed), fields=_rgb_fields())


def _tree_any_changed(before, after, *, atol=1e-7):
    return any(
        not np.allclose(np.asarray(a), np.asarray(b), atol=atol)
        for a, b in zip(before, jax.tree.leaves(after))
    )


def _make_collector(cfg, *, env=None, observe=None, episode_metrics_fn=None):
    return ExperienceCollector(
        env=env if env is not None else _ScriptedEnv(),
        observe=observe if observe is not None else RgbAdapter(),
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

    Wraps rather than replaces, so the genuine ``R2DreamerAgent`` stays under
    test and its real signatures are exercised — only the call record is added.
    ``train_loop`` asserts on *interactions* (how many updates, with which
    ``materialize`` flag), which the agent cannot report on its own.

    Attributes:
        actions: Every action the real policy returned, in call order.
        materialize_flags: The ``materialize`` value seen by each train_step.
    """

    def __init__(self, agent: R2DreamerAgent):
        self.actions: list[int] = []
        self.materialize_flags: list[bool] = []
        # Bind the jitted descriptors once; the instance attributes below
        # shadow them, so the spy must keep its own handle on the real ones.
        self._real_act = agent.act
        self._real_train_step = agent.train_step
        agent.act = self._act
        agent.train_step = self._train_step

    def _act(self, params, obs, is_first, state, rng_key, training=True):
        action, next_state = self._real_act(
            params, obs, is_first, state, rng_key, training
        )
        self.actions.append(int(action))
        return action, next_state

    def _train_step(self, train_state, batch, rng_key, **kwargs):
        # Mirror the real keyword-only default (agent.train_step: materialize=True).
        self.materialize_flags.append(bool(kwargs.get("materialize", True)))
        return self._real_train_step(train_state, batch, rng_key, **kwargs)

    @property
    def act_calls(self) -> int:
        return len(self.actions)

    @property
    def train_steps(self) -> int:
        return len(self.materialize_flags)


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
        agent = _tiny_agent(cfg)
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
        observe=RgbAdapter(),
        num_actions=cfg.num_actions,
        buffer=None,
        auto_reset=False,
    )


class TestApplyResume:
    """apply_resume restores agent state and returns the checkpoint step."""

    @pytest.fixture
    def cfg(self, tmp_path):
        return _tiny_cnn_cfg(tmp_path)

    def test_resume_restores_params_and_step(self, cfg, tmp_path):
        original = _tiny_agent(cfg, seed=0)
        step = 12345
        ckpt_path = save_checkpoint(original, step=step, output_dir=str(tmp_path))

        # Build a fresh agent with a different init seed so its weights differ.
        fresh = _tiny_agent(cfg, seed=99)
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
        agent = _tiny_agent(cfg, seed=7)
        with pytest.raises(FileNotFoundError):
            apply_resume(agent, str(tmp_path / "nope.pkl"))


class TestCollectorWiring:
    def test_collector_routes_every_frame_through_the_adapter(self, tmp_path):
        cfg = _tiny_cnn_cfg(tmp_path)
        observe = _CountingAdapter()
        collector = _make_collector(cfg, observe=observe)

        agent_step = collector.reset()

        assert observe.calls == 1
        assert set(agent_step.encoder_obs) == {"image"}
        assert agent_step.encoder_obs["image"].shape == (64, 64, 3)
        assert agent_step.is_first is True

    def test_collector_records_multi_field_obs_through_replay_arrays(self, tmp_path):
        cfg = _tiny_cnn_cfg(tmp_path)
        collector = _make_collector(cfg, observe=_MultiFieldAdapter())

        collector.reset()
        collector.step(1)

        assert collector.buffer_size == 1
        batch = replay_batch_to_arrays(
            collector.buffer.sample(batch_size=1, seq_len=1)
        )
        obs_batch = batch["obs"]
        assert isinstance(obs_batch, dict)
        assert set(obs_batch) == {"image", "camera_pose"}
        assert obs_batch["image"].shape == (1, 1, 64, 64, 3)
        assert obs_batch["image"].dtype == np.uint8
        assert obs_batch["camera_pose"].shape == (1, 1, 9)
        assert obs_batch["camera_pose"].dtype == np.float32


class TestFullPipeline:
    def test_rgb_adapter_runs_through_the_training_pipeline(self, tmp_path):
        cfg = _tiny_cnn_cfg(tmp_path)
        agent = _tiny_agent(cfg)
        env = _TinyCNNEnv()
        collector = ExperienceCollector(
            env=env,
            observe=RgbAdapter(),
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

        logger, _ = _run_train_loop(tmp_path, cfg, agent, total_steps=6, log_every=2)

        # Gate opens at step 1, one update per step for steps 1..5; credit is
        # always >= 1.0 here, so the pending log fires on its own step:
        # steps 2 and 4.
        assert spy.materialize_flags == [False, True, False, True, False]
        assert [step for step, _ in logger.train_metric_calls] == [2, 4]

    def test_fractional_credit_still_logs_train_metrics(self, tmp_path, build_agent):
        # batch_steps = 1 * 9 = 9, train_ratio 3 => +1/3 credit per env step.
        # The gate opens at step 8, so updates land on steps 10, 13, 16, 19, 22
        # while log_every=6 marks steps 12 and 18: no update ever coincides
        # with a log step, which is the production parity mismatch.
        cfg, agent, spy = build_agent(seq_len=9, train_ratio=3)

        logger, _ = _run_train_loop(tmp_path, cfg, agent, total_steps=24, log_every=6)

        assert spy.train_steps == 5
        # The latched flag defers each pending log to the next real update.
        assert [step for step, _ in logger.train_metric_calls] == [13, 19]
        assert all(metrics for _, metrics in logger.train_metric_calls)
        # materialize is True exactly on the iterations that log.
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
