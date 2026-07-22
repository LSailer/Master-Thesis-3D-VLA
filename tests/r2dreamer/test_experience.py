"""Tests for src/r2dreamer/experience.py — the ExperienceCollector (ADR 0006)."""

import numpy as np
import pytest

from src.buffer.replay_buffer import ReplayBuffer
from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters import ObsAdapter
from src.r2dreamer.experience import AgentStep, ExperienceCollector

NUM_ACTIONS = 4


class _ScriptedEnv:
    """Env whose episodes end every ``done_every`` steps (never, if None)."""

    def __init__(self, done_every: int | None = None, scene_id: str = "scene-a"):
        self._done_every = done_every
        self.scene_id = scene_id
        self.t = 0
        self.reset_calls = 0
        self.closed = False

    def reset(self) -> ObservationFrame:
        self.reset_calls += 1
        self.t = 0
        return ObservationFrame(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            is_first=True,
            scene_id=self.scene_id,
        )

    def step(self, action: int) -> ObservationFrame:
        self.t += 1
        done = self._done_every is not None and self.t % self._done_every == 0
        return ObservationFrame(
            image=np.full((64, 64, 3), self.t % 256, dtype=np.uint8),
            is_first=False,
            previous_action=int(action),
            reward=1.0,
            done=done,
            scene_id=self.scene_id,
        )

    def close(self) -> None:
        self.closed = True


class _WrongActionEnv(_ScriptedEnv):
    """Returns frames whose previous_action never matches the input action."""

    def step(self, action: int) -> ObservationFrame:
        frame = super().step(action)
        return ObservationFrame(
            image=frame.image,
            is_first=False,
            previous_action=int(action) + 1,
            reward=frame.reward,
            done=frame.done,
        )


class _SpyAdapter(ObsAdapter):
    """Default adapter plus call recording for scene hooks and augmentation."""

    def __init__(self):
        super().__init__()
        self.scenes: list[str] = []
        self.augment_calls = 0
        self.on_episode_reset = self.scenes.append

    def augment_replay_batch(self, batch):
        self.augment_calls += 1
        return batch


def _collector(env=None, adapter=None, *, capacity=32, buffer=True, **kwargs):
    return ExperienceCollector(
        env=env if env is not None else _ScriptedEnv(),
        adapter=adapter if adapter is not None else _SpyAdapter(),
        num_actions=NUM_ACTIONS,
        buffer=(
            ReplayBuffer(capacity=capacity, num_actions=NUM_ACTIONS)
            if buffer
            else None
        ),
        **kwargs,
    )


class TestRecording:
    def test_step_records_and_reset_does_not(self):
        collector = _collector()

        collector.reset()
        assert collector.buffer_size == 0

        collector.step(1)
        collector.step(2)
        assert collector.buffer_size == 2

    def test_previous_action_mismatch_raises(self):
        collector = _collector(env=_WrongActionEnv())
        collector.reset()

        with pytest.raises(ValueError, match="previous_action"):
            collector.step(1)

    def test_no_buffer_skips_recording_and_action_check(self):
        collector = _collector(env=_WrongActionEnv(), buffer=False)
        collector.reset()

        result = collector.step(1)

        assert collector.buffer_size == 0
        assert result.reward == 1.0


class TestAutoReset:
    def test_done_returns_summary_and_new_episode_first_step(self):
        env = _ScriptedEnv(done_every=2)
        adapter = _SpyAdapter()
        collector = _collector(env=env, adapter=adapter)
        collector.reset()

        collector.step(0)
        result = collector.step(1)

        assert result.done is True
        assert result.episode is not None
        assert result.episode.reward == 2.0
        assert result.episode.steps == 2
        assert result.episode.action_counts.tolist() == [1, 1, 0, 0]
        assert result.agent_step.is_first is True
        assert env.reset_calls == 2

    def test_scene_hook_fires_on_every_reset(self):
        env = _ScriptedEnv(done_every=1, scene_id="scene-x")
        adapter = _SpyAdapter()
        collector = _collector(env=env, adapter=adapter)

        collector.reset()
        collector.step(0)

        assert adapter.scenes == ["scene-x", "scene-x"]

    def test_accumulators_reset_between_episodes(self):
        collector = _collector(env=_ScriptedEnv(done_every=3))
        collector.reset()

        first = None
        for _ in range(6):
            result = collector.step(0)
            if result.episode is not None and first is None:
                first = result.episode
        second = result.episode

        assert first is not None and second is not None
        assert first.reward == second.reward == 3.0
        assert first.steps == second.steps == 3

    def test_default_metrics_is_episode_reward(self):
        collector = _collector(env=_ScriptedEnv(done_every=2))
        collector.reset()
        collector.step(0)
        result = collector.step(0)

        assert result.episode.metrics == {"episode/reward": 2.0}

    def test_metrics_fn_receives_final_frame_and_aggregates(self):
        seen = {}

        def metrics_fn(last_obs, reward, steps, action_counts):
            seen.update(
                obs=last_obs, reward=reward, steps=steps, counts=action_counts.copy()
            )
            return {"metrics/sr": 1.0}

        collector = _collector(
            env=_ScriptedEnv(done_every=2), episode_metrics_fn=metrics_fn
        )
        collector.reset()
        collector.step(3)
        result = collector.step(3)

        assert result.episode.metrics == {"metrics/sr": 1.0}
        assert seen["obs"].done is True
        assert seen["reward"] == 2.0
        assert seen["steps"] == 2
        assert seen["counts"].tolist() == [0, 0, 0, 2]

    def test_summarize_false_suppresses_metrics_but_still_resets(self):
        calls = []
        collector = _collector(
            env=_ScriptedEnv(done_every=1),
            episode_metrics_fn=lambda *a: calls.append(a) or {},
        )
        collector.reset()

        result = collector.step(0, summarize=False)

        assert result.done is True
        assert result.episode is None
        assert not calls
        assert result.agent_step.is_first is True


class TestNoAutoReset:
    def test_done_does_not_reset_and_returns_no_summary(self):
        env = _ScriptedEnv(done_every=2)
        collector = _collector(env=env, buffer=False, auto_reset=False)
        collector.reset()

        collector.step(0)
        result = collector.step(0)

        assert result.done is True
        assert result.episode is None
        assert env.reset_calls == 1

    def test_finish_episode_summarizes_last_frame(self):
        seen = {}

        def metrics_fn(last_obs, reward, steps, action_counts):
            seen["done"] = last_obs.done
            return {"metrics/sr": float(last_obs.done)}

        collector = _collector(
            env=_ScriptedEnv(done_every=None),
            buffer=False,
            auto_reset=False,
            episode_metrics_fn=metrics_fn,
        )
        collector.reset()
        for _ in range(3):
            collector.step(0)

        summary = collector.finish_episode()

        # Step-budget exit: the last frame is not a done frame.
        assert seen["done"] is False
        assert summary.reward == 3.0
        assert summary.steps == 3

    def test_finish_episode_before_step_raises(self):
        collector = _collector(buffer=False, auto_reset=False)
        with pytest.raises(RuntimeError):
            collector.finish_episode()


class TestSample:
    def test_sample_returns_augmented_batch(self):
        adapter = _SpyAdapter()
        collector = _collector(adapter=adapter)
        collector.reset()
        for _ in range(6):
            collector.step(0)

        batch = collector.sample(batch_size=2, seq_len=2)

        assert adapter.augment_calls == 1
        assert batch.rewards.shape == (2, 2)

    def test_sample_without_buffer_raises(self):
        collector = _collector(buffer=False)
        with pytest.raises(RuntimeError, match="does not record"):
            collector.sample(batch_size=1, seq_len=1)


class _VideoEnv(_ScriptedEnv):
    """Scripted env with the state hooks the top-down renderer needs."""

    class _AgentState:
        position = np.array([1.0, 0.0, 2.0])

    class _Goal:
        view_points = ()
        position = np.array([3.0, 0.0, 4.0])

    class _Episode:
        def __init__(self):
            self.goals = [_VideoEnv._Goal()]

    agent_state = _AgentState()

    def __init__(self, done_every=None):
        super().__init__(done_every=done_every)
        self.current_episode = self._Episode()


@pytest.fixture
def _stub_rendering(monkeypatch):
    monkeypatch.setattr(
        "src.r2dreamer.experience.render_topdown_frame",
        lambda env, trajectory, goals: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "src.r2dreamer.experience.compose_frame",
        lambda image, topdown: image,
    )


class TestVideoCapture:
    def test_frames_arrive_in_summary_only_when_capture_started(
        self, _stub_rendering
    ):
        collector = _collector(env=_VideoEnv(done_every=2), buffer=False)
        collector.reset()
        collector.step(0)
        first = collector.step(0).episode
        assert first.video_frames is None

        collector.start_video_capture()
        collector.step(0)
        second = collector.step(0).episode

        # Reset frame + two step frames.
        assert len(second.video_frames) == 3

    def test_capture_stops_at_episode_end(self, _stub_rendering):
        collector = _collector(env=_VideoEnv(done_every=1), buffer=False)
        collector.reset()
        collector.start_video_capture()

        collector.step(0)
        third = collector.step(0).episode

        # Capture was not re-armed for the auto-reset episode.
        assert third.video_frames is None

    def test_supports_video_requires_agent_state(self):
        assert _collector(env=_VideoEnv(), buffer=False).supports_video is True
        assert _collector(buffer=False).supports_video is False

    def test_start_before_reset_raises(self):
        collector = _collector(env=_VideoEnv(), buffer=False)
        with pytest.raises(RuntimeError):
            collector.start_video_capture()


class TestLifecycle:
    def test_close_closes_env(self):
        env = _ScriptedEnv()
        collector = _collector(env=env)
        collector.close()
        assert env.closed is True

    def test_diagnostics_and_growth_delegate_to_adapter(self):
        class _DiagAdapter(_SpyAdapter):
            def diagnostics(self):
                return {"house_buffer/points": 7.0}

            @property
            def growth_history(self):
                return [(10, 100)]

        collector = _collector(adapter=_DiagAdapter(), buffer=False)
        assert collector.diagnostics() == {"house_buffer/points": 7.0}
        assert collector.growth_history == [(10, 100)]

    def test_reset_returns_agent_step(self):
        collector = _collector()
        step = collector.reset()
        assert isinstance(step, AgentStep)
        assert step.is_first is True
