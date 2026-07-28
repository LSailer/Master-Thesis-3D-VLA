"""Experience collection behind a trainer-facing protocol (ADR 0006).

``ExperienceCollector`` owns the environment, the frame adapter, and the
(optional) ``ReplayBuffer``. The training loop only ever sees prepared
encoder observations (``AgentStep``), finished-episode aggregates
(``EpisodeSummary``), and sampled replay batches — never raw
``ObservationFrame`` objects, the env, or the buffer.

The boundary follows the frozen/trainable split: frozen extraction (VGGT,
house-point accumulation) happens inside the adapter on this side; the
trainable encoders stay in ``agent.params["encoder"]`` and run inside
``agent.train_step`` on the other side.

The adapter arrives as a single ``AdapterFn``: a variant that needs frozen
features holds its own extractor and calls it inside, so the collector neither
knows about VGGT nor about cache lifetimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from src.adapters.contract import (
    AdapterFn,
    AdapterOutput,
    encoder_obs_from_fields,
    transition_from_fields,
)
from src.buffer.replay_buffer import ReplayBatch, ReplayBuffer
from src.environments.observation import ObservationFrame
from src.shared.video_utils import compose_frame, render_topdown_frame


class Env(Protocol):
    """Minimal environment interface any rollout driver needs.

    Deliberately limited to what every rollout needs. Habitat-only state
    (``current_episode``, ``agent_state``) stays out: it is read exclusively on
    the video-capture path, which the collector already gates behind
    :attr:`ExperienceCollector.supports_video`, so requiring it here would lock
    non-habitat envs out of plain training for a feature they never use.
    """

    @property
    def num_actions(self) -> int: ...

    def reset(self) -> ObservationFrame: ...
    def step(self, action: int) -> ObservationFrame: ...
    def close(self) -> None: ...


# Called at episode end with the final observation and episode aggregates. The
# env (if needed) is bound by the callable itself (e.g. HabitatEpisodeMetrics),
# so it is not passed here.
EpisodeMetricsFn = Callable[[ObservationFrame, float, int, np.ndarray], dict[str, Any]]


@dataclass(frozen=True)
class AgentStep:
    """What the agent needs to act: prepared encoder inputs plus boundary flag.

    Attributes:
        encoder_obs: Frozen-stage features ready for the trainable encoder.
        is_first: Whether this observation starts a sequence (episode reset).
    """

    encoder_obs: Any
    is_first: bool


@dataclass(frozen=True)
class EpisodeSummary:
    """Aggregates of one finished episode; env internals stay inside.

    Attributes:
        metrics: Episode metrics with the collector's metrics fn applied
            (defaults to ``{"episode/reward": reward}`` when no fn is set).
        reward: Total undiscounted episode reward.
        steps: Number of env steps in the episode.
        action_counts: Per-action histogram of shape ``(num_actions,)``.
        video_frames: Composed video frames when capture was started for this
            episode, else ``None``.
    """

    metrics: dict[str, Any]
    reward: float
    steps: int
    action_counts: np.ndarray
    video_frames: list[np.ndarray] | None


@dataclass(frozen=True)
class StepResult:
    """Outcome of one collector step.

    Attributes:
        agent_step: The next observation to act on. When the episode ended and
            the collector auto-resets, this is the *new* episode's first
            observation.
        reward: Reward returned by the env for this step.
        done: Whether this step ended the episode.
        episode: The finished episode's summary — set exactly when ``done`` is
            true and the collector both auto-resets and summarizes.
    """

    agent_step: AgentStep
    reward: float
    done: bool
    episode: EpisodeSummary | None


class ExperienceSource(Protocol):
    """Everything the training loop may touch on the experience side."""

    def reset(self) -> AgentStep: ...
    def step(self, action: int, *, summarize: bool = True) -> StepResult: ...
    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch: ...
    @property
    def buffer_size(self) -> int: ...
    @property
    def supports_video(self) -> bool: ...
    def start_video_capture(self) -> None: ...
    def finish_episode(self) -> EpisodeSummary: ...
    def diagnostics(self) -> dict[str, float]: ...
    def close(self) -> None: ...


class ExperienceCollector:
    """Owns env + adapter + optional buffer behind ``ExperienceSource``.

    Args:
        env: Environment to roll out in.
        observe: Adapter turning one env frame into routed fields (frozen
            extraction happens inside it).
        num_actions: Size of the discrete action space (for action histograms).
        buffer: Replay buffer to record transitions into. ``None`` disables
            recording (validation collectors).
        episode_metrics_fn: Optional episode-end metrics callback; receives the
            final frame and the episode aggregates.
        auto_reset: Whether ``step`` resets the env when an episode ends. Keep
            ``False`` for validation — an extra reset would advance the env's
            episode iterator and change which episodes run.
        diagnostics_fn: Optional end-of-run health metrics from the adapter
            (e.g. house-buffer fill). Called once when the run finishes, so it
            may synchronize device scalars to host.
    """

    def __init__(
        self,
        env: Env,
        observe: AdapterFn,
        *,
        num_actions: int,
        buffer: ReplayBuffer | None = None,
        episode_metrics_fn: EpisodeMetricsFn | None = None,
        auto_reset: bool = True,
        diagnostics_fn: Callable[[], dict[str, float]] | None = None,
    ) -> None:
        self.env = env
        self.observe = observe
        self.buffer = buffer
        self.diagnostics_fn = diagnostics_fn
        self.num_actions = num_actions
        self.episode_metrics_fn = episode_metrics_fn
        self.auto_reset = auto_reset
        self._last_frame: ObservationFrame | None = None
        self._last_fields: AdapterOutput | None = None
        self._recording: dict[str, Any] | None = None
        self._reset_accumulators()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def reset(self) -> AgentStep:
        """Reset the env and return the first agent step.

        Reset frames are never recorded into the buffer — replay windows start
        at the first stepped transition, matching prefill and train alike.

        Returns:
            The new episode's first prepared observation.
        """
        self._reset_accumulators()
        self._recording = None
        return self._do_reset()

    def reset_fields(self) -> AdapterOutput:
        """Reset and return the adapter's routed fields for the first frame.

        The composition root uses this to learn the encoder routing and field
        shapes before building the agent, without having to call the env or the
        adapter itself.

        Returns:
            The adapter output for the new episode's first frame.
        """
        self.reset()
        assert self._last_fields is not None  # set by every _do_reset
        return self._last_fields

    def step(self, action: int, *, summarize: bool = True) -> StepResult:
        """Step the env, record the transition, and handle episode ends.

        Args:
            action: Discrete action chosen by the caller (policy or random —
                the collector never decides actions).
            summarize: Whether to build an ``EpisodeSummary`` when the episode
                ends. Pass ``False`` during prefill so the episode metrics fn
                (which may mutate rolling trackers) never fires there.

        Returns:
            The step outcome. On ``done`` with ``auto_reset``, the contained
            ``agent_step`` already belongs to the next episode.

        Raises:
            ValueError: If the env frame's ``previous_action`` does not match
                ``action`` (only checked when recording into a buffer).
        """
        frame = self.env.step(action)
        fields = self.observe(frame)

        if self.buffer is not None:
            if frame.previous_action != action:
                raise ValueError(
                    "ObservationFrame.previous_action does not match recorded "
                    f"action: expected {action}, got {frame.previous_action}"
                )
            self.buffer.add(transition_from_fields(frame, fields))

        self._action_counts[action] += 1
        self._episode_reward += float(frame.reward)
        self._episode_steps += 1
        self._last_frame = frame
        self._last_fields = fields
        if self._recording is not None:
            self._append_video_frame(frame)

        agent_step = AgentStep(
            encoder_obs=encoder_obs_from_fields(fields), is_first=frame.is_first
        )
        episode: EpisodeSummary | None = None
        if frame.done and self.auto_reset:
            if summarize:
                episode = self._build_summary(frame)
            self._reset_accumulators()
            self._recording = None
            agent_step = self._do_reset()

        return StepResult(
            agent_step=agent_step,
            reward=float(frame.reward),
            done=bool(frame.done),
            episode=episode,
        )

    def finish_episode(self) -> EpisodeSummary:
        """Summarize the current episode without resetting the env.

        For ``auto_reset=False`` callers (validation): call after the rollout
        loop exits — whether it ended on ``done`` or on a step budget — then
        call :meth:`reset` to begin the next episode.

        Returns:
            The episode summary built from the last observed frame.

        Raises:
            RuntimeError: If no step has been taken since the last reset.
        """
        if self._last_frame is None:
            raise RuntimeError("finish_episode called before any step")
        summary = self._build_summary(self._last_frame)
        self._reset_accumulators()
        self._recording = None
        return summary

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch:
        """Sample a replay batch.

        The batch carries the adapter's encoder routing and the latest live
        field alongside the per-step observations, so no post-hoc augmentation
        step is needed.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Length of each sampled window.

        Returns:
            The replay batch, ready for ``agent.train_step``.

        Raises:
            RuntimeError: If this collector records no experience
                (``buffer=None``).
        """
        if self.buffer is None:
            raise RuntimeError("this collector does not record experience")
        return self.buffer.sample(batch_size, seq_len)

    @property
    def buffer_size(self) -> int:
        """Number of stored transitions (0 when recording is disabled)."""
        return self.buffer.size if self.buffer is not None else 0

    # ------------------------------------------------------------------
    # Video capture
    # ------------------------------------------------------------------

    @property
    def supports_video(self) -> bool:
        """Whether the env exposes the state needed for top-down videos."""
        return hasattr(self.env, "agent_state")

    def start_video_capture(self) -> None:
        """Begin capturing the current episode, starting at its latest frame.

        Frames are delivered in the episode's ``EpisodeSummary``; capture stops
        automatically when the episode ends.

        Raises:
            RuntimeError: If called before the first reset.
            AttributeError: If the env does not expose the habitat state the
                top-down map needs (``agent_state``, ``current_episode``).
        """
        if self._last_frame is None:
            raise RuntimeError("start_video_capture called before reset")
        self._recording = {
            "trajectory": [self._agent_position()],
            "goals": self._goal_positions(),
            "frames": [],
        }
        self._append_video_frame(self._last_frame)

    # ------------------------------------------------------------------
    # Diagnostics / lifecycle
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, float]:
        """Return the adapter's end-of-run health metrics (empty when unwired)."""
        return self.diagnostics_fn() if self.diagnostics_fn is not None else {}

    def close(self) -> None:
        """Close the owned environment."""
        self.env.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_accumulators(self) -> None:
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._action_counts = np.zeros(self.num_actions, dtype=int)

    def _do_reset(self) -> AgentStep:
        # Reset frames pass through the adapter (so per-episode state - VGGT
        # cache, house-cloud compaction - reacts to the boundary) but are never
        # recorded: they carry no previous action.
        frame = self.env.reset()
        fields = self.observe(frame)
        self._last_frame = frame
        self._last_fields = fields
        return AgentStep(
            encoder_obs=encoder_obs_from_fields(fields), is_first=frame.is_first
        )

    def _build_summary(self, last_frame: ObservationFrame) -> EpisodeSummary:
        if self.episode_metrics_fn is not None:
            metrics = self.episode_metrics_fn(
                last_frame,
                self._episode_reward,
                self._episode_steps,
                self._action_counts,
            )
        else:
            metrics = {"episode/reward": self._episode_reward}
        frames = self._recording["frames"] if self._recording is not None else None
        return EpisodeSummary(
            metrics=metrics,
            reward=self._episode_reward,
            steps=self._episode_steps,
            action_counts=self._action_counts,
            video_frames=frames,
        )

    def _goal_positions(self) -> list[list[float]]:
        # Habitat-only state, reached from video capture alone. Fetch it the same
        # defensive way as _agent_position so a non-habitat env asked for video
        # fails here with the env name, not with an AttributeError raised deep
        # inside the goal loop.
        episode = getattr(self.env, "current_episode", None)
        if episode is None:
            raise AttributeError(
                f"{type(self.env).__name__} does not expose current_episode "
                "for video logging"
            )
        positions = []
        for goal in episode.goals:
            if goal.view_points:
                pos = goal.view_points[0].agent_state.position
            else:
                pos = goal.position
            positions.append(pos.tolist() if hasattr(pos, "tolist") else list(pos))
        return positions

    def _agent_position(self) -> list[float]:
        agent_state = getattr(self.env, "agent_state", None)
        if agent_state is None:
            raise AttributeError(
                f"{type(self.env).__name__} does not expose agent_state "
                "for video logging"
            )
        pos = agent_state.position
        return pos.tolist() if hasattr(pos, "tolist") else list(pos)

    def _append_video_frame(self, frame: ObservationFrame) -> None:
        recording = self._recording
        assert recording is not None
        if recording["frames"]:
            recording["trajectory"].append(self._agent_position())
        topdown = render_topdown_frame(
            self.env, recording["trajectory"], recording["goals"]
        )
        # frame.image is typed as a device array, while compose_frame resizes
        # and pads through PIL and therefore needs host memory. Materialise at
        # this boundary so the transfer is visible instead of hidden inside
        # compose_frame.
        recording["frames"].append(compose_frame(np.asarray(frame.image), topdown))
