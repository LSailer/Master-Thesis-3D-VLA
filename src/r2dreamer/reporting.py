"""Reporting collaborators for ``Trainer``: CSV/W&B metrics and video capture.

``MetricsLogger`` owns the CSV-row + W&B-log side effects for episode,
training-step, adapter-summary, and validation metrics (plus the optional
decoder-reconstruction image logging). ``EpisodeRecorder`` owns the
topdown/video frame capture used for W&B episode videos. Both are
constructor-injected into ``Trainer`` so the loop logic in ``trainer.py``
stays focused on orchestration (prefill/train/validate) rather than the
mechanics of where metrics and frames go.
"""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from typing import Any, Iterator

import jax
import numpy as np

from src.buffer.replay_buffer import ReplayBatch
from src.environments.observation import ObservationFrame
from src.shared.video_utils import compose_frame, log_episode_video, render_topdown_frame


class MetricsLogger:
    """Writes episode/train/validation metrics to CSV and (optionally) W&B.

    Owns the ``metrics.csv`` lifecycle: :meth:`open_csv` opens the file
    (truncating or appending per resume), writes the header only on a fresh
    run, and holds the open ``csv.writer`` + file handle for the duration of
    the ``with`` block. The metric-logging methods write to that internal
    handle, so ``Trainer`` passes the logger rather than threading raw
    ``(writer, f)`` handles through its loop methods.

    Args:
        wandb_run: The ``wandb`` module handle (already ``wandb.init``-ed by
            the caller), or ``None`` to disable W&B logging entirely. Passed
            in rather than imported here so the optional-wandb import guard
            stays centralized in ``Trainer.__init__``.

    Attributes:
        wandb_run: The injected ``wandb`` handle (or ``None`` when disabled).
    """

    def __init__(self, wandb_run: Any | None = None) -> None:
        self._wandb = wandb_run
        self._t0 = time.time()
        self._last_log_time = self._t0
        self._last_log_step = -1
        self._writer: Any = None
        self._f: Any = None

    def _require_writer(self) -> None:
        """Assert an :meth:`open_csv` context is active before writing rows.

        Raises:
            RuntimeError: If no CSV is open (``open_csv`` was never entered or
                has already exited), so callers fail loudly instead of hitting
                an ``AttributeError`` on ``None._writer``.
        """
        if self._writer is None:
            raise RuntimeError(
                "MetricsLogger: no active CSV — call open_csv() first"
            )

    @contextmanager
    def open_csv(self, path: str, resume: bool) -> Iterator[None]:
        """Open ``metrics.csv`` for the run and hold it for the ``with`` block.

        On a fresh run (``resume`` False) the file is truncated and the
        ``step,metric,value`` header row is written; on resume the file is
        opened in append mode and the header is *not* re-written, so the prior
        rows survive. The open writer/handle back all metric-logging methods
        until the context exits, which closes the file.

        Args:
            path: Path to ``metrics.csv``.
            resume: True to append to an existing CSV (skip the header), False
                to start a fresh CSV with a header row.

        Yields:
            None. Use as ``with logger.open_csv(path, resume):``.
        """
        mode = "a" if resume else "w"
        with open(path, mode, newline="") as f:
            writer = csv.writer(f)
            if not resume:
                writer.writerow(["step", "metric", "value"])
            self._writer = writer
            self._f = f
            try:
                yield
            finally:
                self._writer = None
                self._f = None

    def start_timing(self, start_step: int) -> None:
        """Reset the fps/timing baselines at the start of a train loop.

        Args:
            start_step: The step the train loop begins at (``resume_step`` on
                resume, else 0). Used so ``perf/fps_cumulative`` only counts
                steps executed in this run.
        """
        self._t0 = time.time()
        self._last_log_time = self._t0
        self._last_log_step = start_step - 1

    def log_episode_end(
        self,
        ep_metrics: dict[str, Any],
        episode_reward: float,
        episode_steps: int,
        step: int,
    ) -> None:
        """Write end-of-episode metrics to CSV/W&B and print a console summary.

        Requires an active :meth:`open_csv` context.

        Args:
            ep_metrics: Episode metrics dict (from ``episode_metrics_fn`` or
                the default ``{"episode/reward": ...}`` fallback).
            episode_reward: Total reward accrued in the finished episode.
            episode_steps: Number of steps in the finished episode.
            step: Global training step at episode end.

        Raises:
            RuntimeError: If called outside an active :meth:`open_csv` context.
        """
        self._require_writer()
        for k, v in ep_metrics.items():
            self._writer.writerow([step, k, v])
        self._f.flush()

        if self._wandb is not None:
            self._wandb.log(ep_metrics, step=step)

        sr = ep_metrics.get("metrics/sr", "")
        sr_str = f" SR={sr:.3f}" if isinstance(sr, float) else ""
        print(
            f"[step {step:>8d}] reward={episode_reward:.2f}"
            f" steps={episode_steps}{sr_str}"
        )

    def log_adapter_summary(
        self,
        stats: dict[str, Any],
        history: list[tuple[int, int]],
        final_step: int,
    ) -> None:
        """Write end-of-run adapter diagnostics and the buffer growth curve.

        Growth rows are keyed by the adapter's own env-step counter (which
        includes prefill), not the trainer step, and land in ``metrics.csv``
        as ``house_buffer/points_growth``. Final stats also go to the W&B
        run summary when W&B is active. Requires an active :meth:`open_csv`
        context.

        Args:
            stats: Adapter diagnostics dict (``obs_adapter.diagnostics()``).
                No-op if empty.
            history: ``(env_step, total_points)`` growth history
                (``obs_adapter.growth_history``).
            final_step: Trainer step to key the ``stats`` rows under.

        Raises:
            RuntimeError: If called (with non-empty ``stats``) outside an active
                :meth:`open_csv` context.
        """
        if not stats:
            return
        self._require_writer()
        for k, v in stats.items():
            self._writer.writerow([final_step, k, v])
        for env_step, points in history:
            self._writer.writerow([env_step, "house_buffer/points_growth", points])
        self._f.flush()
        if self._wandb is not None:
            self._wandb.summary.update(stats)
        print("=== house buffer summary ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if history:
            print("  growth (env_step -> total_points):")
            for env_step, points in history:
                print(f"    {env_step:>9d} -> {points}")

    def write_metric_rows(self, rows: list[tuple[int, str, Any]]) -> None:
        """Write raw ``(step, metric, value)`` rows to the CSV, then flush.

        For ad-hoc verification rows (e.g. the overfit-loop pass/fail markers)
        that have no W&B counterpart. Requires an active :meth:`open_csv`
        context.

        Args:
            rows: ``(step, metric, value)`` triples to append to ``metrics.csv``.

        Raises:
            RuntimeError: If called outside an active :meth:`open_csv` context.
        """
        self._require_writer()
        for step, metric, value in rows:
            self._writer.writerow([step, metric, value])
        self._f.flush()

    def log_train_metrics(
        self,
        metrics: dict,
        step: int,
        total_steps: int,
        resume_step: int,
    ) -> None:
        """Augment training metrics with fps/timing fields, then log + print.

        Requires an active :meth:`open_csv` context.

        Args:
            metrics: Training-step metrics dict from ``agent.train_step``
                (mutated in place with ``perf/*`` fields, matching prior
                Trainer behavior).
            step: Global training step.
            total_steps: Configured total step count (for the console line).
            resume_step: Step the current run started at (for cumulative fps).

        Raises:
            RuntimeError: If called outside an active :meth:`open_csv` context.
        """
        self._require_writer()
        now = time.time()
        elapsed = now - self._t0
        steps_this_run = step + 1 - resume_step
        fps = steps_this_run / elapsed if elapsed > 0 else 0

        interval_steps = max(1, step - self._last_log_step)
        interval_elapsed = now - self._last_log_time
        fps_interval = interval_steps / interval_elapsed if interval_elapsed > 0 else 0
        metrics["perf/fps_cumulative"] = fps
        metrics["perf/fps_interval"] = fps_interval
        metrics["perf/ms_per_step_interval"] = (
            1000.0 / fps_interval if fps_interval > 0 else 0
        )
        self._last_log_time = now
        self._last_log_step = step

        for k, v in metrics.items():
            self._writer.writerow([step, k, v])
        self._f.flush()

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        print(
            f"[step {step:>8d}/{total_steps}] "
            f"total={metrics.get('total_loss', 0):.3f} "
            f"dyn={metrics.get('loss/dyn', 0):.3f} "
            f"rew={metrics.get('loss/rew', 0):.3f} "
            f"policy={metrics.get('loss/policy', 0):.3f} "
            f"fps={fps:.0f} "
            f"fps_interval={fps_interval:.1f} "
            f"ms_step={metrics['perf/ms_per_step_interval']:.1f}"
        )

    def maybe_log_recon(
        self,
        agent: Any,
        batch: ReplayBatch,
        step: int,
        decoder_enabled: bool,
    ) -> None:
        """Log decoder input/reconstruction image pairs to W&B (3D-51).

        No-op unless a decoder is configured and W&B is active. Decodes the
        sampled training batch and logs up to 4 side-by-side ``input | recon``
        panels so the learned hybrid representation can be eyeballed during a run.

        Args:
            agent: The R2Dreamer-like agent providing ``reconstruct``.
            batch: The just-sampled/augmented training batch.
            step: Global training step (used as the W&B log step).
            decoder_enabled: Whether ``agent_config.decoder`` is truthy.
        """
        if self._wandb is None or not decoder_enabled:
            return
        pair = agent.reconstruct(batch)
        if pair is None:
            return
        target, recon = jax.device_get(pair)  # (B*T, 3, 64, 64) in [0, 1]
        n = min(4, target.shape[0])
        images = []
        for i in range(n):
            tgt = np.transpose(target[i], (1, 2, 0))  # CHW -> HWC
            rec = np.transpose(recon[i], (1, 2, 0))
            combo = np.concatenate([tgt, rec], axis=1)  # side by side
            combo = np.clip(combo * 255.0, 0, 255).astype(np.uint8)
            images.append(self._wandb.Image(combo, caption=f"input | recon ({i})"))
        self._wandb.log({"decoder/reconstructions": images}, step=step)

    def prefix_val_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Prefix validation metric keys with ``val/`` (idempotent).

        Args:
            metrics: Raw validation-episode metrics.

        Returns:
            A new dict with every key prefixed by ``val/`` unless already so.
        """
        return {
            f"val/{k}" if not k.startswith("val/") else k: v for k, v in metrics.items()
        }

    def log_val_metrics(
        self,
        val_logged: dict[str, Any],
        step: int,
    ) -> None:
        """Write validation metrics to CSV/W&B.

        Requires an active :meth:`open_csv` context.

        Args:
            val_logged: Already ``val/``-prefixed metrics dict.
            step: Global training step.

        Raises:
            RuntimeError: If called outside an active :meth:`open_csv` context.
        """
        self._require_writer()
        for k, v in val_logged.items():
            self._writer.writerow([step, k, v])
        self._f.flush()
        if self._wandb is not None:
            self._wandb.log(val_logged, step=step)

    def print_val_summary(
        self,
        val_logged: dict[str, Any],
        step: int,
        elapsed: float,
        val_episodes: int,
    ) -> None:
        """Print a one-line console summary of a validation loop pass.

        Args:
            val_logged: Already ``val/``-prefixed metrics dict.
            step: Global training step.
            elapsed: Wall-clock seconds the validation loop took.
            val_episodes: Number of episodes run in this validation pass.
        """
        sr = val_logged.get("val/metrics/sr", 0.0)
        spl = val_logged.get("val/metrics/spl", 0.0)
        softspl = val_logged.get("val/metrics/softspl", 0.0)
        dtg = val_logged.get("val/metrics/dtg", 0.0)
        sr_str = f"{sr:.3f}" if isinstance(sr, float) else str(sr)
        spl_str = f"{spl:.3f}" if isinstance(spl, float) else str(spl)
        soft_str = f"{softspl:.3f}" if isinstance(softspl, float) else str(softspl)
        dtg_str = f"{dtg:.3f}" if isinstance(dtg, float) else str(dtg)
        print(
            f"[step {step:>8d}] VAL-LOOP "
            f"sr={sr_str} spl={spl_str} softspl={soft_str} dtg={dtg_str}m "
            f"({val_episodes} eps in {elapsed:.1f}s)"
        )


class EpisodeRecorder:
    """Captures composed RGB+topdown video frames for W&B episode videos.

    Args:
        wandb_run: The ``wandb`` module handle, or ``None`` to disable video
            logging entirely (mirrors ``MetricsLogger``'s injection style).
    """

    def __init__(self, wandb_run: Any | None = None) -> None:
        self._wandb = wandb_run

    def should_record_video(
        self,
        env: Any,
        step: int,
        next_video_step: int,
        video_log_every: int,
        video_log_episodes: int,
    ) -> bool:
        """Return whether a new video recording should start at ``step``.

        Args:
            env: The environment the video would be recorded from — must
                expose ``_env`` (Habitat-only feature).
            step: Current global training step.
            next_video_step: Step at/after which the next recording is due.
            video_log_every: Recording cadence in steps (``<= 0`` disables).
            video_log_episodes: Number of episodes to capture per cadence hit
                (``<= 0`` disables).

        Returns:
            True if a recording should be started now.
        """
        return (
            self._wandb is not None
            and video_log_every > 0
            and video_log_episodes > 0
            and step >= next_video_step
            and hasattr(env, "_env")
        )

    def _goal_positions(self, env: Any) -> list[list[float]]:
        positions = []
        for goal in env.current_episode.goals:
            if goal.view_points:
                pos = goal.view_points[0].agent_state.position
            else:
                pos = goal.position
            positions.append(pos.tolist() if hasattr(pos, "tolist") else list(pos))
        return positions

    def _agent_position(self, env: Any) -> list[float]:
        pos = env._env.sim.get_agent_state().position
        return pos.tolist() if hasattr(pos, "tolist") else list(pos)

    def start_recording(self, env: Any, obs: ObservationFrame) -> dict[str, Any]:
        """Start a new recording, capturing the initial frame.

        Args:
            env: Environment to read agent/goal positions from.
            obs: Initial observation to compose into the first frame.

        Returns:
            A recording dict with ``trajectory``, ``goals``, and ``frames``
            keys, ready to be extended by ``append_frame``.
        """
        recording = {
            "trajectory": [self._agent_position(env)],
            "goals": self._goal_positions(env),
            "frames": [],
        }
        self.append_frame(env, recording, obs)
        return recording

    def append_frame(
        self, env: Any, recording: dict[str, Any], obs: ObservationFrame
    ) -> None:
        """Append one composed RGB+topdown frame to an in-progress recording.

        Args:
            env: Environment to read the current agent position from.
            recording: Recording dict previously returned by
                ``start_recording`` (mutated in place).
            obs: Observation whose image is composed with the topdown render.
        """
        if recording["frames"]:
            recording["trajectory"].append(self._agent_position(env))
        topdown = render_topdown_frame(env, recording["trajectory"], recording["goals"])
        recording["frames"].append(compose_frame(obs.image, topdown))

    def log_video(self, key: str, recording: dict[str, Any], step: int) -> None:
        """Flush a finished recording's frames to W&B as an episode video.

        Args:
            key: W&B log key (e.g. ``"train/episode_video"``).
            recording: Recording dict with a ``frames`` key.
            step: Global training step to log the video under.
        """
        log_episode_video(self._wandb, key, recording["frames"], step)
