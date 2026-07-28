"""Run lifecycle: metric sinks and the teardown context manager.

``RunLogger`` owns every sink a run writes to (metrics.csv, W&B, console,
MANIFEST.json). ``run_session`` owns the teardown that must happen however the
run loop ends: finalizing the logger with the run status, the
``hard_exit_on_finish`` escape hatch, and closing the collector (which closes
the env). Composition stays in ``src.main`` - this module never builds
anything.
"""

from __future__ import annotations

import contextlib
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
import wandb as _wandb_module

from src.configs.config import R2DreamerConfig, TrainerConfig
from src.r2dreamer.checkpointing import config_snapshot
from src.r2dreamer.experience import EpisodeSummary, ExperienceSource
from src.r2dreamer.manifest import write_manifest_end, write_manifest_start
from src.shared.video_utils import log_episode_video


class RunLoggerLike(Protocol):
    """Metric/video sinks the run loop writes to; no file or W&B types leak."""

    @property
    def wandb_active(self) -> bool: ...
    def start_timing(self, start_step: int) -> None: ...
    def log_episode(self, episode: EpisodeSummary, step: int) -> None: ...
    def log_video(self, key: str, frames: list[np.ndarray], step: int) -> None: ...
    def log_train_metrics(self, metrics: dict[str, Any], step: int) -> None: ...
    def log_reconstructions(
        self, target: np.ndarray, recon: np.ndarray, step: int
    ) -> None: ...
    def write_row(self, step: int, key: str, value: Any) -> None: ...


class RunLogger:
    """All run sinks in one place: metrics.csv, W&B, console, MANIFEST.json.

    Owns the CSV/W&B/manifest lifecycle so the run loop never touches file
    handles or the ``wandb`` module. Construct right before the run starts
    (writes the manifest-start entry and opens metrics.csv); ``run_session``
    calls :meth:`finish` however the loop ends.

    Args:
        agent_config: R2DreamerConfig, snapshotted into the manifest and W&B.
        trainer_config: TrainerConfig (output_dir, W&B knobs, cadences).
        resume: Whether this run resumes a previous one - appends to the
            existing metrics.csv instead of truncating it.
    """

    def __init__(
        self,
        agent_config: R2DreamerConfig,
        trainer_config: TrainerConfig,
        *,
        resume: bool = False,
    ) -> None:
        self.acfg = agent_config
        self.tcfg = trainer_config
        os.makedirs(trainer_config.output_dir, exist_ok=True)

        # MANIFEST.json - emit on start, finalized by finish() with run status.
        write_manifest_start(
            Path(trainer_config.output_dir), config_snapshot(agent_config)
        )

        self._wandb = None
        if trainer_config.wandb_project:
            self._wandb = _wandb_module
            init_kwargs: dict[str, Any] = {
                "project": trainer_config.wandb_project,
                "name": trainer_config.wandb_name,
                "config": config_snapshot(agent_config),
                "tags": trainer_config.wandb_tags,
            }
            if trainer_config.wandb_id is not None:
                # resume="must" fails loudly if the run-id does not exist,
                # which is what we want - silent re-creation orphans runs.
                init_kwargs.update(id=trainer_config.wandb_id, resume="must")
            self._wandb.init(**init_kwargs)

        csv_path = os.path.join(trainer_config.output_dir, "metrics.csv")
        # Append to the existing CSV when resuming so the prior rows survive.
        self._file = open(
            csv_path, "a" if resume else "w", newline="", encoding="utf-8"
        )
        self._writer = csv.writer(self._file)
        if not resume:
            self._writer.writerow(["step", "metric", "value"])

        self._t0 = 0.0
        self._last_log_time = 0.0
        self._last_log_step = 0
        self._start_step = 0

    @property
    def wandb_active(self) -> bool:
        """Whether a W&B run is attached to this logger."""
        return self._wandb is not None

    @property
    def wandb_module(self) -> Any | None:
        """The attached ``wandb`` module, or ``None`` when W&B is off.

        The eval recorder logs per-episode videos under episode-indexed keys,
        which is not a shape :meth:`log_video` covers.
        """
        return self._wandb

    def start_timing(self, start_step: int) -> None:
        """Anchor the fps counters at the run loop's first step."""
        self._t0 = time.time()
        self._last_log_time = self._t0
        self._last_log_step = start_step - 1
        self._start_step = start_step

    def write_row(self, step: int, key: str, value: Any) -> None:
        """Write one metrics.csv row and flush."""
        self._writer.writerow([step, key, value])
        self._file.flush()

    def log_episode(self, episode: EpisodeSummary, step: int) -> None:
        """Log one finished episode's metrics to CSV, W&B, and console."""
        for k, v in episode.metrics.items():
            self._writer.writerow([step, k, v])
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(episode.metrics, step=step)

        sr = episode.metrics.get("metrics/sr", "")
        sr_str = f" SR={sr:.3f}" if isinstance(sr, float) else ""
        print(
            f"[step {step:>8d}] reward={episode.reward:.2f}"
            f" steps={episode.steps}{sr_str}"
        )

    def log_video(self, key: str, frames: list[np.ndarray], step: int) -> None:
        """Log an episode video to W&B (no-op when W&B is off)."""
        log_episode_video(self._wandb, key, frames, step)

    def log_train_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log train-step metrics plus derived perf/fps counters."""
        now = time.time()
        elapsed = now - self._t0
        steps_this_run = step + 1 - self._start_step
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
        self._file.flush()

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        print(
            f"[step {step:>8d}/{self.tcfg.total_steps}] "
            f"total={metrics.get('total_loss', 0):.3f} "
            f"dyn={metrics.get('loss/dyn', 0):.3f} "
            f"rew={metrics.get('loss/rew', 0):.3f} "
            f"policy={metrics.get('loss/policy', 0):.3f} "
            f"fps={fps:.0f} "
            f"fps_interval={fps_interval:.1f} "
            f"ms_step={metrics['perf/ms_per_step_interval']:.1f}"
        )

    def log_reconstructions(
        self, target: np.ndarray, recon: np.ndarray, step: int
    ) -> None:
        """Log up to 4 side-by-side ``input | recon`` panels to W&B (3D-51)."""
        if self._wandb is None:
            return
        n = min(4, target.shape[0])
        images = []
        for i in range(n):
            combo = np.concatenate([target[i], recon[i]], axis=1)  # side by side
            combo = np.clip(combo * 255.0, 0, 255).astype(np.uint8)
            images.append(self._wandb.Image(combo, caption=f"input | recon ({i})"))
        self._wandb.log({"decoder/reconstructions": images}, step=step)

    def log_adapter_summary(
        self,
        stats: dict[str, float],
        final_step: int,
    ) -> None:
        """Write the adapter's end-of-run diagnostics.

        Rows land in ``metrics.csv`` under the final trainer step; the same
        stats also go to the W&B run summary when W&B is active. Adapters
        without a diagnostics hook report nothing and this is a no-op.
        """
        if not stats:
            return
        for k, v in stats.items():
            self._writer.writerow([final_step, k, v])
        self._file.flush()
        if self._wandb is not None:
            self._wandb.summary.update(stats)
        print("=== adapter summary ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    def close_metrics_file(self) -> None:
        """Close metrics.csv (idempotent); called before the final checkpoint."""
        if not self._file.closed:
            self._file.close()

    def finish(self, status: str) -> None:
        """Finalize the run: close CSV, write manifest end, finish W&B."""
        self.close_metrics_file()
        write_manifest_end(Path(self.tcfg.output_dir), status)
        if self._wandb is not None:
            self._wandb.finish()


@contextlib.contextmanager
def run_session(
    logger: RunLogger,
    collector: ExperienceSource,
    *,
    hard_exit: bool,
) -> Iterator[None]:
    """Own the run's teardown, however the loop inside ends.

    The body runs the loop plus whatever end-of-run writes belong to a
    *successful* run (final checkpoint, adapter summary) - if the body raises,
    those are skipped and the status records the failure.

    On exit: finalize the logger with the run status, then either hard-exit
    (``hard_exit`` and completed: habitat_sim's GL teardown SIGABRTs on some
    magnum builds, poisoning the exit code of a fully completed run whose
    checkpoint, manifest and W&B are already flushed) or close the collector,
    which closes the env. Failures always take the close path so their
    non-zero exit and traceback survive.

    Args:
        logger: The run's sinks; ``finish(status)`` is guaranteed.
        collector: The run's rollout owner; ``close()`` on the non-exit path.

    Yields:
        Nothing - the caller runs its loop in the body.
    """
    status = "failed"
    try:
        yield
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        logger.finish(status)
        if hard_exit and status == "completed":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        collector.close()
