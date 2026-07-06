"""Direct unit tests for the reporting collaborators (MetricsLogger/Recorder).

Covers the ``MetricsLogger(None)`` W&B-disabled no-op path, train-metric CSV
row emission, the F4 CSV open/resume/header lifecycle (header only on fresh
runs, prior rows survive on resume), and ``EpisodeRecorder`` video-gating logic.
"""
# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

import csv
import time

import pytest

from src.r2dreamer.reporting import EpisodeRecorder, MetricsLogger


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_open_csv_fresh_writes_header_and_rows(tmp_path):
    """A fresh run writes the header then the logged train-metric rows."""
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(None)  # W&B disabled
    logger.start_timing(0)
    with logger.open_csv(str(path), resume=False):
        logger.log_train_metrics({"total_loss": 1.0}, step=0, total_steps=10, resume_step=0)

    rows = _read_rows(path)
    assert rows[0] == ["step", "metric", "value"]
    metrics = {r[1] for r in rows[1:]}
    assert "total_loss" in metrics
    # W&B-disabled logger stores nothing to a run and does not raise.
    assert logger._wandb is None


def test_open_csv_resume_appends_without_new_header(tmp_path):
    """Resume opens in append mode, skips the header, and keeps prior rows."""
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(None)

    # Fresh run: header + one row.
    logger.start_timing(0)
    with logger.open_csv(str(path), resume=False):
        logger.log_train_metrics({"total_loss": 1.0}, step=0, total_steps=10, resume_step=0)
    first_rows = _read_rows(path)

    # Resume run: no new header, prior rows survive, a new row is appended.
    logger.start_timing(5)
    with logger.open_csv(str(path), resume=True):
        logger.log_train_metrics({"total_loss": 0.5}, step=5, total_steps=10, resume_step=5)
    resumed_rows = _read_rows(path)

    header_count = sum(1 for r in resumed_rows if r == ["step", "metric", "value"])
    assert header_count == 1  # header written exactly once (fresh run only)
    assert len(resumed_rows) > len(first_rows)  # prior rows survived + appended
    assert resumed_rows[: len(first_rows)] == first_rows


def test_start_timing_first_row_interval_is_measured(tmp_path):
    """The overfit loop's first row measures a real (>0) timing interval.

    ``start_timing(0)`` sets ``_last_log_time`` to now, so the very first
    ``log_train_metrics`` after a short delay reports positive interval fps
    and ms/step — the accepted behavior (the prior HEAD reported 0 only by
    accident of an unset attribute + getattr default).
    """
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(None)
    logger.start_timing(0)
    time.sleep(0.01)
    metrics = {"total_loss": 1.0}
    with logger.open_csv(str(path), resume=False):
        logger.log_train_metrics(metrics, step=0, total_steps=10, resume_step=0)

    assert metrics["perf/fps_interval"] > 0
    assert metrics["perf/ms_per_step_interval"] > 0

    rows = _read_rows(path)
    logged = {r[1]: r[2] for r in rows[1:]}
    assert float(logged["perf/fps_interval"]) > 0
    assert float(logged["perf/ms_per_step_interval"]) > 0


def test_write_metric_rows(tmp_path):
    """write_metric_rows appends raw (step, metric, value) triples."""
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(None)
    with logger.open_csv(str(path), resume=False):
        logger.write_metric_rows([(9, "verify/overfit_pass", 1.0)])
    rows = _read_rows(path)
    assert ["9", "verify/overfit_pass", "1.0"] in rows


def test_flush_persists_row_before_close(tmp_path):
    """Each row is flushed to disk immediately, before the context exits."""
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(None)
    with logger.open_csv(str(path), resume=False):
        logger.write_metric_rows([(3, "verify/mid_context", 1.0)])
        # Read from disk WITHOUT leaving the context: the row must already be there.
        rows = _read_rows(path)
        assert ["3", "verify/mid_context", "1.0"] in rows


def test_log_outside_open_csv_raises():
    """Row-writing methods raise RuntimeError with no active open_csv context."""
    logger = MetricsLogger(None)
    with pytest.raises(RuntimeError):
        logger.write_metric_rows([(0, "verify/no_context", 1.0)])


def test_open_csv_closes_on_exception(tmp_path):
    """An exception inside open_csv still releases the handle and file."""
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(None)

    class _Sentinel(Exception):
        pass

    with pytest.raises(_Sentinel):
        with logger.open_csv(str(path), resume=False):
            logger.write_metric_rows([(0, "verify/before_raise", 1.0)])
            raise _Sentinel()

    # Handle released on context exit despite the exception.
    assert logger._writer is None
    assert logger._f is None

    # The file can be reopened/appended and prior rows survive.
    with logger.open_csv(str(path), resume=True):
        logger.write_metric_rows([(1, "verify/after_reopen", 2.0)])
    rows = _read_rows(path)
    assert ["0", "verify/before_raise", "1.0"] in rows
    assert ["1", "verify/after_reopen", "2.0"] in rows


def test_maybe_log_recon_noop_when_wandb_disabled():
    """maybe_log_recon is a no-op (never touches the agent) without W&B."""
    logger = MetricsLogger(None)

    class _ExplodingAgent:
        def reconstruct(self, batch):
            raise AssertionError("reconstruct must not be called when W&B is off")

    # Should return without raising and without calling reconstruct.
    logger.maybe_log_recon(_ExplodingAgent(), batch=None, step=0, decoder_enabled=True)


def test_episode_recorder_should_record_video_disabled_without_wandb():
    """should_record_video is False whenever the W&B handle is None."""
    recorder = EpisodeRecorder(None)

    class _EnvWithHabitat:
        _env = object()

    assert not recorder.should_record_video(
        _EnvWithHabitat(),
        step=100,
        next_video_step=0,
        video_log_every=10,
        video_log_episodes=1,
    )


def test_episode_recorder_should_record_video_gating():
    """With W&B active, gating obeys cadence, count, step, and Habitat env."""
    sentinel = object()
    recorder = EpisodeRecorder(sentinel)  # non-None "wandb" handle

    class _HabitatEnv:
        _env = object()

    class _NonHabitatEnv:
        pass

    common = dict(next_video_step=50, video_log_every=10, video_log_episodes=2)

    # All conditions satisfied -> record.
    assert recorder.should_record_video(_HabitatEnv(), step=50, **common)
    # Step below the next-video-step gate -> skip.
    assert not recorder.should_record_video(_HabitatEnv(), step=49, **common)
    # Non-Habitat env (no _env attr) -> skip.
    assert not recorder.should_record_video(_NonHabitatEnv(), step=50, **common)
    # Disabled cadence -> skip.
    assert not recorder.should_record_video(
        _HabitatEnv(),
        step=50,
        next_video_step=50,
        video_log_every=0,
        video_log_episodes=2,
    )
    # Disabled per-cadence episode count -> skip.
    assert not recorder.should_record_video(
        _HabitatEnv(),
        step=50,
        next_video_step=50,
        video_log_every=10,
        video_log_episodes=0,
    )
