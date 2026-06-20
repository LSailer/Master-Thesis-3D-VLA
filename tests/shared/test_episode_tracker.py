"""Unit tests for EpisodeTracker — per-category accumulators + collision-rate flag."""

from typing import TypedDict

import pytest

from src.shared.wandb_utils import EpisodeTracker


class EpisodeKwargs(TypedDict):
    reward: float
    success: float
    spl: float
    category: str
    scene_id: str


def _make_episode(
    category: str, spl: float, success: float = 0.0, reward: float = 0.0
) -> EpisodeKwargs:
    return EpisodeKwargs(
        reward=reward, success=success, spl=spl,
        category=category, scene_id="some/scene.basis.glb",
    )


def test_per_category_spl_emitted_alongside_per_category_sr():
    tracker = EpisodeTracker(window=100)
    out = tracker.record(**_make_episode("chair", spl=0.8, success=1.0))
    assert out["goal/chair/sr"] == pytest.approx(1.0)
    assert out["goal/chair/spl"] == pytest.approx(0.8)
    assert out["goal/chair/reward"] == pytest.approx(0.0)


def test_per_category_spl_isolated_across_categories():
    tracker = EpisodeTracker(window=100)
    tracker.record(**_make_episode("chair", spl=1.0, success=1.0))
    tracker.record(**_make_episode("chair", spl=0.0, success=0.0))
    out = tracker.record(**_make_episode("plant", spl=0.5, success=1.0))
    # chair: mean(1.0, 0.0) = 0.5
    assert out["goal/chair/spl"] == pytest.approx(0.5)
    # plant: single entry
    assert out["goal/plant/spl"] == pytest.approx(0.5)


def test_per_category_spl_uses_rolling_window():
    tracker = EpisodeTracker(window=3)
    tracker.record(**_make_episode("chair", spl=0.0))
    tracker.record(**_make_episode("chair", spl=0.0))
    tracker.record(**_make_episode("chair", spl=0.0))
    # 4th entry pushes the oldest 0.0 out of the rolling window
    out = tracker.record(**_make_episode("chair", spl=1.0))
    # window now: [0.0, 0.0, 1.0] -> mean = 1/3
    assert out["goal/chair/spl"] == pytest.approx(1.0 / 3.0)


def test_success_rate_logs_last_rolling_and_cumulative_mean():
    tracker = EpisodeTracker(window=2)
    tracker.record(**_make_episode("chair", spl=0.0, success=1.0))
    tracker.record(**_make_episode("chair", spl=0.0, success=0.0))
    out = tracker.record(**_make_episode("chair", spl=0.0, success=0.0))

    assert out["episode/success"] == pytest.approx(0.0)
    assert out["metrics/sr_last"] == pytest.approx(0.0)
    # Existing training curve: rolling window over the last two episodes.
    assert out["metrics/sr"] == pytest.approx(0.0)
    # Paper/report summary: cumulative mean over all completed episodes.
    assert out["metrics/sr_mean"] == pytest.approx(1.0 / 3.0)


def test_cumulative_metric_mean_is_not_limited_by_rolling_window():
    tracker = EpisodeTracker(window=2)
    tracker.record(**_make_episode("chair", reward=1.0, spl=1.0))
    tracker.record(**_make_episode("chair", reward=1.0, spl=1.0))
    out = tracker.record(**_make_episode("chair", reward=4.0, spl=0.0))

    assert out["metrics/reward"] == pytest.approx(2.5)
    assert out["metrics/reward_mean"] == pytest.approx(2.0)
    assert out["metrics/spl"] == pytest.approx(0.5)
    assert out["metrics/spl_mean"] == pytest.approx(2.0 / 3.0)


def test_collision_rate_only_emitted_when_flag_enabled():
    train_tracker = EpisodeTracker(window=100, track_collision_rate=False)
    val_tracker = EpisodeTracker(window=100, track_collision_rate=True)

    train_out = train_tracker.record(
        reward=0.0, success=0.0, spl=0.0,
        category="chair", scene_id="x.basis.glb",
        collision_rate=0.3,
    )
    val_out = val_tracker.record(
        reward=0.0, success=0.0, spl=0.0,
        category="chair", scene_id="x.basis.glb",
        collision_rate=0.3,
    )

    assert "episode/collision_rate" not in train_out
    assert "metrics/collision_rate" not in train_out
    assert val_out["episode/collision_rate"] == pytest.approx(0.3)
    assert val_out["metrics/collision_rate"] == pytest.approx(0.3)


def test_softspl_and_dtg_emitted_for_every_episode():
    tracker = EpisodeTracker(window=100)
    out = tracker.record(
        reward=1.0, success=1.0, spl=1.0,
        category="bed", scene_id="x.basis.glb",
        softspl=0.7, dtg=0.1,
    )
    assert out["episode/softspl"] == pytest.approx(0.7)
    assert out["episode/dtg"] == pytest.approx(0.1)
    assert out["metrics/softspl"] == pytest.approx(0.7)
    assert out["metrics/dtg"] == pytest.approx(0.1)
