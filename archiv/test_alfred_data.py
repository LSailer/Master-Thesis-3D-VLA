"""Tests for ALFRED data utilities. Skipped if dataset not downloaded."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alfred_env.data_utils import DATA_ROOT, load_trajectory, iter_dataset, get_dataset_stats

TRAIN_DIR = DATA_ROOT / "train"
skip_no_data = pytest.mark.skipif(
    not TRAIN_DIR.exists(),
    reason="ALFRED data not downloaded — run: bash scripts/download_alfred.sh json",
)


def _first_traj_dir() -> Path:
    """Find first trajectory directory in train split."""
    for task_type_dir in sorted(TRAIN_DIR.iterdir()):
        if not task_type_dir.is_dir():
            continue
        for trial_dir in sorted(task_type_dir.iterdir()):
            if (trial_dir / "traj_data.json").exists():
                return trial_dir
    pytest.skip("No trajectories found in train split")


@skip_no_data
def test_load_trajectory():
    traj = load_trajectory(_first_traj_dir())
    assert traj.task_id
    assert traj.task_type
    assert traj.scene
    assert traj.goal
    assert isinstance(traj.instructions, list)
    assert isinstance(traj.low_actions, list)
    assert len(traj.low_actions) > 0


@skip_no_data
def test_iter_dataset():
    trajs = list(iter_dataset("train"))
    assert len(trajs) > 0
    # Spot-check first trajectory
    t = trajs[0]
    assert t.task_id
    assert t.task_type


@skip_no_data
def test_get_dataset_stats():
    stats = get_dataset_stats("train")
    assert stats.num_trajectories > 0
    assert len(stats.task_type_counts) > 0
    assert len(stats.unique_scenes) > 0
    assert stats.avg_low_actions > 0
    assert stats.avg_instructions > 0


@skip_no_data
def test_iter_dataset_valid_seen():
    valid_dir = DATA_ROOT / "valid_seen"
    if not valid_dir.exists():
        pytest.skip("valid_seen not downloaded")
    trajs = list(iter_dataset("valid_seen"))
    assert len(trajs) > 0


def test_iter_dataset_missing_split():
    with pytest.raises(FileNotFoundError):
        list(iter_dataset("nonexistent_split"))
