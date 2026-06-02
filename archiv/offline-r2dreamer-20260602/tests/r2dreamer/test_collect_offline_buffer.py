import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.r2dreamer.collect_offline_buffer import (
    AGGREGATOR_DIM,
    WP_CP_DIM,
    StreamingNpzArray,
    _atomic_write_metadata,
    _atomic_write_skeleton,
    verify_offline_buffer,
)
from src.r2dreamer.adapters.vggt_adapter import (
    flatten_world_points_camera_pose,
    pool_aggregator_tokens,
)


def test_flatten_wp_cp_shape_and_dtype():
    out = {
        "world_points": np.ones((37, 37, 3), dtype=np.float32),
        "camera_pose": np.arange(9, dtype=np.float32),
    }

    got = np.asarray(flatten_world_points_camera_pose(out), dtype=np.float32)

    assert got.shape == (WP_CP_DIM,)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got[-9:], np.arange(9, dtype=np.float32))


def test_pool_aggregator_keeps_camera_and_pools_patches():
    features = np.zeros((5 + 2, 1024), dtype=np.float32)
    features[0] = 1.0
    features[5] = 2.0
    features[6] = 4.0

    got = np.asarray(
        pool_aggregator_tokens(
            {"aggregator_features": features}, expected_shape=features.shape,
        ),
        dtype=np.float32,
    )

    assert got.shape == (AGGREGATOR_DIM,)
    np.testing.assert_array_equal(got[:1024], np.ones(1024, dtype=np.float32))
    np.testing.assert_array_equal(got[1024:2048], np.full(1024, 3.0, dtype=np.float32))
    np.testing.assert_array_equal(got[2048:], np.full(1024, 4.0, dtype=np.float32))


def test_streaming_npz_array_writes_loadable_npz(tmp_path: Path):
    path = tmp_path / "z.npz"

    with StreamingNpzArray(path, array_name="features", shape=(2, 3), dtype=np.float16) as writer:
        writer.append(np.array([1, 2, 3], dtype=np.float32))
        writer.append(np.array([4, 5, 6], dtype=np.float32))

    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == ["features.npy"]
    data = np.load(path)
    np.testing.assert_array_equal(data["features"], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16))


def test_verify_offline_buffer_accepts_valid_tiny_buffer(tmp_path: Path):
    n = 2
    np.savez(
        tmp_path / "trajectory_skeleton.npz",
        action=np.array([1, 2], dtype=np.int32),
        reward=np.array([0.1, 0.2], dtype=np.float32),
        done=np.array([False, True], dtype=np.bool_),
        episode_id=np.array([0, 0], dtype=np.int32),
    )
    np.savez(tmp_path / "z_wp_cp.npz", features=np.zeros((n, WP_CP_DIM), dtype=np.float16))
    np.savez(
        tmp_path / "z_aggregator.npz",
        features=np.zeros((n, AGGREGATOR_DIM), dtype=np.float16),
    )
    (tmp_path / "rollout_log.jsonl").write_text(
        json.dumps(
            {
                "episode_id": 0,
                "start_step": 0,
                "end_step_exclusive": 2,
                "steps": 2,
                "reward": 0.3,
                "completed": True,
            }
        )
        + "\n"
    )
    (tmp_path / "collection_metadata.json").write_text(
        json.dumps(
            {
                "integrity": {
                    "wp_cp_fp32_vs_fp16_cosine": {"count": 2, "min": 1.0, "mean": 1.0},
                    "aggregator_fp32_vs_fp16_cosine": {
                        "count": 2,
                        "min": 1.0,
                        "mean": 1.0,
                    },
                }
            }
        )
    )

    result = verify_offline_buffer(tmp_path, expected_n_steps=n)

    assert result["n_steps"] == n
    assert result["episodes"] == 1
    assert "rgb_frames" not in result


def _write_partial_z_files(tmp_path: Path, capacity: int) -> None:
    """Write z_*.npz pre-allocated to `capacity` rows (mimics streaming writer)."""
    np.savez(
        tmp_path / "z_wp_cp.npz",
        features=np.zeros((capacity, WP_CP_DIM), dtype=np.float16),
    )
    np.savez(
        tmp_path / "z_aggregator.npz",
        features=np.zeros((capacity, AGGREGATOR_DIM), dtype=np.float16),
    )


def test_atomic_write_skeleton_truncates_to_n_completed(tmp_path: Path):
    capacity = 10
    actions = np.arange(capacity, dtype=np.int32)
    rewards = np.arange(capacity, dtype=np.float32)
    dones = np.zeros(capacity, dtype=np.bool_)
    dones[3] = True
    dones[6] = True
    episode_ids = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)

    _atomic_write_skeleton(tmp_path, actions, rewards, dones, episode_ids, n_completed=7)

    data = np.load(tmp_path / "trajectory_skeleton.npz")
    try:
        np.testing.assert_array_equal(data["action"], np.arange(7, dtype=np.int32))
        np.testing.assert_array_equal(data["done"], dones[:7])
        np.testing.assert_array_equal(data["episode_id"], episode_ids[:7])
    finally:
        data.close()
    assert not (tmp_path / "trajectory_skeleton.npz.tmp").exists()


def test_atomic_write_metadata_replaces_existing(tmp_path: Path):
    _atomic_write_metadata(tmp_path, {"status": "in_progress", "n_completed_steps": 100})
    _atomic_write_metadata(tmp_path, {"status": "completed", "n_completed_steps": 400_000})

    metadata = json.loads((tmp_path / "collection_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["n_completed_steps"] == 400_000
    assert not (tmp_path / "collection_metadata.json.tmp").exists()


def test_verify_offline_buffer_accepts_partial_buffer(tmp_path: Path):
    """A partial buffer has skeleton shorter than z_*.npz capacity but is valid."""
    capacity = 5  # what z_*.npz was pre-allocated to
    n_actual = 3  # what actually got collected before crash
    np.savez(
        tmp_path / "trajectory_skeleton.npz",
        action=np.array([1, 2, 0], dtype=np.int32),
        reward=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        done=np.array([False, False, True], dtype=np.bool_),
        episode_id=np.zeros(n_actual, dtype=np.int32),
    )
    _write_partial_z_files(tmp_path, capacity=capacity)
    (tmp_path / "rollout_log.jsonl").write_text(
        json.dumps(
            {
                "episode_id": 0,
                "start_step": 0,
                "end_step_exclusive": 3,
                "steps": 3,
                "reward": 0.6,
                "completed": True,
            }
        )
        + "\n"
    )
    (tmp_path / "collection_metadata.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "n_completed_steps": n_actual,
                "integrity": {
                    "wp_cp_fp32_vs_fp16_cosine": {"count": 2, "min": 1.0, "mean": 1.0},
                    "aggregator_fp32_vs_fp16_cosine": {
                        "count": 2,
                        "min": 1.0,
                        "mean": 1.0,
                    },
                },
            }
        )
    )

    result = verify_offline_buffer(tmp_path, expected_n_steps=capacity)

    assert result["n_steps"] == n_actual
    assert result["episodes"] == 1
    assert result["z_wp_cp_shape"][0] == capacity


def test_verify_offline_buffer_rejects_completed_with_mismatched_capacity(tmp_path: Path):
    """A 'completed' run with skeleton != z_* shape is still a hard error."""
    n_skeleton = 2
    n_z = 5
    np.savez(
        tmp_path / "trajectory_skeleton.npz",
        action=np.array([1, 2], dtype=np.int32),
        reward=np.array([0.1, 0.2], dtype=np.float32),
        done=np.array([False, True], dtype=np.bool_),
        episode_id=np.zeros(n_skeleton, dtype=np.int32),
    )
    _write_partial_z_files(tmp_path, capacity=n_z)
    (tmp_path / "rollout_log.jsonl").write_text(
        json.dumps(
            {
                "episode_id": 0,
                "start_step": 0,
                "end_step_exclusive": 2,
                "steps": 2,
                "reward": 0.3,
                "completed": True,
            }
        )
        + "\n"
    )
    (tmp_path / "collection_metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "integrity": {
                    "wp_cp_fp32_vs_fp16_cosine": {"count": 2, "min": 1.0, "mean": 1.0},
                    "aggregator_fp32_vs_fp16_cosine": {
                        "count": 2,
                        "min": 1.0,
                        "mean": 1.0,
                    },
                },
            }
        )
    )

    with pytest.raises(AssertionError, match="z_wp_cp shape"):
        verify_offline_buffer(tmp_path)
