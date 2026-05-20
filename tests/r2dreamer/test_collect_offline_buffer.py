import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.r2dreamer.collect_offline_buffer import (
    AGGREGATOR_DIM,
    RGB_SIZE,
    WP_CP_DIM,
    StreamingNpzArray,
    flatten_wp_cp,
    pool_aggregator,
    verify_offline_buffer,
)


def test_flatten_wp_cp_shape_and_dtype():
    out = {
        "world_points": np.ones((37, 37, 3), dtype=np.float32),
        "camera_pose": np.arange(9, dtype=np.float32),
    }

    got = flatten_wp_cp(out)

    assert got.shape == (WP_CP_DIM,)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got[-9:], np.arange(9, dtype=np.float32))


def test_pool_aggregator_keeps_camera_and_pools_patches():
    features = np.zeros((5 + 2, 1024), dtype=np.float32)
    features[0] = 1.0
    features[5] = 2.0
    features[6] = 4.0

    got = pool_aggregator({"aggregator_features": features})

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
    rgb_dir = tmp_path / "rgb_frames"
    rgb_dir.mkdir()
    for i in range(n):
        Image.fromarray(np.zeros((RGB_SIZE, RGB_SIZE, 3), dtype=np.uint8)).save(
            rgb_dir / f"{i:06d}.png"
        )
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
    assert result["rgb_frames"] == n
    assert result["episodes"] == 1
