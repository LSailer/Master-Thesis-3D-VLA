"""Tests for OfflineBufferDataset (3D-26).

Builds a tiny synthetic buffer on disk and exercises the train/heldout split,
sample(), and iter_heldout_batches(). No GPU / no JAX-jit needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.buffer.offline_buffer_dataset import (
    OfflineBufferDataset,
    load_offline_buffer_metadata,
)


def _make_synthetic_buffer(
    tmp_path: Path,
    *,
    n_steps: int = 80,
    n_episodes: int = 5,
    heldout_episodes: int = 1,
    feature_dim: int = 16,
    status: str = "completed",
    capacity: int | None = None,
) -> Path:
    """Write a complete or partial synthetic buffer to tmp_path/<status>/."""
    out_dir = tmp_path / status
    out_dir.mkdir(parents=True, exist_ok=True)

    capacity = capacity if capacity is not None else n_steps
    actions = np.arange(n_steps, dtype=np.int32) % 4
    rewards = np.arange(n_steps, dtype=np.float32) / 100.0
    dones = np.zeros(n_steps, dtype=np.bool_)
    episode_ids = np.zeros(n_steps, dtype=np.int32)

    boundaries = np.linspace(0, n_steps, n_episodes + 1, dtype=np.int32)
    for ep in range(n_episodes):
        start, end = int(boundaries[ep]), int(boundaries[ep + 1])
        episode_ids[start:end] = ep
        dones[end - 1] = True

    np.savez(
        out_dir / "trajectory_skeleton.npz",
        action=actions, reward=rewards, done=dones, episode_id=episode_ids,
    )
    # Synthetic z_*.npz: each row encodes its absolute step index, so we can
    # verify sampling pulls the right slice.
    rows = np.broadcast_to(
        np.arange(capacity, dtype=np.float16)[:, None], (capacity, feature_dim),
    )
    np.savez(out_dir / "z_wp_cp.npz", features=rows.astype(np.float16))
    np.savez(out_dir / "z_aggregator.npz", features=rows.astype(np.float16))

    heldout_start = n_episodes - heldout_episodes
    metadata = {
        "status": status,
        "n_completed_steps": n_steps,
        "num_episodes": n_episodes,
        "heldout_split": {
            "rule": "last_10_percent_of_episodes",
            "episode_id_start_inclusive": heldout_start,
            "episode_id_end_exclusive": n_episodes,
            "num_episodes": heldout_episodes,
        },
        "code_sha": "deadbeef",
        "checkpoint_sha256": "cafe",
        "collect_seed": 42,
    }
    (out_dir / "collection_metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_dir


def test_load_offline_buffer_metadata(tmp_path: Path):
    out = _make_synthetic_buffer(tmp_path, n_steps=40, n_episodes=4, heldout_episodes=1)
    meta = load_offline_buffer_metadata(out)
    assert meta.n_completed_steps == 40
    assert meta.num_episodes == 4
    assert meta.heldout_start_episode == 3


def test_train_split_excludes_heldout_episodes(tmp_path: Path):
    out = _make_synthetic_buffer(
        tmp_path, n_steps=40, n_episodes=4, heldout_episodes=1, feature_dim=8,
    )
    train = OfflineBufferDataset(out, encoder_kind="wp_cp", split="train", seed=0)
    heldout = OfflineBufferDataset(out, encoder_kind="wp_cp", split="heldout", seed=0)

    assert set(train.episode_ids.tolist()) == {0, 1, 2}
    assert set(heldout.episode_ids.tolist()) == {3}
    assert train.size + heldout.size == 40


def test_sample_returns_correct_shapes_and_is_first(tmp_path: Path):
    out = _make_synthetic_buffer(
        tmp_path, n_steps=40, n_episodes=4, heldout_episodes=1, feature_dim=8,
    )
    train = OfflineBufferDataset(out, encoder_kind="wp_cp", split="train", seed=7)
    batch = train.sample(batch_size=3, seq_len=5)

    assert batch["obs"].shape == (3, 5, 8)
    assert batch["actions"].shape == (3, 5)
    assert batch["rewards"].shape == (3, 5)
    assert batch["dones"].shape == (3, 5)
    assert batch["is_first"].shape == (3, 5)
    np.testing.assert_array_equal(batch["is_first"][:, 0], np.ones(3, dtype=np.float32))
    # is_first[t] should be dones[t-1] for t>0.
    np.testing.assert_array_equal(
        np.asarray(batch["is_first"])[:, 1:],
        np.asarray(batch["dones"])[:, :-1],
    )


def test_sample_pulls_from_split_slice_only(tmp_path: Path):
    """Since our synthetic obs encodes the absolute step index, the heldout
    sampler should only return obs in [heldout_start, n_steps)."""
    out = _make_synthetic_buffer(
        tmp_path, n_steps=40, n_episodes=4, heldout_episodes=1, feature_dim=8,
    )
    heldout = OfflineBufferDataset(out, encoder_kind="wp_cp", split="heldout", seed=0)
    heldout_step_min = int(np.where(np.load(out / "trajectory_skeleton.npz")["episode_id"] == 3)[0][0])

    for _ in range(20):
        batch = heldout.sample(batch_size=2, seq_len=3)
        # Encoded step indices live in obs[..., 0].
        step_vals = np.asarray(batch["obs"])[..., 0].astype(np.int32)
        assert int(step_vals.min()) >= heldout_step_min
        assert int(step_vals.max()) < 40


def test_partial_collection_truncates_z(tmp_path: Path):
    """For status='partial', z_*.npz has more rows than skeleton — we slice."""
    out = _make_synthetic_buffer(
        tmp_path, n_steps=20, n_episodes=4, heldout_episodes=1, feature_dim=4,
        status="partial", capacity=50,
    )
    train = OfflineBufferDataset(out, encoder_kind="wp_cp", split="train", seed=0)
    # train slice = episodes 0..2 = first 15 steps (boundaries 0,5,10,15,20)
    assert train.size == 15
    assert train.obs.shape == (15, 4)


def test_invalid_encoder_kind_raises(tmp_path: Path):
    out = _make_synthetic_buffer(tmp_path, n_steps=20, n_episodes=2, heldout_episodes=1)
    with pytest.raises(ValueError, match="unknown encoder_kind"):
        OfflineBufferDataset(out, encoder_kind="bogus", split="train")  # type: ignore[arg-type]


def test_iter_heldout_batches_is_deterministic(tmp_path: Path):
    out = _make_synthetic_buffer(
        tmp_path, n_steps=40, n_episodes=4, heldout_episodes=1, feature_dim=4,
    )
    heldout = OfflineBufferDataset(out, encoder_kind="wp_cp", split="heldout", seed=0)
    batches_a = [np.asarray(b["obs"]).copy() for b in heldout.iter_heldout_batches(2, 3)]
    batches_b = [np.asarray(b["obs"]).copy() for b in heldout.iter_heldout_batches(2, 3)]
    assert len(batches_a) == len(batches_b)
    for a, b in zip(batches_a, batches_b):
        np.testing.assert_array_equal(a, b)
