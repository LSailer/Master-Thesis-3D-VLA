"""Semantics of the external-R2Dreamer offline buffer adapter (3D-45).

Validates that `OfflineVectorBuffer` (scripts/r2dreamer/train_external_offline.py)
reproduces the JAX `OfflineBufferDataset` sampling contract: the train/heldout
split, contiguous windows, `is_first` reset on `done`, `is_terminal == is_last ==
done`, one-hot actions, and a zeroed initial latent.

These run under the *external* venv (torch + tensordict + gymnasium). Under the
main venv they are skipped — tensordict is not installed there.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "r2dreamer" / "train_external_offline.py"


def _load_module():
    # The external package imports siblings by bare name; put it on the path so
    # the module's lazy `from dreamer import Dreamer` etc. could resolve too.
    sys.path.insert(0, str(REPO_ROOT / "external" / "r2dreamer"))
    spec = importlib.util.spec_from_file_location("train_external_offline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_buffer(tmp: Path, *, ep_lengths, obs_dim=8, heldout_start=None, seed=0):
    """Write a tiny offline buffer (skeleton + z_wp_cp + metadata) to `tmp`."""
    rng = np.random.default_rng(seed)
    actions, rewards, dones, episode_ids = [], [], [], []
    for ep, length in enumerate(ep_lengths):
        actions.append(rng.integers(0, 4, size=length).astype(np.int32))
        rewards.append(rng.standard_normal(length).astype(np.float32))
        d = np.zeros(length, dtype=bool)
        d[-1] = True  # episode terminates on its last step
        dones.append(d)
        episode_ids.append(np.full(length, ep, dtype=np.int32))
    action = np.concatenate(actions)
    reward = np.concatenate(rewards)
    done = np.concatenate(dones)
    episode_id = np.concatenate(episode_ids)
    n = action.shape[0]

    np.savez(tmp / "trajectory_skeleton.npz", action=action, reward=reward, done=done, episode_id=episode_id)
    np.savez(tmp / "z_wp_cp.npz", features=rng.standard_normal((n, obs_dim)).astype(np.float16))

    meta = {"n_completed_steps": n, "num_episodes": len(ep_lengths)}
    if heldout_start is not None:
        meta["heldout_split"] = {"episode_id_start_inclusive": heldout_start}
    (tmp / "collection_metadata.json").write_text(json.dumps(meta))
    return n


def _buf(mod, tmp, *, split, batch_size=4, seq_len=5, obs_dim=8):
    return mod.OfflineVectorBuffer(
        tmp, "wp_cp", split=split,
        batch_size=batch_size, seq_len=seq_len, device="cpu", seed=0,
        deter_size=16, stoch_classes=6, stoch_discrete=3,
    )


def test_split_partitions_episodes(tmp_path):
    mod = _load_module()
    # 5 episodes of 10 steps = 50; heldout = episodes >= 4 (last episode, 10 steps).
    _make_buffer(tmp_path, ep_lengths=[10] * 5, heldout_start=4)
    train = _buf(mod, tmp_path, split="train")
    heldout = _buf(mod, tmp_path, split="heldout")
    assert train.size == 40
    assert heldout.size == 10
    assert train.size + heldout.size == 50


def test_sample_shapes_and_flags(tmp_path):
    import torch

    mod = _load_module()
    _make_buffer(tmp_path, ep_lengths=[10] * 5, heldout_start=4, obs_dim=8)
    buf = _buf(mod, tmp_path, split="train", batch_size=4, seq_len=5, obs_dim=8)
    data, index, initial = buf.sample()

    assert tuple(data.batch_size) == (4, 5)
    assert data["vector"].shape == (4, 5, 8)
    assert data["action"].shape == (4, 5, mod.NUM_ACTIONS)
    assert data["reward"].shape == (4, 5, 1)
    for k in ("is_first", "is_last", "is_terminal"):
        assert data[k].shape == (4, 5, 1)

    # Actions are one-hot.
    assert torch.allclose(data["action"].sum(-1), torch.ones(4, 5))
    # First column of every window is a reset.
    assert bool(data["is_first"][:, 0].all())
    # is_terminal == is_last == done.
    assert torch.equal(data["is_terminal"], data["is_last"])
    # Zeroed initial latent of the right shape.
    s, d = initial
    assert s.shape == (4, 6, 3) and d.shape == (4, 16)
    assert float(s.abs().sum()) == 0.0 and float(d.abs().sum()) == 0.0
    assert index.shape == (4, 5)


def test_is_first_tracks_done_shift(tmp_path):
    mod = _load_module()
    # Two episodes of 8 in a single (all-train) split, so windows straddle the
    # boundary and exercise the interior is_first reset.
    _make_buffer(tmp_path, ep_lengths=[8, 8], heldout_start=99)
    buf = _buf(mod, tmp_path, split="train", batch_size=8, seq_len=5)
    data, index, _ = buf.sample()
    isf = data["is_first"][..., 0].bool().numpy()  # (B, L)

    # Invariant (matches JAX OfflineBufferDataset): is_first[:,0]=True, and
    # is_first[:,t] = done at the previous global step.
    expected = np.zeros_like(isf)
    expected[:, 0] = True
    expected[:, 1:] = buf.dones[index[:, :-1]]
    assert np.array_equal(isf, expected)
    # The boundary at global index 7 must show up as a reset somewhere.
    assert isf[:, 1:].any() or not (index[:, :-1] == 7).any()


def test_missing_metadata_is_all_train(tmp_path):
    mod = _load_module()
    # No heldout_split / no metadata file beyond n_completed -> everything is train.
    n = _make_buffer(tmp_path, ep_lengths=[6, 6, 6], heldout_start=None)
    train = _buf(mod, tmp_path, split="train", seq_len=4)
    assert train.size == n
