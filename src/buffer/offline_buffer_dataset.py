"""Offline buffer dataset for the 3D-26 R2Dreamer ablation.

Loads the canonical offline buffer produced by 3D-25
(scripts/r2dreamer/collect_offline_buffer.py):

    data/offline_buffer/
      trajectory_skeleton.npz   action / reward / done / episode_id
      z_wp_cp.npz               (N, 4116) float16
      z_aggregator.npz          (N, 3072) float16
      collection_metadata.json  status, n_completed_steps, heldout_split, ...
      rollout_log.jsonl         per-episode summaries

Exposes the same `sample(batch_size, seq_len) -> dict` signature as
`src.buffer.replay_buffer.ReplayBuffer`, so the offline training loop can
drop it in wherever the online trainer used `self.buffer`. The split into
train / heldout episodes is driven by `collection_metadata["heldout_split"]`.

Partial collections (status='partial', z_*.npz pre-allocated beyond
n_completed_steps) are sliced down to n_completed_steps on load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np


EncoderKind = Literal["wp_cp", "aggregator", "aggregator_both"]


@dataclass(frozen=True)
class OfflineBufferMetadata:
    n_completed_steps: int
    status: str
    num_episodes: int
    heldout_start_episode: int
    code_sha: str | None
    checkpoint_sha256: str | None
    collect_seed: int


def _resolve_z_path(buffer_dir: Path, encoder_kind: EncoderKind) -> Path:
    if encoder_kind == "wp_cp":
        return buffer_dir / "z_wp_cp.npz"
    if encoder_kind == "aggregator":
        return buffer_dir / "z_aggregator.npz"
    if encoder_kind == "aggregator_both":
        return buffer_dir / "z_aggregator_both.npz"
    raise ValueError(f"unknown encoder_kind: {encoder_kind!r}")


def _load_array(path: Path, key_candidates: tuple[str, ...] = ("features",)) -> np.ndarray:
    data = np.load(path)
    try:
        for key in key_candidates:
            if key in data.files:
                return data[key]
        return data[data.files[0]]
    finally:
        data.close()


def load_offline_buffer_metadata(buffer_dir: Path) -> OfflineBufferMetadata:
    metadata_path = buffer_dir / "collection_metadata.json"
    with metadata_path.open() as f:
        metadata = json.load(f)
    skeleton = np.load(buffer_dir / "trajectory_skeleton.npz")
    try:
        episode_ids = skeleton["episode_id"]
    finally:
        skeleton.close()

    n = int(episode_ids.shape[0])
    n_completed = int(metadata.get("n_completed_steps", n))
    if n_completed > n:
        raise ValueError(
            f"metadata says n_completed_steps={n_completed} but skeleton only "
            f"has {n} rows; rebuild the buffer or correct the metadata"
        )
    num_episodes = int(episode_ids[:n_completed].max()) + 1 if n_completed > 0 else 0
    heldout = metadata.get("heldout_split", {})
    return OfflineBufferMetadata(
        n_completed_steps=n_completed,
        status=metadata.get("status", "completed"),
        num_episodes=num_episodes,
        heldout_start_episode=int(heldout.get("episode_id_start_inclusive", num_episodes)),
        code_sha=metadata.get("code_sha"),
        checkpoint_sha256=metadata.get("checkpoint_sha256"),
        collect_seed=int(metadata.get("collect_seed", -1)),
    )


class OfflineBufferDataset:
    """Loads (obs, action, reward, done) from disk and samples fixed-length sequences.

    Same `sample(batch_size, seq_len)` shape as `ReplayBuffer.sample`. Episode
    boundaries are derived from `done`. Sequences are sampled from a single
    contiguous slice of [start_step, end_step) so the train/heldout split can be
    enforced by selecting the slice once at construction time.

    Sequences may straddle episode boundaries — this matches what
    `ReplayBuffer.sample` does in the online trainer (the `is_first` mask gets
    set wherever `done` flipped on the previous step).

    obs dtype is preserved as float16 in memory (~25-30 GB for a 400k buffer of
    4116-d wp_cp features); the per-batch cast to float32 happens in sample().
    """

    def __init__(
        self,
        buffer_dir: str | Path,
        encoder_kind: EncoderKind,
        *,
        split: Literal["train", "heldout"] = "train",
        seed: int = 0,
    ) -> None:
        buffer_dir = Path(buffer_dir)
        meta = load_offline_buffer_metadata(buffer_dir)
        self.metadata = meta
        self.encoder_kind: EncoderKind = encoder_kind
        self.split = split

        skeleton = np.load(buffer_dir / "trajectory_skeleton.npz")
        try:
            actions = skeleton["action"][: meta.n_completed_steps]
            rewards = skeleton["reward"][: meta.n_completed_steps]
            dones = skeleton["done"][: meta.n_completed_steps]
            episode_ids = skeleton["episode_id"][: meta.n_completed_steps]
        finally:
            skeleton.close()

        z = _load_array(_resolve_z_path(buffer_dir, encoder_kind))[: meta.n_completed_steps]

        if split == "train":
            mask = episode_ids < meta.heldout_start_episode
        elif split == "heldout":
            mask = episode_ids >= meta.heldout_start_episode
        else:
            raise ValueError(f"unknown split: {split!r}")
        if not mask.any():
            raise ValueError(
                f"split={split!r} is empty; heldout_start_episode="
                f"{meta.heldout_start_episode}, total_episodes={meta.num_episodes}"
            )

        indices = np.where(mask)[0]
        start = int(indices[0])
        end = int(indices[-1]) + 1
        if not np.all(np.diff(indices) == 1):
            raise ValueError(
                f"split={split!r} is not a contiguous range of transitions; "
                f"got {len(indices)} indices but they aren't sequential — the "
                f"collector reorders episodes? Aborting to avoid wrap-around bugs."
            )

        self.obs = z[start:end]
        self.actions = actions[start:end]
        self.rewards = rewards[start:end]
        self.dones = dones[start:end]
        self.episode_ids = episode_ids[start:end]
        self.size = int(self.obs.shape[0])
        self.obs_shape: tuple[int, ...] = tuple(self.obs.shape[1:])

        self._rng = np.random.default_rng(seed)

        ep_count = int(self.episode_ids.max()) + 1 - int(self.episode_ids.min()) if self.size else 0
        print(
            f"OfflineBufferDataset[{split}]: {self.size} steps across {ep_count} "
            f"episodes from {buffer_dir} ({encoder_kind}, obs={self.obs.shape}, "
            f"obs_dtype={self.obs.dtype}, status={meta.status})"
        )

    def sample(self, batch_size: int, seq_len: int) -> dict[str, jnp.ndarray]:
        """Return one training batch with the ReplayBuffer.sample interface."""
        if self.size < seq_len:
            raise ValueError(
                f"split has {self.size} steps but seq_len={seq_len}; pick a "
                f"shorter sequence or collect more data"
            )
        n_valid = self.size - seq_len + 1
        starts = self._rng.integers(0, n_valid, size=batch_size)
        indices = starts[:, None] + np.arange(seq_len)[None, :]

        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        is_first = np.zeros_like(dones)
        is_first[:, 0] = True
        is_first[:, 1:] = dones[:, :-1]

        # "terminal" in DreamerV3 is the success-bonus flag. The offline buffer
        # only records env-side `done`; we conservatively map it to is_terminal
        # too. (The online trainer distinguishes via `success>0`; without that
        # signal here, treating done as terminal is the safe default — at worst
        # we slightly over-bootstrap the continuation head.)
        terminals = dones

        return {
            "obs": jnp.asarray(obs, dtype=jnp.float32),
            "actions": jnp.asarray(actions, dtype=jnp.int32),
            "rewards": jnp.asarray(rewards, dtype=jnp.float32),
            "dones": jnp.asarray(dones, dtype=jnp.float32),
            "terminals": jnp.asarray(terminals, dtype=jnp.float32),
            "is_first": jnp.asarray(is_first, dtype=jnp.float32),
        }

    def iter_heldout_batches(self, batch_size: int, seq_len: int, *, max_batches: int | None = None):
        """Deterministic non-overlapping iteration for held-out eval.

        Yields batches by chunking the split into windows of size seq_len from
        a fixed grid; useful for reproducible eval metrics that don't depend on
        the sampler RNG. Returns a generator of batch dicts.
        """
        stride = seq_len
        starts_all = np.arange(0, self.size - seq_len + 1, stride)
        n_total = len(starts_all)
        if max_batches is not None:
            n_total = min(n_total, max_batches * batch_size)
        for chunk_start in range(0, n_total, batch_size):
            starts = starts_all[chunk_start: chunk_start + batch_size]
            if len(starts) == 0:
                break
            indices = starts[:, None] + np.arange(seq_len)[None, :]
            obs = self.obs[indices]
            actions = self.actions[indices]
            rewards = self.rewards[indices]
            dones = self.dones[indices]
            is_first = np.zeros_like(dones)
            is_first[:, 0] = True
            is_first[:, 1:] = dones[:, :-1]
            yield {
                "obs": jnp.asarray(obs, dtype=jnp.float32),
                "actions": jnp.asarray(actions, dtype=jnp.int32),
                "rewards": jnp.asarray(rewards, dtype=jnp.float32),
                "dones": jnp.asarray(dones, dtype=jnp.float32),
                "terminals": jnp.asarray(dones, dtype=jnp.float32),
                "is_first": jnp.asarray(is_first, dtype=jnp.float32),
            }
