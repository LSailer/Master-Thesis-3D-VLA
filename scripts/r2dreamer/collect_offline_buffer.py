"""Collect the canonical offline replay buffer for the VGGT readout ablation.

The collector rolls out a CNN-policy checkpoint, stores the canonical
trajectory skeleton, and computes both VGGT readouts from one VGGT extractor
call per transition. RGB frames are not persisted — PNG encoding dominated the
hot path (~36 % of per-step time) and the downstream ablation only needs the
fp16 readouts.

    python scripts/r2dreamer/collect_offline_buffer.py \
        --checkpoint output/.../step_001000000.pkl \
        --n-steps 500000 \
        --collect-seed 42 \
        --out-dir data/offline_buffer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RGB_SIZE = 518
CNN_SIZE = 64
WP_CP_DIM = 4116
AGGREGATOR_DIM = 3072
PATCH_START_IDX = 5
INTEGRITY_SAMPLE_SIZE = 1000


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resize_chw_uint8(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] != 3:
        raise ValueError(f"expected CHW RGB image, got shape {image.shape}")
    hwc = np.transpose(image, (1, 2, 0))
    resized = Image.fromarray(hwc).resize((size, size), Image.Resampling.BILINEAR)
    return np.transpose(np.asarray(resized, dtype=np.uint8), (2, 0, 1))


def flatten_wp_cp(vggt_out: dict[str, Any]) -> np.ndarray:
    """Flatten VGGT world-points + camera-pose into the canonical 4116 vector."""
    world_points = np.asarray(vggt_out["world_points"], dtype=np.float32).reshape(-1)
    camera_pose = np.asarray(vggt_out["camera_pose"], dtype=np.float32).reshape(-1)
    out = np.concatenate([world_points, camera_pose], axis=0)
    if out.shape != (WP_CP_DIM,):
        raise ValueError(f"expected WP/CP shape ({WP_CP_DIM},), got {out.shape}")
    return out


def pool_aggregator(vggt_out: dict[str, Any]) -> np.ndarray:
    """Pool VGGT pre-head aggregator tokens into [camera | mean patches | max patches]."""
    features = np.asarray(vggt_out["aggregator_features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] <= PATCH_START_IDX:
        raise ValueError(
            "expected aggregator_features with shape "
            f"(tokens>{PATCH_START_IDX}, dim), got {features.shape}"
        )
    cam = features[0]
    patches = features[PATCH_START_IDX:]
    out = np.concatenate([cam, patches.mean(axis=0), patches.max(axis=0)], axis=0)
    if out.shape != (AGGREGATOR_DIM,):
        raise ValueError(
            f"expected aggregator pooled shape ({AGGREGATOR_DIM},), got {out.shape}"
        )
    return out.astype(np.float32, copy=False)


def extract_dual_readouts(
    extractor: Any,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both issue-required readouts from a single VGGT extractor call."""
    out = extractor.extract(image)
    return flatten_wp_cp(out), pool_aggregator(out)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a32 = np.asarray(a, dtype=np.float32).reshape(-1)
    b32 = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a32) * np.linalg.norm(b32))
    if denom == 0.0:
        return 1.0 if np.array_equal(a32, b32) else 0.0
    return float(np.dot(a32, b32) / denom)


class StreamingNpzArray:
    """Write a single-array npz incrementally without keeping it in RAM."""

    def __init__(
        self,
        path: Path,
        *,
        array_name: str,
        shape: tuple[int, int],
        dtype: np.dtype | type,
    ) -> None:
        self.path = path
        self.array_name = array_name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.index = 0
        self._zip = zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED)
        self._fp = self._zip.open(f"{array_name}.npy", mode="w", force_zip64=True)
        header = {
            "descr": np.lib.format.dtype_to_descr(self.dtype),
            "fortran_order": False,
            "shape": shape,
        }
        np.lib.format.write_array_header_2_0(self._fp, header)

    def append(self, row: np.ndarray) -> None:
        if self.index >= self.shape[0]:
            raise IndexError(f"{self.path} already has {self.shape[0]} rows")
        arr = np.asarray(row, dtype=self.dtype)
        expected = self.shape[1:]
        if arr.shape != expected:
            raise ValueError(f"expected row shape {expected}, got {arr.shape}")
        self._fp.write(np.ascontiguousarray(arr).tobytes(order="C"))
        self.index += 1

    def close(self) -> None:
        try:
            self._fp.close()
        finally:
            self._zip.close()
        if self.index != self.shape[0]:
            raise RuntimeError(
                f"{self.path} has {self.index} rows, expected {self.shape[0]}"
            )

    def abort(self) -> None:
        self._fp.close()
        self._zip.close()

    def __enter__(self) -> "StreamingNpzArray":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._fp.close()
            self._zip.close()


class StepTimer:
    """Accumulate per-component wall-clock timings during the collection loop.

    The first ``warmup`` steps are dropped from the means so JAX compile, the
    first habitat scene swap, and other one-shot costs don't dominate.
    """

    PHASES = (
        "vggt_extract",
        "fp16_cast",
        "npz_append",
        "resize",
        "agent_act",
        "env_step",
        "bookkeeping",
    )

    def __init__(self, warmup: int = 100) -> None:
        self.warmup = warmup
        self.sums = {p: 0.0 for p in self.PHASES}
        self.step = 0
        self._t: float | None = None

    def start(self) -> None:
        self._t = time.perf_counter()

    def lap(self, phase: str) -> None:
        if self._t is None:
            raise RuntimeError("StepTimer.start() not called")
        now = time.perf_counter()
        if self.step >= self.warmup:
            self.sums[phase] += now - self._t
        self._t = now

    def end_step(self) -> None:
        self.step += 1

    def summary(self) -> dict[str, Any]:
        active = max(self.step - self.warmup, 0)
        if active <= 0:
            return {"active_steps": 0, "warmup_steps": self.warmup}
        total = sum(self.sums.values())
        components_ms = {p: 1000.0 * self.sums[p] / active for p in self.PHASES}
        components_pct = {
            p: (100.0 * self.sums[p] / total) if total > 0 else 0.0
            for p in self.PHASES
        }
        return {
            "warmup_steps": self.warmup,
            "active_steps": active,
            "per_step_ms": 1000.0 * total / active,
            "components_ms": components_ms,
            "components_pct": components_pct,
        }


@dataclass
class IntegrityAccumulator:
    sample_indices: set[int]
    wp_cosines: list[float]
    agg_cosines: list[float]

    @classmethod
    def for_steps(cls, n_steps: int, seed: int) -> "IntegrityAccumulator":
        sample_n = min(INTEGRITY_SAMPLE_SIZE, n_steps)
        rng = np.random.default_rng(seed)
        indices = set(int(i) for i in rng.choice(n_steps, size=sample_n, replace=False))
        return cls(sample_indices=indices, wp_cosines=[], agg_cosines=[])

    def maybe_record(
        self,
        step: int,
        wp_fp32: np.ndarray,
        agg_fp32: np.ndarray,
        wp_fp16: np.ndarray,
        agg_fp16: np.ndarray,
    ) -> None:
        if step not in self.sample_indices:
            return
        self.wp_cosines.append(cosine_similarity(wp_fp32, wp_fp16.astype(np.float32)))
        self.agg_cosines.append(cosine_similarity(agg_fp32, agg_fp16.astype(np.float32)))

    def summary(self) -> dict[str, Any]:
        def stats(values: list[float]) -> dict[str, float | int | None]:
            if not values:
                return {"count": 0, "min": None, "mean": None}
            arr = np.asarray(values, dtype=np.float64)
            return {
                "count": int(arr.size),
                "min": float(arr.min()),
                "mean": float(arr.mean()),
            }

        return {
            "sample_size": len(self.sample_indices),
            "wp_cp_fp32_vs_fp16_cosine": stats(self.wp_cosines),
            "aggregator_fp32_vs_fp16_cosine": stats(self.agg_cosines),
        }


def _load_npz_array(path: Path) -> np.ndarray:
    data = np.load(path)
    try:
        key = "features" if "features" in data.files else data.files[0]
        return data[key]
    finally:
        data.close()


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total


def _missing_pickle_class(module: str, name: str) -> type:
    class MissingPickleClass:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __setstate__(self, state):
            self.state = state

    MissingPickleClass.__name__ = name
    MissingPickleClass.__module__ = module
    return MissingPickleClass


class _CheckpointUnpickler(pickle.Unpickler):
    """Load old checkpoints even when unused optimizer-state classes moved."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return _missing_pickle_class(module, name)


def load_policy_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        ckpt = _CheckpointUnpickler(f).load()
    missing = {"params", "slow_critic_params"} - set(ckpt)
    if missing:
        raise KeyError(f"checkpoint {path} is missing required keys: {sorted(missing)}")
    return ckpt


def verify_offline_buffer(
    out_dir: Path,
    *,
    expected_n_steps: int | None = None,
    min_cosine: float = 0.999,
) -> dict[str, Any]:
    """Verify row counts, dtypes, episode boundaries, and cosine stats."""
    skeleton_path = out_dir / "trajectory_skeleton.npz"
    wp_path = out_dir / "z_wp_cp.npz"
    agg_path = out_dir / "z_aggregator.npz"
    metadata_path = out_dir / "collection_metadata.json"
    rollout_log_path = out_dir / "rollout_log.jsonl"

    skeleton = np.load(skeleton_path)
    try:
        actions = skeleton["action"]
        rewards = skeleton["reward"]
        done = skeleton["done"]
        episode_id = skeleton["episode_id"]
    finally:
        skeleton.close()

    n = int(actions.shape[0])
    if expected_n_steps is not None and n != expected_n_steps:
        raise AssertionError(f"skeleton has {n} rows, expected {expected_n_steps}")
    if actions.dtype != np.int32:
        raise AssertionError(f"action dtype is {actions.dtype}, expected int32")
    if rewards.dtype != np.float32:
        raise AssertionError(f"reward dtype is {rewards.dtype}, expected float32")
    if done.dtype != np.bool_:
        raise AssertionError(f"done dtype is {done.dtype}, expected bool")
    if episode_id.dtype != np.int32:
        raise AssertionError(f"episode_id dtype is {episode_id.dtype}, expected int32")

    wp = _load_npz_array(wp_path)
    agg = _load_npz_array(agg_path)
    if wp.shape != (n, WP_CP_DIM):
        raise AssertionError(f"z_wp_cp shape is {wp.shape}, expected {(n, WP_CP_DIM)}")
    if agg.shape != (n, AGGREGATOR_DIM):
        raise AssertionError(
            f"z_aggregator shape is {agg.shape}, expected {(n, AGGREGATOR_DIM)}"
        )
    if wp.dtype != np.float16:
        raise AssertionError(f"z_wp_cp dtype is {wp.dtype}, expected float16")
    if agg.dtype != np.float16:
        raise AssertionError(f"z_aggregator dtype is {agg.dtype}, expected float16")

    with metadata_path.open() as f:
        metadata = json.load(f)
    integrity = metadata.get("integrity", {})
    for key in ("wp_cp_fp32_vs_fp16_cosine", "aggregator_fp32_vs_fp16_cosine"):
        min_value = integrity.get(key, {}).get("min")
        if min_value is None:
            raise AssertionError(f"missing integrity cosine stats for {key}")
        if float(min_value) <= min_cosine:
            raise AssertionError(f"{key} min={min_value} <= {min_cosine}")

    log_entries = []
    with rollout_log_path.open() as f:
        for line in f:
            if line.strip():
                log_entries.append(json.loads(line))
    for entry in log_entries:
        if not entry.get("completed", False):
            continue
        end_step = int(entry["end_step_exclusive"]) - 1
        if end_step < 0 or end_step >= n or not bool(done[end_step]):
            raise AssertionError(
                f"completed episode {entry['episode_id']} does not end on done=True"
            )
        start = int(entry["start_step"])
        if start < end_step and bool(done[start:end_step].any()):
            raise AssertionError(
                f"episode {entry['episode_id']} has an early done=True marker"
            )

    total_size = _directory_size_bytes(out_dir)
    return {
        "n_steps": n,
        "z_wp_cp_shape": list(wp.shape),
        "z_aggregator_shape": list(agg.shape),
        "total_size_bytes": total_size,
        "total_size_gib": total_size / (1024 ** 3),
        "episodes": len(log_entries),
        "cosine_min_wp_cp": integrity["wp_cp_fp32_vs_fp16_cosine"]["min"],
        "cosine_min_aggregator": integrity["aggregator_fp32_vs_fp16_cosine"]["min"],
    }


def _build_agent(checkpoint_path: Path, seed: int) -> tuple[Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    from src.r2dreamer.agent import R2DreamerAgent
    from src.r2dreamer.config import R2DreamerConfig

    cfg = R2DreamerConfig(obs_shape=(3, CNN_SIZE, CNN_SIZE), num_actions=4)
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    agent = R2DreamerAgent(cfg, init_key)
    ckpt = load_policy_checkpoint(checkpoint_path)
    agent.params = jax.tree.map(jnp.asarray, ckpt["params"])
    agent.slow_critic_params = jax.tree.map(jnp.asarray, ckpt["slow_critic_params"])
    return agent, ckpt


def _seed_env(env: Any, seed: int) -> None:
    raw = getattr(env, "_env", None)
    if raw is not None and hasattr(raw, "seed"):
        raw.seed(seed)


def _open_wandb(args: argparse.Namespace, metadata: dict[str, Any]):
    if args.wandb_project is None:
        return None, None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
        config=metadata,
        settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
    )
    return wandb, run


def collect_offline_buffer(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    from src.environments.habitat import HabitatObjectNavEnv
    from src.shared.configs import DreamerConfig
    from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive")

    random.seed(args.collect_seed)
    np.random.seed(args.collect_seed)

    code_sha = _git_output("rev-parse", "HEAD")
    metadata: dict[str, Any] = {
        "issue": "3D-25",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "code_sha": code_sha,
        "code_dirty": bool(_git_output("status", "--porcelain")),
        "collect_seed": args.collect_seed,
        "n_steps": args.n_steps,
        "render_resolution": RGB_SIZE,
        "cnn_policy_resolution": CNN_SIZE,
        "split": args.split,
        "curriculum_path": args.curriculum_path,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at_unix": time.time(),
    }

    wandb_module, wandb_run = _open_wandb(args, metadata)
    if wandb_run is not None:
        metadata["wandb_run_id"] = wandb_run.id
        metadata["wandb_run_url"] = getattr(wandb_run, "url", None)
    else:
        metadata["wandb_run_id"] = None
        metadata["wandb_run_url"] = None

    agent, ckpt = _build_agent(checkpoint_path, args.collect_seed)
    metadata["checkpoint_step"] = int(ckpt.get("step", -1))

    env_cfg = DreamerConfig(
        obs_shape=(3, RGB_SIZE, RGB_SIZE),
        max_episode_steps=args.max_episode_steps,
        split=args.split,
        reward_type="geodesic_delta",
    )
    env = HabitatObjectNavEnv(
        env_cfg,
        semantic=args.semantic,
        curriculum_path=args.curriculum_path,
        curriculum_mode=args.curriculum_mode,
        seed=args.collect_seed,
    )
    _seed_env(env, args.collect_seed)

    extractor = JAXVGGTFeatureExtractor(
        total_budget=args.vggt_total_budget,
        budgets_static=tuple([args.vggt_static_budget] * 24),
        compute_heads=True,
    )

    actions = np.zeros((args.n_steps,), dtype=np.int32)
    rewards = np.zeros((args.n_steps,), dtype=np.float32)
    dones = np.zeros((args.n_steps,), dtype=np.bool_)
    episode_ids = np.zeros((args.n_steps,), dtype=np.int32)
    integrity = IntegrityAccumulator.for_steps(args.n_steps, args.collect_seed)

    rollout_log_path = out_dir / "rollout_log.jsonl"
    rng_key = jax.random.PRNGKey(args.collect_seed)
    episode_id = 0
    episode_start_step = 0
    episode_reward = 0.0
    episode_steps = 0
    obs = env.reset()
    extractor.reset()

    wp_writer = StreamingNpzArray(
        out_dir / "z_wp_cp.npz",
        array_name="features",
        shape=(args.n_steps, WP_CP_DIM),
        dtype=np.float16,
    )
    agg_writer = StreamingNpzArray(
        out_dir / "z_aggregator.npz",
        array_name="features",
        shape=(args.n_steps, AGGREGATOR_DIM),
        dtype=np.float16,
    )

    timer = StepTimer(warmup=args.profile_warmup) if args.profile else None

    completed_writes = False
    try:
        with rollout_log_path.open("w") as rollout_log:
            for step in range(args.n_steps):
                if timer is not None:
                    timer.start()
                rgb = obs["image"]
                wp_fp32, agg_fp32 = extract_dual_readouts(extractor, rgb)
                if timer is not None:
                    timer.lap("vggt_extract")
                wp_fp16 = wp_fp32.astype(np.float16)
                agg_fp16 = agg_fp32.astype(np.float16)
                integrity.maybe_record(step, wp_fp32, agg_fp32, wp_fp16, agg_fp16)
                if timer is not None:
                    timer.lap("fp16_cast")
                wp_writer.append(wp_fp16)
                agg_writer.append(agg_fp16)
                if timer is not None:
                    timer.lap("npz_append")

                cnn_obs = {
                    "image": _resize_chw_uint8(rgb, CNN_SIZE),
                    "is_first": bool(obs.get("is_first", False)),
                }
                if timer is not None:
                    timer.lap("resize")
                rng_key, act_key = jax.random.split(rng_key)
                action = int(agent.act(cnn_obs, act_key, training=False))
                if timer is not None:
                    timer.lap("agent_act")
                next_obs = env.step(action)
                if timer is not None:
                    timer.lap("env_step")

                reward = float(next_obs["reward"])
                done = bool(next_obs["done"])
                actions[step] = action
                rewards[step] = reward
                dones[step] = done
                episode_ids[step] = episode_id
                episode_reward += reward
                episode_steps += 1

                if wandb_module is not None and (step + 1) % args.wandb_log_every == 0:
                    wandb_module.log(
                        {
                            "collect/step": step + 1,
                            "collect/episode_id": episode_id,
                            "collect/episode_steps": episode_steps,
                        },
                        step=step + 1,
                    )

                if done:
                    entry = {
                        "episode_id": episode_id,
                        "start_step": episode_start_step,
                        "end_step_exclusive": step + 1,
                        "steps": episode_steps,
                        "reward": episode_reward,
                        "completed": True,
                    }
                    rollout_log.write(json.dumps(entry) + "\n")
                    rollout_log.flush()
                    episode_id += 1
                    episode_start_step = step + 1
                    episode_reward = 0.0
                    episode_steps = 0
                    obs = env.reset()
                    extractor.reset()
                else:
                    obs = next_obs

                if (step + 1) % args.log_every == 0:
                    print(
                        f"collected {step + 1}/{args.n_steps} transitions "
                        f"(episode {episode_id}, current_ep_steps={episode_steps})",
                        flush=True,
                    )

                if timer is not None:
                    timer.lap("bookkeeping")
                    timer.end_step()

            if episode_steps > 0:
                rollout_log.write(
                    json.dumps(
                        {
                            "episode_id": episode_id,
                            "start_step": episode_start_step,
                            "end_step_exclusive": args.n_steps,
                            "steps": episode_steps,
                            "reward": episode_reward,
                            "completed": False,
                        }
                    )
                    + "\n"
                )
        completed_writes = True
    finally:
        if completed_writes:
            wp_writer.close()
            agg_writer.close()
        else:
            wp_writer.abort()
            agg_writer.abort()
        env.close()

    np.savez(
        out_dir / "trajectory_skeleton.npz",
        action=actions,
        reward=rewards,
        done=dones,
        episode_id=episode_ids,
    )

    num_episodes = int(episode_ids.max()) + 1 if len(episode_ids) else 0
    heldout_count = int(np.ceil(num_episodes * 0.10)) if num_episodes else 0
    heldout_start = max(0, num_episodes - heldout_count)
    metadata.update(
        {
            "completed_at_unix": time.time(),
            "num_episodes": num_episodes,
            "heldout_split": {
                "rule": "last_10_percent_of_episodes",
                "episode_id_start_inclusive": heldout_start,
                "episode_id_end_exclusive": num_episodes,
                "num_episodes": heldout_count,
            },
            "integrity": integrity.summary(),
        }
    )
    if timer is not None:
        profile = timer.summary()
        metadata["profile"] = profile
        print("profile=" + json.dumps(profile, indent=2), flush=True)
    (out_dir / "collection_metadata.json").write_text(json.dumps(metadata, indent=2))

    verification = verify_offline_buffer(out_dir, expected_n_steps=args.n_steps)
    metadata["verification"] = verification
    (out_dir / "collection_metadata.json").write_text(json.dumps(metadata, indent=2))

    if wandb_module is not None:
        wandb_module.log({"collect/verification": verification})
        wandb_module.finish()

    print(json.dumps(verification, indent=2))
    return verification


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="CNN baseline checkpoint path")
    parser.add_argument("--n-steps", type=int, required=True, help="Number of transitions")
    parser.add_argument("--collect-seed", type=int, required=True, help="Rollout seed")
    parser.add_argument("--out-dir", required=True, help="Output buffer directory")
    parser.add_argument("--verify-only", action="store_true", help="Only verify an existing buffer")
    parser.add_argument("--split", default="train", help="Habitat split for collection")
    parser.add_argument("--curriculum-path", default=None, help="Optional curriculum JSON")
    parser.add_argument("--curriculum-mode", default="train", help="Curriculum episode key")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default="offline-buffer,3d-25")
    parser.add_argument("--wandb-log-every", type=int, default=1000)
    parser.add_argument("--wandb-init-timeout", type=int, default=600)
    parser.add_argument("--vggt-total-budget", type=int, default=200_000)
    parser.add_argument("--vggt-static-budget", type=int, default=8333)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Time each loop component (vggt/png/env/...) and report a per-step breakdown.",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=100,
        help="Drop this many initial steps from the profile means (JAX compile, first scene swap).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.verify_only:
        result = verify_offline_buffer(out_dir, expected_n_steps=args.n_steps)
        print(json.dumps(result, indent=2))
        return
    collect_offline_buffer(args)


if __name__ == "__main__":
    main()
