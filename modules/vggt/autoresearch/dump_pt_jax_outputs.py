"""Dump PyTorch vs JAX VGGT extractor outputs on 10 fixed L1 frames.

Produces three files under output/vggt/parity/:
  input_frames_l1.npz  — cached (10, 3, 518, 518) uint8 RGB
  pt_outputs.npz       — per-frame PT extractor outputs, stacked
  jax_outputs.npz      — per-frame JAX extractor outputs, stacked

Prints a per-key diff summary (maxabs, rms, rel_err).

Notes
-----
- Both extractors (modules.vggt.feature_extractor and
  modules.vggt.jax.feature_extractor) expose a streaming `extract(rgb)` API
  that returns only `world_points` (37, 37, 3) and `camera_pose` (9,).
  They do NOT surface separate depth / extrinsic / intrinsic tensors — the
  camera_pose is a 9-dim pose encoding from camera_head and point_head emits
  3D world points directly. So the `depth`, `extrinsic`, `intrinsic` keys
  requested by the task spec are not available without modifying the
  extractor source (which we are not doing). We dump what's actually there.
- Habitat's action space is {0: STOP, 1: MOVE_FORWARD, 2: TURN_LEFT,
  3: TURN_RIGHT}; there is no index 4 (look_up). The task's proposed action
  4 is remapped to TURN_LEFT so the sequence stays 10 valid steps.
- Habitat rgb sensor here is 518x518 (set via DreamerConfig.obs_shape) so no
  resize is needed at collection time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure repo root importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

CACHE_DIR = _REPO_ROOT / "output" / "vggt" / "parity"
CACHE_INPUT = CACHE_DIR / "input_frames_l1.npz"
PT_DUMP = CACHE_DIR / "pt_outputs.npz"
JAX_DUMP = CACHE_DIR / "jax_outputs.npz"

CURRICULUM_PATH = _REPO_ROOT / "data" / "curriculum" / "level1_1house_1goal.json"
TARGET_SCENE = "fK2vEV32Lag"
TARGET_EPISODE = "23795"

# Habitat discrete action indices (see modules/envs/habitat.py ACTIONS dict):
# 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT. No index 4 — we remap
# the proposed `look_up` slot to TURN_LEFT so the sequence has 10 valid steps.
ACTIONS = [1, 1, 2, 1, 3, 1, 1, 2, 1, 1]
N_FRAMES_DEFAULT = 10


# =========================================================================
# 1) Collect / cache input frames
# =========================================================================

def _collect_frames(n_frames: int) -> np.ndarray:
    """Reset L1 env, step 10 fixed actions, return (N, 3, 518, 518) uint8."""
    from modules.shared.configs import DreamerConfig
    from modules.envs.habitat import HabitatObjectNavEnv

    # 518x518 so VGGT input prep is a no-op (both extractors require 518).
    cfg = DreamerConfig(obs_shape=(3, 518, 518), max_episode_steps=1_000,
                        split="train")

    env = HabitatObjectNavEnv(
        cfg,
        curriculum_path=str(CURRICULUM_PATH),
        curriculum_mode="train",
    )

    # Force target episode. Reduce the episode list to ONLY the target so
    # that habitat's shuffled episode_iterator has no other choice.
    episodes = env._env._dataset.episodes
    target = [ep for ep in episodes if ep.episode_id == TARGET_EPISODE
              and TARGET_SCENE in ep.scene_id]
    assert target, f"episode {TARGET_EPISODE} in {TARGET_SCENE} not found"
    env._env._dataset.episodes = target
    env._env._setup_episode_iterator()
    env._env.current_episode = next(env._env.episode_iterator)

    obs = env.reset()
    confirmed = env._env.current_episode.episode_id
    print(f"[collect] scene={TARGET_SCENE} target_episode={TARGET_EPISODE} "
          f"confirmed={confirmed}")
    assert confirmed == TARGET_EPISODE, (
        f"episode pin failed: wanted {TARGET_EPISODE}, got {confirmed}"
    )
    frames = [obs["image"]]  # (3, 518, 518) uint8
    for i, a in enumerate(ACTIONS[:n_frames - 1]):
        out = env.step(a)
        frames.append(out["image"])
        if out["done"]:
            print(f"[collect] episode done at step {i+1}, padding with last frame")
            while len(frames) < n_frames:
                frames.append(frames[-1].copy())
            break

    env.close()
    arr = np.stack(frames[:n_frames]).astype(np.uint8)
    assert arr.shape == (n_frames, 3, 518, 518), arr.shape
    return arr


def _load_or_collect_frames(n_frames: int, use_cache: bool) -> np.ndarray:
    if use_cache and CACHE_INPUT.exists():
        data = np.load(CACHE_INPUT)
        frames = data["frames"]
        if frames.shape[0] >= n_frames:
            print(f"[cache] using {CACHE_INPUT} ({frames.shape}, {frames.dtype})")
            return frames[:n_frames]
        else:
            print(f"[cache] existing cache only has {frames.shape[0]} frames, "
                  f"re-collecting")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = _collect_frames(n_frames)
    if use_cache:
        np.savez_compressed(CACHE_INPUT, frames=frames)
        print(f"[cache] wrote {CACHE_INPUT} ({frames.shape}, {frames.dtype})")
    return frames


# =========================================================================
# 2) Run extractors
# =========================================================================

def _run_pt(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Run PyTorch streaming extractor on all frames."""
    from modules.vggt.feature_extractor import VGGTFeatureExtractor

    print("[pt] loading VGGTFeatureExtractor...")
    extractor = VGGTFeatureExtractor(device="cuda")
    extractor.reset()

    wps, cps = [], []
    t0 = time.perf_counter()
    for i in range(frames.shape[0]):
        out = extractor.extract(frames[i])
        wps.append(out["world_points"])
        cps.append(out["camera_pose"])
    print(f"[pt] {frames.shape[0]} frames in {time.perf_counter() - t0:.2f}s")

    import torch
    del extractor
    torch.cuda.empty_cache()

    return {
        "world_points": np.stack(wps).astype(np.float32),   # (N, 37, 37, 3)
        "camera_pose":  np.stack(cps).astype(np.float32),   # (N, 9)
    }


def _run_jax(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Run JAX streaming extractor on all frames."""
    from modules.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor

    print("[jax] loading JAXVGGTFeatureExtractor...")
    extractor = JAXVGGTFeatureExtractor(device="cuda")
    extractor.reset()

    wps, cps = [], []
    t0 = time.perf_counter()
    for i in range(frames.shape[0]):
        out = extractor.extract(frames[i])
        wps.append(np.asarray(out["world_points"]))
        cps.append(np.asarray(out["camera_pose"]))
    print(f"[jax] {frames.shape[0]} frames in {time.perf_counter() - t0:.2f}s")

    return {
        "world_points": np.stack(wps).astype(np.float32),
        "camera_pose":  np.stack(cps).astype(np.float32),
    }


# =========================================================================
# 3) Diff summary
# =========================================================================

def _diff_summary(pt: dict[str, np.ndarray], jx: dict[str, np.ndarray]) -> None:
    keys = sorted(set(pt.keys()) & set(jx.keys()))
    print()
    print(f"{'key':<16} {'shape':<22} {'maxabs':>10} {'rms':>10} {'rel_err':>10}")
    print("-" * 72)
    for k in keys:
        a = pt[k].astype(np.float64)
        b = jx[k].astype(np.float64)
        if a.shape != b.shape:
            print(f"{k:<16} shape mismatch: pt={a.shape} jax={b.shape}")
            continue
        diff = a - b
        maxabs = float(np.max(np.abs(diff)))
        rms = float(np.sqrt(np.mean(diff ** 2)))
        denom = float(np.linalg.norm(a))
        rel = float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan")
        shp = str(a.shape)
        print(f"{k:<16} {shp:<22} {maxabs:>10.3e} {rms:>10.3e} {rel:>10.3e}")


# =========================================================================
# 4) CLI
# =========================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--recompute", dest="recompute", action="store_true", default=True)
    p.add_argument("--no-recompute", dest="recompute", action="store_false")
    p.add_argument("--n-frames", type=int, default=N_FRAMES_DEFAULT)
    p.add_argument("--cache-input", dest="cache_input", action="store_true", default=True)
    p.add_argument("--no-cache-input", dest="cache_input", action="store_false")
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.recompute and PT_DUMP.exists() and JAX_DUMP.exists():
        print(f"[load] {PT_DUMP}")
        pt = {k: v for k, v in np.load(PT_DUMP).items()}
        print(f"[load] {JAX_DUMP}")
        jx = {k: v for k, v in np.load(JAX_DUMP).items()}
        _diff_summary(pt, jx)
        return 0

    frames = _load_or_collect_frames(args.n_frames, use_cache=args.cache_input)
    print(f"[frames] shape={frames.shape} dtype={frames.dtype} "
          f"min={frames.min()} max={frames.max()}")

    # Run PT first (frees GPU before JAX which initializes its own allocator).
    pt = _run_pt(frames)
    np.savez_compressed(PT_DUMP, **pt)
    print(f"[pt] wrote {PT_DUMP}")

    jx = _run_jax(frames)
    np.savez_compressed(JAX_DUMP, **jx)
    print(f"[jax] wrote {JAX_DUMP}")

    _diff_summary(pt, jx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
