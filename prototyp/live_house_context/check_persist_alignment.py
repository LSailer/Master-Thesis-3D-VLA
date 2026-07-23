"""Definitively test whether ResetMode.PERSIST_SCENE preserves the VGGT world
frame across an episode boundary — with all environment confounds removed.

The first version of this diagnostic compared per-episode clouds from the L1
curriculum, but every ``env.reset()`` loads a *different* episode (different
spawn), so low IoU / high %new was expected regardless of frame consistency
(assumption A1 was wrong). This version removes that confound entirely:

  1. Record ``total`` observation frames from ONE env episode (no reset).
  2. Continuous pass (PERSIST extractor, no mid-stream reset): feed all frames
     with ``is_first=True`` only on frame 0; record ``world_points`` for the
     last ``tail`` frames -> ``wp_continuous``.
  3. Reset+restore pass (same PERSIST extractor, fresh): feed frames
     ``0..split-1``; call ``reset_for_scene(scene_id)`` (saves the cache, the
     episode-boundary operation); feed the SAME tail frames with
     ``is_first=True`` on the first tail frame (the boundary signal that
     triggers the restore); record ``world_points`` -> ``wp_reset``.
  4. Compare ``wp_continuous`` vs ``wp_reset`` voxel IoU + NN median.

Both passes use the SAME extractor, SAME mode, SAME input frames. The only
difference is the mid-stream save + reset_for_scene + restore. If PERSIST
preserves the frame, the tails match (high IoU, NN ~ 0). If VGGT re-anchors
despite the restored cache, the tails differ by a similarity transform (low IoU,
decimeter NN) — and PERSIST alone is insufficient (option C, GT-pose Umeyama,
would be needed).

Run on a GPU node:
    uv run python src/prototyp/live_house_context/check_persist_alignment.py
    uv run python src/prototyp/live_house_context/check_persist_alignment.py --total 80 --split 60 --tail 20

References: PROTOCOL.md §7.4 (the confounded first attempt) and §3.D4.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.launch.habitat_setup import make_habitat_env
from src.r2dreamer.encoders.base import VGGTEncoder
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor, ResetMode

RENDER_RESOLUTION = 518
PACK_OFFSET = 1 << 20  # 21 bits/axis after offsetting; +-2^20 voxels @ 1 cm.


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments for the alignment diagnostic.

    Returns:
        The parsed argparse namespace (total, split, tail, seed, curriculum,
        voxel, output).
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total", type=int, default=80, help="Frames to record.")
    p.add_argument("--split", type=int, default=60, help="Episode-0 length N.")
    p.add_argument("--tail", type=int, default=20, help="Tail frames compared.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--curriculum", type=str, default="L1")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument(
        "--output",
        type=str,
        default="output/check/persist_alignment",
        help="Directory for the recorded-frame PLYs (debug).",
    )
    return p.parse_args()


def pack_voxels(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Return the sorted unique set of occupied voxels as packed int64 keys."""
    idx = np.floor(xyz.astype(np.float64) / voxel).astype(np.int64) + PACK_OFFSET
    if (idx < 0).any() or (idx >= (1 << 21)).any():
        raise ValueError("voxel index out of packable range; scene too large?")
    keys = (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]
    return np.unique(keys)


def voxel_iou(a: np.ndarray, b: np.ndarray, voxel: float) -> float:
    """Voxel-occupancy intersection-over-union of two point clouds."""
    ka, kb = pack_voxels(a, voxel), pack_voxels(b, voxel)
    inter = np.intersect1d(ka, kb, assume_unique=True).size
    union = ka.size + kb.size - inter
    return float(inter) / float(union) if union else 0.0


def nn_distance_quantiles(
    query: jax.Array, reference: jax.Array, chunk: int = 2048
) -> tuple[float, float]:
    """Median and p90 of min-distance from each query point to the reference.

    Both sides are subsampled by the caller to keep the query x ref matrix small.
    """
    mins = []
    for start in range(0, query.shape[0], chunk):
        q = query[start : start + chunk]
        d2 = jnp.sum((q[:, None, :] - reference[None, :, :]) ** 2, axis=-1)
        mins.append(jnp.sqrt(jnp.min(d2, axis=1)))
    d = jnp.concatenate(mins)
    return float(jnp.median(d)), float(jnp.quantile(d, 0.9))


def subsample(xyz: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return up to ``n`` rows of ``xyz`` chosen uniformly without replacement."""
    if xyz.shape[0] <= n:
        return xyz
    return xyz[rng.choice(xyz.shape[0], size=n, replace=False)]


def record_frames(env, total: int, seed: int) -> list[ObservationFrame]:
    """Collect ``total`` frames from one env episode (reset once, step until done).

    Args:
        env: A habitat env with ``reset``/``step``/``num_actions``.
        total: Maximum number of frames to collect; fewer if the episode ends.
        seed: Seed for the random action generator.

    Returns:
        The list of recorded ``ObservationFrame`` objects (length <= ``total``).
    """
    rng = np.random.default_rng(seed)
    obs = env.reset()
    frames = [obs]
    while len(frames) < total:
        action = int(rng.integers(0, env.num_actions))
        obs = env.step(action)
        frames.append(obs)
        if obs.done:
            break
    return frames


def world_points_of(out) -> np.ndarray:
    """Extract the flat world-points array from a VGGT output.

    Args:
        out: A VGGT output (dict or dataclass) carrying ``world_points``.

    Returns:
        An ``(N, 3)`` ``np.ndarray`` of world-frame points.
    """
    wp = out["world_points"] if isinstance(out, dict) else out.world_points
    return np.asarray(wp).reshape(-1, 3)


def _frame_with(obs: ObservationFrame, is_first: bool) -> ObservationFrame:
    """Return a copy of ``obs`` with ``is_first`` overridden.

    ``ObservationFrame`` is frozen, so a new instance is produced via
    ``dataclasses.replace``.

    Args:
        obs: The frame to copy.
        is_first: The ``is_first`` value to set on the copy.

    Returns:
        A new ``ObservationFrame`` identical to ``obs`` but with ``is_first``.
    """
    from dataclasses import replace
    return replace(obs, is_first=is_first)


def run_continuous(extractor, frames, split, tail) -> list[np.ndarray]:
    """Feed all frames continuously (``is_first`` only on frame 0); return tail wp.

    Args:
        extractor: A VGGT feature extractor.
        frames: Recorded ``ObservationFrame`` list to feed.
        split: Index at which the tail begins.
        tail: Number of tail frames to return world points for.

    Returns:
        A list of ``(N, 3)`` world-point arrays for the last ``tail`` frames.
    """
    extractor.reset()
    wps = []
    for i, obs in enumerate(frames):
        out = extractor.extract(_frame_with(obs, is_first=(i == 0)))
        if i >= split:
            wps.append(world_points_of(out))
    return wps[-tail:]


def run_reset_restore(extractor, frames, split, tail, scene_id) -> list[np.ndarray]:
    """Feed 0..split-1, ``reset_for_scene`` (save), then feed the tail with ``is_first`` on the first tail frame.

    This mirrors the production trainer's episode boundary: at ``split`` the
    cache is saved via ``reset_for_scene`` and the first tail frame signals
    ``is_first`` so the same call restores the saved cache.

    Args:
        extractor: A VGGT feature extractor (PERSIST_SCENE).
        frames: Recorded ``ObservationFrame`` list to feed.
        split: Episode-boundary index (cache saved here).
        tail: Number of tail frames to return world points for.
        scene_id: Scene id passed to ``reset_for_scene`` (keys the saved cache).

    Returns:
        A list of ``(N, 3)`` world-point arrays for the last ``tail`` frames.
    """
    extractor.reset()
    wps = []
    for i, obs in enumerate(frames):
        if i == split:
            # Episode boundary: save the cache, switch to the same scene id,
            # and signal the boundary on the first tail frame so reset_for_scene
            # restores the saved cache (exactly the production trainer path).
            extractor.reset_for_scene(scene_id)
        is_first = (i == 0) or (i == split)
        out = extractor.extract(_frame_with(obs, is_first=is_first))
        if i >= split:
            wps.append(world_points_of(out))
    return wps[-tail:]


def compare(a: np.ndarray, b: np.ndarray, voxel: float, rng) -> dict:
    """Compute voxel IoU and NN-distance quantiles between two point clouds.

    Both clouds are subsampled to 4000 points before the NN-distance compute
    to bound the query x reference matrix; the IoU uses the full clouds.

    Args:
        a: First point cloud, ``(N, 3)``.
        b: Second point cloud, ``(M, 3)``.
        voxel: Voxel edge length in metres for the IoU grid.
        rng: NumPy generator for the subsampling choice.

    Returns:
        A dict with ``iou``, ``nn_med_m``, ``nn_p90_m``, ``n_a``, ``n_b``.
    """
    iou = voxel_iou(a, b, voxel)
    sub_a = subsample(a, 4000, rng)
    sub_b = subsample(b, 4000, rng)
    med, p90 = nn_distance_quantiles(jnp.asarray(sub_a), jnp.asarray(sub_b))
    return {"iou": iou, "nn_med_m": med, "nn_p90_m": p90, "n_a": a.shape[0], "n_b": b.shape[0]}


def main() -> None:
    """Run the PERSIST frame-preservation diagnostic and print the verdict.

    Records frames from one env episode, runs the continuous and
    reset+restore passes over the same frames through one PERSIST extractor,
    and prints per-tail-frame IoU/NN plus an overall PASS/PARTIAL/FAIL verdict.
    """
    args = parse_args()
    assert args.split + args.tail <= args.total, "split + tail must be <= total"
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"config: total={args.total} split={args.split} tail={args.tail} voxel={args.voxel}m\n")

    env = make_habitat_env(
        curriculum=args.curriculum, render_resolution=RENDER_RESOLUTION, seed=args.seed
    )
    frames = record_frames(env, args.total, args.seed)
    n = len(frames)
    split = min(args.split, n - 1)
    tail = min(args.tail, n - split)
    print(f"recorded {n} frames; split={split}, tail={tail}")
    if tail < 1:
        print("FAIL: episode ended before the tail; use a longer episode or smaller split.")
        return
    scene_id = getattr(frames[0], "scene_id", None) or "alignment_scene"

    extractor = JAXVGGTFeatureExtractor(
        total_budget=VGGTEncoder.VGGT_TOTAL_BUDGET,
        budgets_static=VGGTEncoder.VGGT_STATIC_BUDGETS,
        compute_heads=True,
        reset_mode=ResetMode.PERSIST_SCENE,
    )
    rng = np.random.default_rng(0)

    print("\n>>> continuous pass (no mid-stream reset; baseline for the same frames)")
    t0 = time.perf_counter()
    wp_cont = run_continuous(extractor, frames, split, tail)
    print(f"  done in {time.perf_counter()-t0:.1f}s; tail sizes: {[w.shape[0] for w in wp_cont]}")

    print("\n>>> reset+restore pass (reset_for_scene at the split, same frames)")
    t0 = time.perf_counter()
    wp_rst = run_reset_restore(extractor, frames, split, tail, scene_id)
    print(f"  done in {time.perf_counter()-t0:.1f}s; tail sizes: {[w.shape[0] for w in wp_rst]}")

    print(f"\n=== tail frame-by-frame: continuous vs reset+restore (voxel={args.voxel}m) ===")
    print(f"  {'frame':>5} {'IoU':>6} {'NNmed(m)':>9} {'NNp90(m)':>9} {'n_cont':>8} {'n_rst':>8}")
    ious = []
    for k, (a, b) in enumerate(zip(wp_cont, wp_rst, strict=True)):
        c = compare(a, b, args.voxel, rng)
        ious.append(c["iou"])
        print(
            f"  {split+k:>5} {c['iou']:>6.3f} {c['nn_med_m']:>9.3f} {c['nn_p90_m']:>9.3f} "
            f"{c['n_a']:>8} {c['n_b']:>8}"
        )

    mean_iou = float(np.mean(ious))
    print(f"\n=== VERDICT ===")
    print(f"  mean tail IoU (continuous vs reset+restore) = {mean_iou:.3f}")
    if mean_iou > 0.5:
        print("  PASS — PERSIST preserves the VGGT world frame across the boundary "
              "(save+reset_for_scene+restore keeps one coordinate system).")
    elif mean_iou > 0.2:
        print("  PARTIAL — frames are closer than FULL re-anchor but not identical; "
              "VGGT partially re-anchors despite a restored cache. Option C may help.")
    else:
        print("  FAIL — reset_for_scene does NOT preserve the frame; VGGT re-anchors "
              "despite the restored cache. PERSIST alone is insufficient -> option C "
              "(GT-pose + running Umeyama) is required.")


if __name__ == "__main__":
    main()