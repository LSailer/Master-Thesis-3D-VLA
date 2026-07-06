"""Check cross-episode frame alignment of the live house-context map.

The per-scene ``HouseContextPoseBuffer`` persists across episodes, but the
VGGT extractor is reset at every episode boundary, so each episode's
``world_points`` are anchored to that episode's first camera (and VGGT depth
is monocular, i.e. up-to-scale per sequence). ``buffer.add`` stores the raw
world points with no re-anchoring, so if episode frames do not coincide the
persistent map accumulates misaligned copies of the same house.

Smoke job 5706115 showed the map growing to 3.38M points in 1200 env steps
with a large jump across the first episode boundary — consistent with (but
not proof of) misalignment. This script settles it: it collects one SEPARATE
buffer per episode on the same scene, saves each episode's cloud as PLY, and
measures pairwise alignment between episode clouds:

  * occupied-voxel IoU at 5 / 10 / 20 cm (exact packed-int voxel sets)
  * nearest-neighbour distance quantiles from episode i to episode 0
  * fraction of episode-i points that are NEW voxels when inserted into a
    buffer seeded with episode 0 (the exact quantity that inflates the
    persistent production buffer)

Aligned episodes: IoU well above zero, median NN distance ~ voxel size, low
new-voxel fraction. Misaligned frames: near-zero IoU, decimeter+ NN
distances, new-voxel fraction near 1.

Run on a GPU node:
    uv run python scripts/r2dreamer/check_episode_frame_alignment.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.r2dreamer.adapters.hybrid_adapter import VGGTHousePointsPoseObsAdapter
from src.r2dreamer.encoders.base import VGGTEncoder
from src.r2dreamer.launch.habitat_setup import make_habitat_env
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor

RENDER_RESOLUTION = 518
CAPACITY = 1 << 22
HASH_TABLE_SIZE = 1 << 23
# 21 bits per axis after offsetting; +-2^20 voxels covers +-10 km at 1 cm.
PACK_OFFSET = 1 << 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=60, help="Env steps per episode.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--curriculum", type=str, default="L1")
    p.add_argument("--voxel", type=float, default=0.01, help="Buffer voxel size (m).")
    p.add_argument(
        "--output",
        type=str,
        default="output/check/episode_frame_alignment",
        help="Root directory for per-episode PLY snapshots.",
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
    ka, kb = pack_voxels(a, voxel), pack_voxels(b, voxel)
    inter = np.intersect1d(ka, kb, assume_unique=True).size
    union = ka.size + kb.size - inter
    return float(inter) / float(union) if union else 0.0


def nn_distance_quantiles(
    query: jax.Array, reference: jax.Array, chunk: int = 2048
) -> tuple[float, float]:
    """Median and p90 of min-distance from each query point to the reference."""
    mins = []
    for start in range(0, query.shape[0], chunk):
        q = query[start : start + chunk]
        d2 = jnp.sum((q[:, None, :] - reference[None, :, :]) ** 2, axis=-1)
        mins.append(jnp.sqrt(jnp.min(d2, axis=1)))
    d = jnp.concatenate(mins)
    return float(jnp.median(d)), float(jnp.quantile(d, 0.9))


def subsample(xyz: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if xyz.shape[0] <= n:
        return xyz
    return xyz[rng.choice(xyz.shape[0], size=n, replace=False)]


def episode_cloud(buffer: HouseContextPoseBuffer) -> np.ndarray:
    return np.asarray(buffer.points_xyz, dtype=np.float32)


def new_voxel_fraction(
    base_xyz: np.ndarray, added_xyz: np.ndarray, confidence: float, voxel: float
) -> float:
    """Fraction of ``added_xyz`` voxels that are new on top of ``base_xyz``.

    Uses a fresh HouseContextPoseBuffer so 'new' means exactly what production
    dedup means at the production voxel size.
    """
    probe = HouseContextPoseBuffer(
        confidence_score=confidence,
        scene_id="alignment_probe",
        voxel_size_m=voxel,
        capacity=CAPACITY,
        hash_table_size=HASH_TABLE_SIZE,
    )
    ones = np.ones((base_xyz.shape[0], 3), dtype=np.float32)
    probe.seed_xyzrgb(jnp.asarray(np.concatenate([base_xyz, ones], axis=1)))
    before = probe.point_count
    ones = np.ones((added_xyz.shape[0], 3), dtype=np.float32)
    probe.seed_xyzrgb(jnp.asarray(np.concatenate([added_xyz, ones], axis=1)))
    added_unique = pack_voxels(added_xyz, voxel).size
    return (probe.point_count - before) / max(added_unique, 1)


def main() -> None:
    args = parse_args()
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(
        f"config: curriculum={args.curriculum}  render={RENDER_RESOLUTION}  "
        f"episodes={args.episodes}  steps/episode={args.steps}  "
        f"voxel={args.voxel}  seed={args.seed}\n"
    )

    env = make_habitat_env(
        curriculum=args.curriculum, render_resolution=RENDER_RESOLUTION, seed=args.seed
    )
    extractor = JAXVGGTFeatureExtractor(
        total_budget=VGGTEncoder.VGGT_TOTAL_BUDGET,
        budgets_static=VGGTEncoder.VGGT_STATIC_BUDGETS,
        compute_heads=True,
    )
    adapter = VGGTHousePointsPoseObsAdapter(extractor)
    rng = np.random.default_rng(args.seed)

    clouds: list[np.ndarray] = []
    episode_ids: list[str] = []
    for ep in range(args.episodes):
        obs = env.reset()
        extractor.reset()
        buffer = HouseContextPoseBuffer(
            confidence_score=adapter._confidence_score,
            scene_id=f"alignment_ep{ep}",
            voxel_size_m=args.voxel,
            capacity=CAPACITY,
            hash_table_size=HASH_TABLE_SIZE,
        )
        episode_ids.append(str(obs.episode_id))
        t0 = time.perf_counter()
        for _ in range(args.steps):
            action = int(rng.integers(0, env.num_actions))
            obs = env.step(action)
            out = extractor.extract(obs)
            buffer.add(out, obs)
            if obs.done:
                break
        jax.block_until_ready(buffer._state)
        cloud = episode_cloud(buffer)
        clouds.append(cloud)
        scene_dir = buffer.save(Path(args.output))
        print(
            f"episode {ep} (id={episode_ids[-1]}, scene={obs.scene_id}): "
            f"{cloud.shape[0]} points in {time.perf_counter() - t0:.1f}s "
            f"-> {scene_dir}"
        )

    print("\n=== pairwise alignment vs episode 0 ===")
    ref_full = clouds[0]
    ref_sub = jnp.asarray(subsample(ref_full, 300_000, rng))
    misaligned = 0
    for ep in range(1, len(clouds)):
        iou5 = voxel_iou(ref_full, clouds[ep], 0.05)
        iou10 = voxel_iou(ref_full, clouds[ep], 0.10)
        iou20 = voxel_iou(ref_full, clouds[ep], 0.20)
        query = jnp.asarray(subsample(clouds[ep], 20_000, rng))
        nn_med, nn_p90 = nn_distance_quantiles(query, ref_sub)
        new_frac = new_voxel_fraction(
            ref_full, clouds[ep], adapter._confidence_score, args.voxel
        )
        print(
            f"episode {ep} vs 0: IoU@5cm={iou5:.3f}  IoU@10cm={iou10:.3f}  "
            f"IoU@20cm={iou20:.3f}  NN median={nn_med * 100:.1f}cm  "
            f"p90={nn_p90 * 100:.1f}cm  new-voxel-frac@{args.voxel * 100:g}cm="
            f"{new_frac:.3f}"
        )
        if iou10 < 0.05 or nn_med > 0.25:
            misaligned += 1

    print("\n=== verdict ===")
    if misaligned == 0:
        print(
            "ALIGNED: episode clouds land on top of each other; map growth is "
            "genuine coverage/noise, not frame misalignment."
        )
    elif misaligned == len(clouds) - 1:
        print(
            "MISALIGNED: episode frames do not coincide; the persistent "
            "per-scene buffer is accumulating misaligned copies of the house. "
            "Fix: re-anchor VGGT world points to a common frame before add()."
        )
    else:
        print(
            f"MIXED: {misaligned}/{len(clouds) - 1} episode pairs misaligned; "
            "inspect the PLYs (e.g. in MeshLab) before concluding."
        )


if __name__ == "__main__":
    main()
