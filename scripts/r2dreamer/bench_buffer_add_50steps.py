"""Time HouseContextPoseBuffer.add (fixed-shape hash-table version) per step.

Successor to ``bench_buffer_add_breakdown.py``, which instrumented the old
dynamic-shape implementation (``jnp.unique`` + host set loop, ~770 ms/step).
The buffer now runs a single donated JIT graph per frame (sort -> unique ->
vectorized open-addressing insert), so there are no host-visible stages left
to break down; this script times the whole ``add`` call against the real
production input (random agent on Habitat, VGGT extract) and saves the
accumulated house context as a PLY per configuration.

By default the full-resolution 518x518 map is fed to the buffer (no stride
subsampling) at both the production voxel size (5 cm) and the dense-map voxel
size (1 cm) so counts are comparable to the June-30 collection runs. Pass
``--stride`` to time the production stride-subsampled path instead.

Run on a GPU node (srun / sbatch):
    uv run python scripts/r2dreamer/bench_buffer_add_50steps.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import jax
import numpy as np

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.r2dreamer.adapters.house_points_adapter import VGGTHousePointsPoseObsAdapter
from src.r2dreamer.encoders.base import VGGTEncoder
from src.r2dreamer.launch.habitat_setup import make_habitat_env
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor

RENDER_RESOLUTION = 518
# Dense 1 cm voxels need headroom over the production 2^20 store: the June-30
# full-res 1 cm run hit 380k points at conf>=5.0 and production admits at 1.5.
DENSE_CAPACITY = 1 << 21
DENSE_HASH_TABLE_SIZE = 1 << 22


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--curriculum", type=str, default="L1")
    p.add_argument(
        "--voxels",
        type=str,
        default="0.05,0.01",
        help="Comma-separated voxel edge lengths (m); one buffer is timed per value.",
    )
    p.add_argument(
        "--stride",
        action="store_true",
        help="Use the production stride-subsampled input instead of the full map.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="output/bench/house_context_50steps",
        help="Root directory for the saved PLY snapshots.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    voxel_sizes = [float(v) for v in args.voxels.split(",") if v]
    input_mode = "stride" if args.stride else "full"
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"config: curriculum={args.curriculum}  render={RENDER_RESOLUTION}  "
          f"steps={args.steps} (warmup {args.warmup})  seed={args.seed}  "
          f"input={input_mode}  voxels={voxel_sizes}\n")

    env = make_habitat_env(
        curriculum=args.curriculum, render_resolution=RENDER_RESOLUTION, seed=args.seed
    )
    extractor = JAXVGGTFeatureExtractor(
        total_budget=VGGTEncoder.VGGT_TOTAL_BUDGET,
        budgets_static=VGGTEncoder.VGGT_STATIC_BUDGETS,
        compute_heads=True,
    )
    # Adapter reused only for its production stride helper + confidence config;
    # separate buffer instances are timed so this script owns the measurement.
    adapter = VGGTHousePointsPoseObsAdapter(extractor)
    labels = [f"{input_mode}_{v * 100:g}cm" for v in voxel_sizes]
    buffers = {
        label: HouseContextPoseBuffer(
            confidence_score=adapter._confidence_score,
            scene_id=f"bench_{args.steps}steps_{label}",
            voxel_size_m=voxel,
            capacity=DENSE_CAPACITY,
            hash_table_size=DENSE_HASH_TABLE_SIZE,
        )
        for label, voxel in zip(labels, voxel_sizes, strict=True)
    }
    print(f"adapter: conf>={adapter._confidence_score}  "
          f"max_input_points={adapter._max_input_points} (stride mode only)")
    print(f"buffers: capacity={DENSE_CAPACITY}  "
          f"hash_table_size={DENSE_HASH_TABLE_SIZE}\n")

    rng = np.random.default_rng(args.seed)
    env.reset()
    extractor.reset()

    add_ms: dict[str, list[float]] = {label: [] for label in labels}
    extract_ms: list[float] = []
    header = f"{'step':>4} {'extract':>9}"
    for label in labels:
        header += f" {'add_' + label:>15} {'pts_' + label:>12}"
    print(header)
    for step in range(args.steps):
        action = int(rng.integers(0, env.num_actions))
        obs = env.step(action)

        t = time.perf_counter()
        out = extractor.extract(obs)
        jax.block_until_ready((out.world_points, out.confidence))
        e_ms = (time.perf_counter() - t) * 1e3

        if args.stride:
            buffer_out, buffer_obs = adapter._subsampled_buffer_input(out, obs)
        else:
            buffer_out, buffer_obs = out, obs

        row = f"{step:4d} {e_ms:8.2f}m"
        for label, buffer in buffers.items():
            t = time.perf_counter()
            buffer.add(buffer_out, buffer_obs)
            jax.block_until_ready(buffer._state)
            a_ms = (time.perf_counter() - t) * 1e3
            if step >= args.warmup:
                add_ms[label].append(a_ms)
            # point_count syncs a scalar to host; fine outside the timed region.
            row += f" {a_ms:14.2f}m {buffer.point_count:12d}"
        print(row)
        if step >= args.warmup:
            extract_ms.append(e_ms)
        if obs.done:
            env.reset()
            extractor.reset()

    print(f"\n=== summary (excluding first {args.warmup} warmup steps) ===")
    print(f"  extract : median {statistics.median(extract_ms):8.2f} ms")
    for label, buffer in buffers.items():
        xs = add_ms[label]
        print(f"  [{label}]")
        print(f"    add     : median {statistics.median(xs):8.2f} ms  "
              f"mean {statistics.mean(xs):8.2f} ms  "
              f"min {min(xs):8.2f}  max {max(xs):8.2f}")
        print(f"    points accumulated : {buffer.point_count}")
        print(f"    overflow_count     : {buffer.overflow_count}")
        print(f"    failed_insert_count: {buffer.failed_insert_count}")
        scene_dir = buffer.save(Path(args.output))
        print(f"    PLY saved under: {scene_dir}")


if __name__ == "__main__":
    main()
