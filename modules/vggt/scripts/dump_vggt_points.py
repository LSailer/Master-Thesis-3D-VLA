"""Dump VGGT world_points to NPZ for visualization.

Usage:
  uv run python -m modules.vggt.scripts.dump_vggt_points --out output/vggt_points.npz

Optionally pass --seed to control the synthetic RGB.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modules.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor


def _make_frame(seed: int, size: int = 518) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(3, size, size), dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path, help="Output .npz path")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ext = JAXVGGTFeatureExtractor(device="cuda")
    ext.reset()
    out = ext.extract(_make_frame(args.seed))

    # Materialize to host for storage.
    world_points = np.asarray(out["world_points"])
    camera_pose = np.asarray(out["camera_pose"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, world_points=world_points, camera_pose=camera_pose)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
