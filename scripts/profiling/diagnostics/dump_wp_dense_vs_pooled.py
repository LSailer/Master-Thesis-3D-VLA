"""Dump dense (518x518x3) vs pooled (37x37x3) VGGT world points for 3D-48.

Runs the JAX VGGT extractor on a *real* input frame and captures the DPT
point map both before and after the ``_adaptive_avg_pool_518_to_37`` step,
so we can visualise exactly what spatial detail the 14x14 pooling discards.

Usage:
  uv run python -m scripts.vggt.dump_wp_dense_vs_pooled \
      --frames tests/r2dreamer/launch/fixtures/sample_habitat_obs.npz \
      --frame-index 0 \
      --out output/3d48/wp_dense_vs_pooled.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor


def _load_frame(frames_path: Path, idx: int) -> np.ndarray:
    """Return one (3, 518, 518) uint8 frame from an npz with key 'frames'."""
    data = np.load(frames_path)
    key = "frames" if "frames" in data.files else data.files[0]
    frames = data[key]
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected (N, 3, H, W) frames, got {frames.shape}")
    return np.ascontiguousarray(frames[idx])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", required=True, type=Path,
                   help="npz with a (N, 3, 518, 518) uint8 'frames' array")
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                   help="fp32 forces the XLA attention path (CPU-capable); "
                        "bf16 uses cuDNN flash attention (GPU only).")
    args = p.parse_args()

    import jax.numpy as jnp
    dtype = {"bf16": jnp.bfloat16, "fp32": jnp.float32}[args.dtype]

    rgb = _load_frame(args.frames, args.frame_index)  # (3, 518, 518) uint8

    ext = JAXVGGTFeatureExtractor(device=args.device, dtype=dtype)
    ext.reset()
    out = ext.extract(rgb, return_dense=True)

    dense = np.asarray(out["dense_world_points"])   # (518, 518, 3)
    pooled = np.asarray(out["world_points"])         # (37, 37, 3)
    camera_pose = np.asarray(out["camera_pose"])     # (9,)

    print(f"dense_world_points: {dense.shape} {dense.dtype}")
    print(f"world_points (pooled): {pooled.shape} {pooled.dtype}")
    print(f"points: dense={dense.shape[0] * dense.shape[1]}  "
          f"pooled={pooled.shape[0] * pooled.shape[1]}  "
          f"ratio={dense.shape[0] * dense.shape[1] / (pooled.shape[0] * pooled.shape[1]):.1f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        dense_world_points=dense,
        world_points=pooled,
        camera_pose=camera_pose,
        input_rgb=rgb,  # (3, 518, 518) uint8, for grid-overlay figures
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
