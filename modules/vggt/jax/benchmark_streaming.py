"""Streaming latency benchmark: JAX vs PyTorch VGGTFeatureExtractor.

Measures per-frame wall time and peak GPU memory for each backend over a
set of sequence lengths (frames per episode). Each sequence starts with a
fresh ``reset()`` so the cache builds up organically.

Writes a CSV to ``output/comparison/vggt_streaming_<timestamp>.csv`` with
columns ``backend, n_frames, mean_latency_ms, median_latency_ms,
peak_mem_mb``.

v1 runs both backends in their default production settings
(PyTorch: bf16 autocast; JAX: eager fp32). Future work: add --jit and
--bf16 flags for JAX and compare against PyTorch compile=True.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np


DEFAULT_SEQ_LENS = (10, 50, 100)
WARMUP_FRAMES = 3


def _make_frame(seed: int, size: int = 518) -> np.ndarray:
    """Synthetic CHW uint8 RGB frame."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(3, size, size), dtype=np.uint8)


def bench_pytorch(n_frames: int) -> dict:
    """Benchmark the PyTorch extractor over ``n_frames`` with fresh cache."""
    import torch

    from modules.vggt.feature_extractor import VGGTFeatureExtractor

    ext = VGGTFeatureExtractor(device="cuda")

    # Warmup.
    ext.reset()
    for i in range(WARMUP_FRAMES):
        ext.extract(_make_frame(i))
    torch.cuda.synchronize()

    ext.reset()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for i in range(n_frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ext.extract(_make_frame(1000 + i))
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    del ext
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "backend": "pytorch",
        "n_frames": n_frames,
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "peak_mem_mb": peak_mem,
    }


def bench_jax(n_frames: int) -> dict:
    """Benchmark the JAX extractor (eager fp32) over ``n_frames``."""
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    from modules.vggt.jax import JAXVGGTFeatureExtractor

    ext = JAXVGGTFeatureExtractor(device="cuda")

    ext.reset()
    for i in range(WARMUP_FRAMES):
        out = ext.extract(_make_frame(i))
        # block_until_ready on a sentinel
        _ = np.asarray(out["world_points"])

    ext.reset()
    latencies = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        out = ext.extract(_make_frame(1000 + i))
        _ = np.asarray(out["world_points"])  # ensure sync
        latencies.append((time.perf_counter() - t0) * 1000.0)

    # JAX doesn't expose a simple peak-memory counter here; fall back to 0.
    peak_mem = 0.0

    del ext
    gc.collect()

    return {
        "backend": "jax",
        "n_frames": n_frames,
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "peak_mem_mb": peak_mem,
    }


def run(seq_lens: tuple[int, ...], backends: tuple[str, ...], out_dir: Path) -> Path:
    rows = []
    for n in seq_lens:
        for backend in backends:
            print(f"[{backend}] n_frames={n} ...", flush=True)
            if backend == "pytorch":
                row = bench_pytorch(n)
            elif backend == "jax":
                row = bench_jax(n)
            else:
                raise ValueError(f"unknown backend {backend}")
            print(
                f"  mean={row['mean_latency_ms']:.1f}ms "
                f"median={row['median_latency_ms']:.1f}ms "
                f"peak_mem={row['peak_mem_mb']:.0f}MB"
            )
            rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"vggt_streaming_{stamp}.csv"
    with out_path.open("w") as f:
        f.write("backend,n_frames,mean_latency_ms,median_latency_ms,peak_mem_mb\n")
        for r in rows:
            f.write(
                f"{r['backend']},{r['n_frames']},"
                f"{r['mean_latency_ms']:.3f},{r['median_latency_ms']:.3f},"
                f"{r['peak_mem_mb']:.1f}\n"
            )
    print(f"\nWrote {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEQ_LENS),
        help="Frames per episode to measure.",
    )
    p.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["pytorch", "jax"],
        choices=["pytorch", "jax"],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/comparison"),
    )
    args = p.parse_args()
    run(tuple(args.seq_lens), tuple(args.backends), args.out_dir)


if __name__ == "__main__":
    main()
