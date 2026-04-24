"""Fitness command for the JAX VGGT autoresearch loop: cached PT baseline vs fresh JAX timing."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SEQ_LEN = 50
WARMUP_FRAMES = 3
TIMED_FRAMES = 20
CACHE_PATH = Path(__file__).parent / ".pt_baseline.json"


def _make_frame(seed: int, size: int = 518) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(3, size, size), dtype=np.uint8)


def bench_pytorch() -> dict:
    import torch

    from modules.vggt.feature_extractor import VGGTFeatureExtractor

    ext = VGGTFeatureExtractor(device="cuda")

    ext.reset()
    for i in range(WARMUP_FRAMES):
        ext.extract(_make_frame(i))
    torch.cuda.synchronize()

    ext.reset()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for i in range(TIMED_FRAMES):
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
        "median_ms": float(np.median(latencies)),
        "mean_ms": float(np.mean(latencies)),
        "peak_mem_mb": peak_mem,
    }


def bench_jax() -> dict:
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    from modules.vggt.jax import JAXVGGTFeatureExtractor

    ext = JAXVGGTFeatureExtractor(device="cuda")

    ext.reset()
    for i in range(WARMUP_FRAMES):
        out = ext.extract(_make_frame(i))
        _ = np.asarray(out["world_points"])

    ext.reset()
    latencies = []
    for i in range(TIMED_FRAMES):
        t0 = time.perf_counter()
        out = ext.extract(_make_frame(1000 + i))
        _ = np.asarray(out["world_points"])  # force device sync
        latencies.append((time.perf_counter() - t0) * 1000.0)

    del ext
    gc.collect()

    return {
        "median_ms": float(np.median(latencies)),
        "mean_ms": float(np.mean(latencies)),
        "peak_mem_mb": 0.0,
    }


def save_baseline(stats: dict) -> None:
    payload = {
        "pt_baseline_ms": stats["median_ms"],
        "pt_mean_ms": stats["mean_ms"],
        "peak_mem_mb": stats["peak_mem_mb"],
        "seq_len": SEQ_LEN,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=2))


def load_baseline() -> dict:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(str(CACHE_PATH))
    return json.loads(CACHE_PATH.read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--setup",
        action="store_true",
        help="Time PyTorch once and cache the result, then exit.",
    )
    args = p.parse_args()

    if args.setup:
        stats = bench_pytorch()
        save_baseline(stats)
        print(f"pt_baseline_ms: {stats['median_ms']:.2f}")
        print(f"pt_mean_ms:     {stats['mean_ms']:.2f}")
        print(f"peak_mem_mb:    {stats['peak_mem_mb']:.1f}")
        print(f"cached_to:      {CACHE_PATH.resolve()}")
        return 0

    try:
        baseline = load_baseline()
    except FileNotFoundError:
        print(
            "ERROR: .pt_baseline.json not found. Run with --setup first.",
            file=sys.stderr,
        )
        return 1

    jax_stats = bench_jax()
    pt_baseline_ms = float(baseline["pt_baseline_ms"])
    speedup = pt_baseline_ms / jax_stats["median_ms"]

    print(f"jax_median_ms:  {jax_stats['median_ms']:.2f}")
    print(f"jax_mean_ms:    {jax_stats['mean_ms']:.2f}")
    print(f"pt_baseline_ms: {pt_baseline_ms:.2f}")
    print(f"speedup:        {speedup:.3f}")
    print(f"peak_mem_mb:    {jax_stats['peak_mem_mb']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
