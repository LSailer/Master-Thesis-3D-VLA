"""Streaming latency benchmark: JAX vs PyTorch VGGTFeatureExtractor.

Measures per-frame wall time and peak GPU memory for each backend over a
set of sequence lengths (frames per episode). Each sequence starts with a
fresh ``reset()`` so the cache builds up organically.

Writes a CSV to ``output/methods/comparisons/vggt_streaming_<timestamp>.csv`` with
columns ``backend, config, n_frames, mean_latency_ms, median_latency_ms,
peak_mem_mb``.

The script honestly reports what it ran: each row's ``config`` column
records the resolved dtype / matmul-precision / compile settings.
Defaults match current production paths — PyTorch uses bf16 autocast
(and optional ``torch.compile``); JAX uses bf16 with the jitted
aggregator + camera head from ``feature_extractor.py`` (PR #80, #81).

Note: the historical eager-fp32 regime cited in
``docs/wiki/experiments/vggt_jax_port_step8_bench.md`` is no longer
reproducible from main — ``feature_extractor.py`` always jits and
defaults to bf16. ``--jax-dtype fp32`` + ``--jax-matmul-precision
highest`` is the closest current proxy (jitted but fp32 weights and
fp32-accumulated matmul); the 5395.9 ms / frame number from that wiki
page is a historical snapshot, not a current claim.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from src.shared.profiling import make_synthetic_rgb_frame


DEFAULT_SEQ_LENS = (10, 50, 100)
WARMUP_FRAMES = 3


def bench_pytorch(
    n_frames: int,
    compile: bool = False,
    compile_mode: str | None = None,
) -> dict:
    """Benchmark the PyTorch extractor over ``n_frames`` with fresh cache."""
    import torch

    from src.vggt.reference.feature_extractor import VGGTFeatureExtractor

    ext = VGGTFeatureExtractor(
        device="cuda",
        compile=compile,
        compile_mode=compile_mode,
    )

    # Warmup.
    ext.reset()
    for i in range(WARMUP_FRAMES):
        ext.extract(make_synthetic_rgb_frame(i))
    torch.cuda.synchronize()

    ext.reset()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for i in range(n_frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ext.extract(make_synthetic_rgb_frame(1000 + i))
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    del ext
    gc.collect()
    torch.cuda.empty_cache()

    if compile:
        compile_tag = (
            f"+compile-{compile_mode}" if compile_mode is not None else "+compile"
        )
    else:
        compile_tag = ""
    return {
        "backend": "pytorch",
        "config": f"bf16-autocast{compile_tag}",
        "n_frames": n_frames,
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "peak_mem_mb": peak_mem,
    }


def bench_jax(
    n_frames: int,
    dtype: str = "bf16",
    matmul_precision: str = "default",
    budgets_static: tuple[int, ...] | None = None,
) -> dict:
    """Benchmark the JAX extractor over ``n_frames`` with fresh cache.

    The aggregator, camera head, and point head are jitted at construction
    time inside ``JAXVGGTFeatureExtractor`` (PR #80, #81); this is not
    toggleable from the public API.
    """
    import jax
    import jax.numpy as jnp

    if matmul_precision == "highest":
        jax.config.update("jax_default_matmul_precision", "highest")
    from src.vggt.jax import JAXVGGTFeatureExtractor

    jax_dtype = jnp.bfloat16 if dtype == "bf16" else jnp.float32
    ext = JAXVGGTFeatureExtractor(
        device="cuda", dtype=jax_dtype, budgets_static=budgets_static
    )

    ext.reset()
    for i in range(WARMUP_FRAMES):
        out = ext.extract(make_synthetic_rgb_frame(i))
        # block until ready without host transfer
        _ = out.world_points.block_until_ready()

    ext.reset()
    latencies = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        out = ext.extract(make_synthetic_rgb_frame(1000 + i))
        out.world_points.block_until_ready()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    # JAX doesn't expose a simple peak-memory counter here; fall back to 0.
    peak_mem = 0.0

    del ext
    gc.collect()

    cfg = f"jax-jit-{dtype}"
    if matmul_precision == "highest":
        cfg += "-matmulhighest"
    if budgets_static is not None:
        cfg += "-staticbudgets"
    return {
        "backend": "jax",
        "config": cfg,
        "n_frames": n_frames,
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "peak_mem_mb": peak_mem,
    }


def run(
    seq_lens: tuple[int, ...],
    backends: tuple[str, ...],
    out_dir: Path,
    pt_compile: bool = False,
    pt_compile_mode: str | None = None,
    jax_dtype: str = "bf16",
    jax_matmul_precision: str = "default",
    jax_static_budgets: bool = False,
) -> Path:
    rows = []
    budgets_static = None
    if jax_static_budgets:
        # Uniform static budgets to avoid host<->device sync in JAX.
        # Match the extractor's budget split: max(total_budget / depth, P).
        depth = 24
        patch_tokens = 5 + (518 // 14) ** 2
        total_budget = 200_000
        uniform = max(total_budget // depth, patch_tokens)
        budgets_static = tuple([uniform] * depth)

    for n in seq_lens:
        for backend in backends:
            print(f"[{backend}] n_frames={n} ...", flush=True)
            if backend == "pytorch":
                row = bench_pytorch(
                    n,
                    compile=pt_compile,
                    compile_mode=pt_compile_mode,
                )
            elif backend == "jax":
                row = bench_jax(
                    n,
                    dtype=jax_dtype,
                    matmul_precision=jax_matmul_precision,
                    budgets_static=budgets_static,
                )
            else:
                raise ValueError(f"unknown backend {backend}")
            print(
                f"  config={row['config']} "
                f"mean={row['mean_latency_ms']:.1f}ms "
                f"median={row['median_latency_ms']:.1f}ms "
                f"peak_mem={row['peak_mem_mb']:.0f}MB"
            )
            rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"vggt_streaming_{stamp}.csv"
    with out_path.open("w") as f:
        f.write(
            "backend,config,n_frames,mean_latency_ms,median_latency_ms,peak_mem_mb\n"
        )
        for r in rows:
            f.write(
                f"{r['backend']},{r['config']},{r['n_frames']},"
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
        default=Path("output/methods/comparisons"),
    )
    p.add_argument(
        "--pt-compile",
        action="store_true",
        help="Wrap PyTorch aggregator/camera/point heads with torch.compile.",
    )
    p.add_argument(
        "--pt-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
        help=(
            "torch.compile mode forwarded to all three PyTorch sub-module "
            "compiles. Ignored unless --pt-compile is set. Omit to use "
            "torch's default mode (matches the originally shipped behaviour)."
        ),
    )
    p.add_argument(
        "--jax-dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="JAX feature-extractor dtype (default bf16, the production setting).",
    )
    p.add_argument(
        "--jax-matmul-precision",
        choices=["default", "highest"],
        default="default",
        help="jax_default_matmul_precision; 'highest' forces fp32 accumulation.",
    )
    p.add_argument(
        "--jax-static-budgets",
        action="store_true",
        help="Use a fixed static per-block budget to avoid host/device sync.",
    )
    args = p.parse_args()
    run(
        tuple(args.seq_lens),
        tuple(args.backends),
        args.out_dir,
        pt_compile=args.pt_compile,
        pt_compile_mode=args.pt_compile_mode,
        jax_dtype=args.jax_dtype,
        jax_matmul_precision=args.jax_matmul_precision,
        jax_static_budgets=args.jax_static_budgets,
    )


if __name__ == "__main__":
    main()
