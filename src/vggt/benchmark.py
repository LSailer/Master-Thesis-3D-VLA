"""Benchmarking utilities for VGGT variants."""

import time
import torch
import pandas as pd
from typing import Any


def run_inference(model: torch.nn.Module, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run a single forward pass, returning dict with point_map and features."""
    device = next(model.parameters()).device
    # Ensure input is on correct device and in (B, C, H, W) format
    if rgb.dim() == 4 and rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)  # BHWC -> BCHW
    rgb = rgb.to(device)

    with torch.no_grad():
        output = model(rgb)

    return output


def benchmark_variant(
    model: torch.nn.Module,
    sequence_lengths: list[int],
    height: int = 480,
    width: int = 640,
    warmup: int = 3,
) -> list[dict[str, Any]]:
    """Benchmark a model across sequence lengths, returning per-length results."""
    device = next(model.parameters()).device
    results = []

    for n_frames in sequence_lengths:
        # Warmup
        dummy = torch.rand(1, 3, height, width, device=device)
        for _ in range(warmup):
            with torch.no_grad():
                model(dummy)
        torch.cuda.synchronize()

        # Measure
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(n_frames):
            with torch.no_grad():
                output = model(dummy)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6  # MB
        latency_ms = (elapsed / n_frames) * 1000

        # Get output shape from first available tensor
        output_shape = None
        if isinstance(output, dict):
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    output_shape = tuple(v.shape)
                    break

        results.append({
            "n_frames": n_frames,
            "latency_ms": latency_ms,
            "peak_mem_mb": peak_mem,
            "output_shape": str(output_shape),
        })

    return results


def build_comparison_table(
    all_results: dict[str, list[dict]],
) -> pd.DataFrame:
    """Build a summary DataFrame from per-variant benchmark results.

    Args:
        all_results: mapping of variant name -> list of per-sequence-length dicts

    Returns:
        DataFrame with columns: variant, latency_ms, peak_mem_mb, output_shape
    """
    rows = []
    for variant_name, seq_results in all_results.items():
        for r in seq_results:
            rows.append({
                "variant": variant_name,
                "n_frames": r["n_frames"],
                "latency_ms": r["latency_ms"],
                "peak_mem_mb": r["peak_mem_mb"],
                "output_shape": r["output_shape"],
            })
    return pd.DataFrame(rows)
