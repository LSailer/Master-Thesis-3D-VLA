"""Tests for VGGT variant comparison (issue #2)."""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

gpu = pytest.mark.skipif(
    not (HAS_TORCH and torch.cuda.is_available()),
    reason="requires CUDA GPU",
)


def test_benchmark_notebook_exists():
    """Acceptance: Benchmark notebook exists at notebooks/vggt_comparison.ipynb."""
    nb = ROOT / "notebooks" / "vggt_comparison.ipynb"
    assert nb.exists(), f"Expected notebook at {nb}"


@gpu
def test_variants_produce_point_maps():
    """Each variant loads and produces point maps from a single 480x640 RGB input."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from vggt_comparison import get_available_variants, load_variant, run_inference

    dummy_rgb = torch.rand(1, 480, 640, 3)  # single RGB frame

    variants = get_available_variants()
    assert len(variants) > 0, "No variants available"

    for name in variants:
        model = load_variant(name)
        result = run_inference(model, dummy_rgb)
        assert "point_map" in result, f"{name}: missing point_map in output"
        pm = result["point_map"]
        assert pm.shape[1] == 480 and pm.shape[2] == 640, (
            f"{name}: point_map spatial dims {pm.shape} != (480, 640)"
        )


@gpu
def test_benchmark_measures_latency_and_memory():
    """Inference latency and peak GPU memory are measured per variant for N=10,20,50,100,500."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from vggt_comparison import get_available_variants, load_variant, benchmark_variant

    expected_lengths = [10, 20, 50, 100, 500]
    variants = get_available_variants()
    assert len(variants) > 0

    for name in variants:
        model = load_variant(name)
        results = benchmark_variant(model, sequence_lengths=expected_lengths)
        assert len(results) == len(expected_lengths)
        for r, n in zip(results, expected_lengths):
            assert r["n_frames"] == n
            assert "latency_ms" in r and isinstance(r["latency_ms"], float)
            assert "peak_mem_mb" in r and isinstance(r["peak_mem_mb"], float)
            assert r["latency_ms"] > 0, f"{name} n={n}: latency must be positive"
            assert r["peak_mem_mb"] > 0, f"{name} n={n}: peak_mem must be positive"
