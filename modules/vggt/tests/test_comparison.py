"""Tests for VGGT variant comparison (issue #2)."""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

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
    nb = ROOT / "modules" / "vggt" / "notebooks" / "comparison.ipynb"
    assert nb.exists(), f"Expected notebook at {nb}"


@gpu
def test_variants_produce_point_maps():
    """Each variant loads and produces point maps from a single 480x640 RGB input."""
    from modules.vggt import get_available_variants, load_variant, run_inference

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
    from modules.vggt import get_available_variants, load_variant, benchmark_variant

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


@gpu
def test_feature_output_shape_consistent():
    """Feature output shape is consistent across variants (same spatial resolution and channel dim)."""
    from modules.vggt import get_available_variants, load_variant, run_inference

    dummy_rgb = torch.rand(1, 480, 640, 3)
    variants = get_available_variants()
    assert len(variants) > 0

    shapes = {}
    for name in variants:
        model = load_variant(name)
        result = run_inference(model, dummy_rgb)
        assert "features" in result, f"{name}: missing 'features' key"
        shapes[name] = result["features"].shape

    # All variants should produce the same feature shape
    shape_list = list(shapes.values())
    for name, shape in shapes.items():
        assert shape == shape_list[0], (
            f"{name} feature shape {shape} != reference {shape_list[0]}"
        )


def test_comparison_table_structure():
    """A summary comparison table (pandas DataFrame) has required columns."""
    from modules.vggt import build_comparison_table

    mock_results = {
        "vggt": [
            {"n_frames": 10, "latency_ms": 12.5, "peak_mem_mb": 1024.0, "output_shape": "(1, 256, 60, 80)"},
            {"n_frames": 20, "latency_ms": 13.1, "peak_mem_mb": 1100.0, "output_shape": "(1, 256, 60, 80)"},
        ],
        "stream_vggt": [
            {"n_frames": 10, "latency_ms": 8.2, "peak_mem_mb": 800.0, "output_shape": "(1, 256, 60, 80)"},
            {"n_frames": 20, "latency_ms": 8.5, "peak_mem_mb": 850.0, "output_shape": "(1, 256, 60, 80)"},
        ],
    }
    df = build_comparison_table(mock_results)

    required_cols = {"variant", "latency_ms", "peak_mem_mb", "output_shape"}
    assert required_cols.issubset(set(df.columns)), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )
    assert len(df) == 4  # 2 variants x 2 sequence lengths
    assert set(df["variant"].unique()) == {"vggt", "stream_vggt"}
