"""Tests for VGGT variant comparison (issue #2)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_benchmark_notebook_exists():
    """Acceptance: Benchmark notebook exists at notebooks/vggt_comparison.ipynb."""
    nb = ROOT / "notebooks" / "vggt_comparison.ipynb"
    assert nb.exists(), f"Expected notebook at {nb}"
