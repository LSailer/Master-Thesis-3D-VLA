"""PyTorch/reference VGGT utilities used for parity checks and comparisons."""

from src.vggt.reference.benchmark import (
    benchmark_variant,
    build_comparison_table,
    run_inference,
)
from src.vggt.reference.plots import plot_comparison
from src.vggt.reference.variants import VARIANTS, get_available_variants, load_variant

__all__ = [
    "VARIANTS",
    "benchmark_variant",
    "build_comparison_table",
    "get_available_variants",
    "load_variant",
    "plot_comparison",
    "run_inference",
]
