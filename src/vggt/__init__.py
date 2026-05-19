"""VGGT variant comparison utilities for issue #2."""

from .variants import VARIANTS, get_available_variants, load_variant
from .benchmark import run_inference, benchmark_variant, build_comparison_table
from .plots import plot_comparison

__all__ = [
    "VARIANTS",
    "get_available_variants",
    "load_variant",
    "run_inference",
    "benchmark_variant",
    "build_comparison_table",
    "plot_comparison",
]
