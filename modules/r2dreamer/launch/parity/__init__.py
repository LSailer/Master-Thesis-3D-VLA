"""Parity and benchmark sub-package for r2dreamer."""
from modules.r2dreamer.launch.parity.train_parity import run as train_parity_run
from modules.r2dreamer.launch.parity.benchmark import run as benchmark_run

__all__ = ["train_parity_run", "benchmark_run"]
