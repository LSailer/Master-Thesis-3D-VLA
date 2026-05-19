"""Benchmark shim — JAX R2 vs PT R2 vs PT DreamerV3 on Crafter."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.main import run_parity_benchmark as run

if __name__ == "__main__":
    run(argv=None)
