"""Eval shim — evaluate a checkpoint on Habitat ObjectNav."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.main import evaluate

if __name__ == "__main__":
    evaluate(
        env="habitat", adapter="rgb", curriculum=None,
        output_dir="output/runs/r2dreamer-eval",
    )
