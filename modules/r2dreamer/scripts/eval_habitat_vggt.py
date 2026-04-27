"""Eval shim — evaluate a VGGT R2Dreamer checkpoint on Habitat ObjectNav."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.evaluate import evaluate

if __name__ == "__main__":
    evaluate(
        env="habitat", encoder="vggt", curriculum=None,
        output_dir="output/runs/r2dreamer-eval-vggt",
    )
