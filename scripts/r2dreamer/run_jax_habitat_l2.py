"""L2 CNN shim — habitat, cnn, L2 (1 house, 6 goals)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.main import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="cnn", curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2",
        wandb_name="r2d-L2-buffix",
        wandb_tags=["curriculum", "level2", "1house", "6goals", "buffer-fix", "rerun"],
    )
