"""L1 CNN shim — habitat, cnn, L1 (1 house, chair only)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.main import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="cnn", curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1",
        wandb_name="r2d-L1-1house-chair",
        wandb_tags=["curriculum", "level1", "1house", "chair-only", "no-goal"],
    )
