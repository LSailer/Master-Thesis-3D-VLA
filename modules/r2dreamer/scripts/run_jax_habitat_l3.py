"""L3 CNN shim — habitat, cnn, L3 (10 houses, chair only)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="cnn", curriculum="L3",
        output_dir="output/r2dreamer-curriculum-l3",
        wandb_name="r2d-L3-buffix",
        wandb_tags=["curriculum", "level3", "10houses", "chair-only", "buffer-fix", "rerun"],
    )
