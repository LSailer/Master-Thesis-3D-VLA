"""L4 CNN shim — habitat, cnn, L4 (10 houses, 6 goals, full curriculum)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="cnn", curriculum="L4",
        output_dir="output/r2dreamer-curriculum-l4",
        wandb_name="r2d-L4-buffix",
        wandb_tags=["curriculum", "level4", "10houses", "6goals", "buffer-fix", "rerun"],
    )
