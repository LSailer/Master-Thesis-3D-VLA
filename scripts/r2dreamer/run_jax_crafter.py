"""Crafter shim — crafter, cnn, no curriculum."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.main import train

if __name__ == "__main__":
    train(
        env="crafter", encoder="cnn", curriculum=None,
        output_dir="output/runs/r2dreamer-crafter",
        wandb_name="r2d-crafter",
        wandb_tags=["crafter", "cnn"],
    )
