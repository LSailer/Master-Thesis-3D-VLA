"""L4 VGGT shim — habitat, vggt, L4 (10 houses, 6 goals, full curriculum, 3D encoder)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="vggt", curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4-vggt",
        wandb_name="r2d-L4-vggt",
        wandb_tags=["curriculum", "level4", "10houses", "6goals", "vggt", "jax", "3d-encoder"],
    )
