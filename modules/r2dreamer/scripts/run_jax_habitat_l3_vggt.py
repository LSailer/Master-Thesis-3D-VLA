"""L3 VGGT shim — habitat, vggt, L3 (10 houses, chair only, 3D encoder)."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="vggt", curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3-vggt",
        wandb_name="r2d-L3-vggt",
        wandb_tags=["curriculum", "level3", "10houses", "chair-only", "vggt", "jax", "3d-encoder"],
    )
