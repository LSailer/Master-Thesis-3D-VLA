"""L1 Hybrid shim — habitat + CNN(RGB) + gated MLP(WP/CP) hybrid encoder (3D-50/51/52)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.main import train

if __name__ == "__main__":
    train(
        env="habitat",
        encoder="hybrid",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-hybrid",
        wandb_name="hybrid-cnn-vggt",
        wandb_tags=[
            "curriculum",
            "level1",
            "1house",
            "chair-only",
            "vggt",
            "hybrid",
            "cnn",
            "wp-cp",
            "3d-encoder",
            "jax",
        ],
    )
