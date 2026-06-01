"""L1 VGGT shim — habitat, full-resolution world-point CNN encoder (3D-53).

Feeds the dense 518x518x3 VGGT world-point map (no 37x37 pooling) into a conv
encoder that treats XYZ as a 3-channel image. Counterpart to the WP/CP MLP run.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.main import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="vggt_wp_dense_cnn", curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-wp-dense",
        wandb_name="vggt_wp_dense_cnn",
        wandb_tags=["curriculum", "level1", "1house", "chair-only", "vggt",
                    "wp-dense", "full-res-518", "cnn", "jax", "3d-encoder"],
    )
