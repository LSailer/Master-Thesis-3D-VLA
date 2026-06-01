"""L1 VGGT shim — habitat, WP+CP MLP at a 64x64 world-point grid (3D-52/3D-53).

Same MLP / camera-pose / replay setup as run_jax_habitat_vggt.py, but VGGT's
dense point map is pooled to 64x64 (obs = 64*64*3 + 9 = 12297) instead of 37x37.
Controlled resolution ablation vs the 37x37 WP+CP MLP run.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.main import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="vggt_wp_cp_64", curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-wp-cp-64",
        wandb_name="wp-cp-mlp-64",
        wandb_tags=["curriculum", "level1", "1house", "chair-only", "vggt",
                    "wp-cp", "mlp-3layer", "wp-64", "3d-52", "jax", "3d-encoder"],
    )
