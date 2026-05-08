"""L2 VGGT-FiLM shim — habitat, vggt_film_v1, L2 (1 house, 6 goals, 3D encoder)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat",
        encoder="vggt_film_v1",
        curriculum="L2",
        output_dir="output/r2dreamer-curriculum-l2-vggt-film",
        wandb_name="vggt_film_v1_l2",
        wandb_tags=[
            "curriculum",
            "level2",
            "1house",
            "6goals",
            "vggt",
            "vggt_film_v1",
            "film",
            "jax",
            "3d-encoder",
            "diagnostic-metrics",
        ],
    )
