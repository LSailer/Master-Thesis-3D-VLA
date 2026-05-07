"""L1 VGGT-FiLM shim — habitat, vggt_film_v1, L1 (1 house, chair only, 3D encoder)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat",
        encoder="vggt_film_v1",
        curriculum="L1",
        output_dir="output/r2dreamer-curriculum-l1-vggt-film",
        wandb_name="vggt_film_v1",
        wandb_tags=[
            "curriculum",
            "level1",
            "1house",
            "chair-only",
            "vggt",
            "vggt_film_v1",
            "film",
            "jax",
            "3d-encoder",
            "diagnostic-metrics",
        ],
    )
