"""L4 VGGT-FiLM shim — habitat, vggt_film_v1, L4 (10 houses, 6 goals, full curriculum)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat",
        encoder="vggt_film_v1",
        curriculum="L4",
        output_dir="output/r2dreamer-curriculum-l4-vggt-film",
        wandb_name="vggt_film_v1_l4",
        wandb_tags=[
            "curriculum",
            "level4",
            "10houses",
            "6goals",
            "vggt",
            "vggt_film_v1",
            "film",
            "jax",
            "3d-encoder",
            "diagnostic-metrics",
        ],
    )
