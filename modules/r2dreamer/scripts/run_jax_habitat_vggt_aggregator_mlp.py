"""L1 Variant 1 shim — habitat + VGGT aggregator MLP encoder."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat",
        encoder="vggt_aggregator_mlp",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-aggregator-mlp",
        wandb_name="variant-1-aggregator-mlp",
        wandb_tags=[
            "curriculum",
            "level1",
            "1house",
            "chair-only",
            "vggt",
            "aggregator-mlp",
            "variant-1",
            "jax",
            "3d-encoder",
        ],
    )
