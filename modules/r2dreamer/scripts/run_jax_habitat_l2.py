"""L2 CNN shim — habitat, cnn, L2 (1 house, 6 goals)."""
from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="cnn", curriculum="L2",
        output_dir="output/r2dreamer-curriculum-l2",
        wandb_name="r2d-L2-buffix",
        wandb_tags=["curriculum", "level2", "1house", "6goals", "buffer-fix", "rerun"],
    )
