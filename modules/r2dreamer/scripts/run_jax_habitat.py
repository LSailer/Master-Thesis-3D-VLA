"""L1 CNN shim — habitat, cnn, L1 (1 house, chair only)."""
from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="cnn", curriculum="L1",
        output_dir="output/r2dreamer-curriculum-l1",
        wandb_name="r2d-L1-1house-chair",
        wandb_tags=["curriculum", "level1", "1house", "chair-only", "no-goal"],
    )
