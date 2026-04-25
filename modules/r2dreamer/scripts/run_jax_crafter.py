"""Crafter shim — crafter, cnn, no curriculum."""
from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="crafter", encoder="cnn", curriculum=None,
        output_dir="output/r2dreamer-crafter",
        wandb_name="r2d-crafter",
        wandb_tags=["crafter", "cnn"],
    )
