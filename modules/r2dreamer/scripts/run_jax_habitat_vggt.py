"""L1 VGGT shim — habitat, vggt, L1 (1 house, chair only, 3D encoder)."""
from modules.r2dreamer.launch.train import train

if __name__ == "__main__":
    train(
        env="habitat", encoder="vggt", curriculum="L1",
        output_dir="output/r2dreamer-curriculum-l1-vggt",
        wandb_name="r2d-L1-vggt-buffix",
        wandb_tags=["curriculum", "level1", "1house", "chair-only", "vggt", "3d-encoder", "buffer-fix", "rerun"],
    )
