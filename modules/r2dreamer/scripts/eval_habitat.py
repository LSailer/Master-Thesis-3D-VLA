"""Eval shim — evaluate a checkpoint on Habitat ObjectNav."""
from modules.r2dreamer.launch.evaluate import evaluate

if __name__ == "__main__":
    evaluate(
        env="habitat", encoder="cnn", curriculum=None,
        output_dir="output/r2dreamer-eval",
    )
