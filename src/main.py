"""Single public entry point for training and evaluation workflows."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.environments.crafter import CrafterEnv
from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv
from src.r2dreamer.launch.evaluate import evaluate
from src.r2dreamer.launch.train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 3D ObjectNav pipeline from a single entry point."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train R2Dreamer end to end.")
    train_parser.add_argument(
        "--env", default="habitat", choices=["habitat", "crafter"]
    )
    train_parser.add_argument(
        "--encoder",
        default="cnn",
        choices=[
            "cnn",
            "vggt",
            "vggt_aggregator_mlp",
            "vggt_wp_dense_cnn",
            "vggt_wp_cp_64",
            "hybrid",
        ],
    )
    train_parser.add_argument("--curriculum", default=None)
    train_parser.add_argument("--curriculum_path", default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent.")
    eval_parser.add_argument("--env", default="habitat", choices=["habitat"])
    eval_parser.add_argument(
        "--encoder",
        default="cnn",
        choices=[
            "cnn",
            "vggt",
            "vggt_aggregator_mlp",
            "vggt_wp_dense_cnn",
            "vggt_wp_cp_64",
            "hybrid",
        ],
    )
    eval_parser.add_argument("--curriculum", default="")
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--output_dir", default=None)

    return parser



def make_env(
    env: str,
    *,
    curriculum: str,
    mode: str = "train",
) -> HabitatObjectNavEnv | CrafterEnv:
    """Build an env instance for notebook / CLI experimentation."""
    if env == "habitat":
        return HabitatObjectNavEnv(
            config=HabitatEnvConfig(
                curriculum=curriculum,
                mode=mode,
            ),
            seed=0,
        )
    if env == "crafter":
        return CrafterEnv(seed=0)
    raise ValueError(f"Unknown environment: {env}")

def main(argv: Sequence[str] | None = None) -> object:
    """Dispatch to train/evaluate while forwarding workflow-specific flags."""
    parser = _build_parser()
    args, rest = parser.parse_known_args(list(argv) if argv is not None else None)

    # Instant of Environment
    env = make_env(args.env, curriculum=args.curriculum,  mode=args.mode)
    if args.command == "train":
        # Trainer Object
        # run function
        return train(
            env=args.env,
            encoder=args.encoder,
            curriculum=args.curriculum,
            argv=rest,
        )
    if args.command == "evaluate":
        return evaluate(
            env=args.env,
            encoder=args.encoder,
            curriculum=args.curriculum,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            argv=rest,
        )

    parser.error(f"Unknown command: {args.command}")
    return None


if __name__ == "__main__":
    main(sys.argv[1:])
