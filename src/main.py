"""Single public entry point for training and evaluation workflows."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.r2dreamer.launch.evaluate import evaluate
from src.r2dreamer.launch.train import train


def run_parity_training(argv: list[str] | None = None) -> object:
    """Run the R2Dreamer parity-training command."""
    from src.r2dreamer.launch.parity.train_parity import run

    return run(argv=argv)


def run_parity_benchmark(argv: list[str] | None = None) -> object:
    """Run the R2Dreamer parity benchmark command."""
    from src.r2dreamer.launch.parity.benchmark import run

    return run(argv=argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 3D ObjectNav pipeline from a single entry point."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train R2Dreamer end to end.")
    train_parser.add_argument("--env", default="habitat", choices=["habitat", "crafter"])
    train_parser.add_argument(
        "--encoder",
        default="cnn",
        choices=["cnn", "vggt", "vggt_aggregator_mlp", "vggt_wp_dense_cnn", "vggt_wp_cp_64", "hybrid"],
    )
    train_parser.add_argument("--curriculum", default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent.")
    eval_parser.add_argument("--env", default="habitat", choices=["habitat"])
    eval_parser.add_argument(
        "--encoder",
        default="cnn",
        choices=["cnn", "vggt", "vggt_aggregator_mlp", "vggt_wp_dense_cnn", "vggt_wp_cp_64", "hybrid"],
    )
    eval_parser.add_argument("--curriculum", default=None)
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--output_dir", default=None)

    subparsers.add_parser("parity-train", help="Run the JAX/PyTorch parity trainer.")
    subparsers.add_parser("parity-benchmark", help="Run parity benchmark comparisons.")
    return parser


def main(argv: Sequence[str] | None = None) -> object:
    """Dispatch to train/evaluate while forwarding workflow-specific flags."""
    parser = _build_parser()
    args, rest = parser.parse_known_args(list(argv) if argv is not None else None)

    if args.command == "train":
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
    if args.command == "parity-train":
        return run_parity_training(argv=rest)
    if args.command == "parity-benchmark":
        return run_parity_benchmark(argv=rest)

    parser.error(f"Unknown command: {args.command}")
    return None


if __name__ == "__main__":
    main(sys.argv[1:])
