"""Wrapper that runs external/r2dreamer/train.py with W&B logging via TensorBoard auto-sync.

We can't edit external/ (vendored upstream, hook-blocked), so we wrap it: call
wandb.init(sync_tensorboard=True) BEFORE invoking the upstream main(). wandb
monkey-patches torch.utils.tensorboard.SummaryWriter at init time, so the
upstream Logger's writes are mirrored to W&B in real-time without any changes
to the external code.

Usage (from repo root, with the external venv active):
  python modules/r2dreamer/launch/run_external_crafter.py \
      --wandb_project r2dreamer-parity-crafter \
      --wandb_name external-torch-crafter-ref-s0-12345 \
      --wandb_tags ab,parity,external,torch,crafter \
      --logdir output/runs/r2dreamer-parity-crafter/external-12345 \
      --seed 0 \
      -- env=crafter model=size12M
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXT_DIR = REPO_ROOT / "external" / "r2dreamer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", required=True)
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--wandb_tags", default="", help="comma-separated")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", default="crafter")
    parser.add_argument("hydra_overrides", nargs="*",
                        help="extra Hydra overrides forwarded to train.py (e.g. model=size12M trainer.steps=2000)")
    args = parser.parse_args()

    logdir = Path(args.logdir).expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(EXT_DIR))

    import wandb
    tags = [t for t in args.wandb_tags.split(",") if t]
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        tags=tags,
        group=args.wandb_group,
        mode=args.wandb_mode,
        sync_tensorboard=True,
        dir=str(logdir),
        config={
            "wrapper": "run_external_crafter.py",
            "env": args.env,
            "seed": args.seed,
            "hydra_overrides": args.hydra_overrides,
        },
    )
    print(f"[wrapper] wandb run: {run.url}")

    hydra_argv = [
        "train.py",
        f"env={args.env}",
        f"seed={args.seed}",
        f"logdir={logdir}",
    ]
    hydra_argv.extend(args.hydra_overrides)
    print(f"[wrapper] launching: {' '.join(hydra_argv)}")

    saved_argv = sys.argv
    saved_cwd = os.getcwd()
    sys.argv = hydra_argv
    try:
        os.chdir(EXT_DIR)
        # runpy executes train.py as if it were invoked as `python train.py`.
        # `from train import main; main()` doesn't work because Hydra's
        # `config_path` resolution depends on the caller being __main__.
        import runpy
        runpy.run_path(str(EXT_DIR / "train.py"), run_name="__main__")
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        # Call wandb.finish() explicitly here (not via atexit) to avoid a
        # known wandb 0.26.x bug where the Go-runtime tensorboard channel
        # panics during interpreter shutdown ("close of closed channel").
        # Calling finish in a controlled order during normal flow sidesteps it.
        try:
            wandb.finish()
        except Exception as e:
            print(f"[wrapper] wandb.finish raised (non-fatal): {e}")


if __name__ == "__main__":
    main()
