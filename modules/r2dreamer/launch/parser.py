"""Shared argparse helpers for the launch package."""

import argparse


def _build_parser_train() -> argparse.ArgumentParser:
    """Build CLI parser for train(). Union of flags from all r2dreamer entrypoints."""
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--steps", type=int, default=2_400_000)
    p.add_argument("--prefill", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument("--checkpoint_every", type=int, default=50_000)
    p.add_argument("--wandb_project", type=str, default="3d-vla-objectnav")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None,
                   help="Comma-separated tags (appended to shim defaults)")
    # escape hatch: override the shim-hardcoded curriculum path
    p.add_argument("--curriculum_path", type=str, default=None,
                   help="Override shim curriculum path (escape hatch)")
    p.add_argument("--curriculum_mode", type=str, default="train")
    p.add_argument("--render_resolution", type=int, default=518,
                   help="Render resolution for VGGT encoder")
    p.add_argument("--val_data", type=str, default=None)
    p.add_argument("--val_loss_every", type=int, default=10_000)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--wandb_id", type=str, default=None,
                   help="W&B run-id to reattach to (resume='must')")
    p.add_argument("--act_entropy", type=float, default=3e-2,
                   help="Actor entropy coefficient. 3e-2 is the Habitat 4-action ObjectNav "
                        "baseline; the DreamerV3 paper default 3e-4 (tuned for 17-action Crafter) "
                        "collapses the policy here.")
    return p


def _build_parser_eval() -> argparse.ArgumentParser:
    """Build CLI parser for evaluate()."""
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--encoder", type=str, default=None, choices=["cnn", "vggt"])
    p.add_argument("--random", action="store_true",
                   help="Use random agent instead of a checkpoint")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to write results JSON")
    p.add_argument("--seed", type=int, default=42)
    # escape hatch: override shim-hardcoded curriculum
    p.add_argument("--curriculum_path", type=str, default=None,
                   help="Override shim curriculum path (escape hatch)")
    p.add_argument("--render_resolution", type=int, default=None,
                   help="Render resolution (default: 518 for vggt, 64 for cnn)")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--render_topdown", action="store_true")
    return p
