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
    # Val-Episode-Loop (3D-36). 0 disables. Default 50_000 matches the
    # checkpoint cadence so val signals land alongside checkpoints.
    p.add_argument("--val_every", type=int, default=50_000,
                   help="Run a deterministic val-episode loop every N steps (0 disables)")
    p.add_argument("--val_episodes", type=int, default=50,
                   help="Episodes per val-loop trigger")
    p.add_argument("--val_video_episodes", type=int, default=1,
                   help="Number of val episodes to record as W&B videos per trigger")
    p.add_argument("--val_max_episode_steps", type=int, default=500,
                   help="Per-episode step cap inside the val loop")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--wandb_id", type=str, default=None,
                   help="W&B run-id to reattach to (resume='must')")
    p.add_argument("--video_log_every", type=int, default=25_000,
                   help="Log one Habitat episode video every N train steps")
    p.add_argument("--video_log_episodes", type=int, default=1,
                   help="Number of Habitat train episodes to record per interval")
    p.add_argument("--act_entropy", type=float, default=3e-2,
                   help="Actor entropy coefficient. 3e-2 is the Habitat 4-action ObjectNav "
                        "baseline; the DreamerV3 paper default 3e-4 (tuned for 17-action Crafter) "
                        "collapses the policy here.")
    # --- Diagnostic / overfit-one-batch knobs (Karpathy step 3) ---
    p.add_argument("--overfit_one_batch", action="store_true",
                   help="After prefill, freeze a single sampled batch and call "
                        "train_step on it for --overfit_steps iterations. Disables "
                        "env rollouts, validation and checkpointing.")
    p.add_argument("--overfit_steps", type=int, default=1000,
                   help="Number of gradient steps to run on the frozen batch "
                        "when --overfit_one_batch is set.")
    p.add_argument("--overfit_batch_size", type=int, default=1,
                   help="B for the frozen overfit batch (default 1).")
    p.add_argument("--overfit_seq_len", type=int, default=8,
                   help="T for the frozen overfit batch (default 8).")
    p.add_argument("--overfit_min_loss_drop", type=float, default=0.20,
                   help="Fail --overfit_one_batch unless total_loss drops by this "
                        "fraction over the frozen-batch run (default 0.20).")
    # Loss-scale overrides (Protocol C). None => keep config default.
    p.add_argument("--actor_loss_weight", type=float, default=None,
                   help="Override cfg.scale_policy. Set 0 to disable actor loss.")
    p.add_argument("--value_loss_weight", type=float, default=None,
                   help="Override cfg.scale_value. Set 0 to disable critic loss.")
    p.add_argument("--repval_loss_weight", type=float, default=None,
                   help="Override cfg.scale_repval. Set 0 to disable replay value loss.")
    # Protocol D toggle
    p.add_argument("--barlow_grad_to_encoder", action="store_true",
                   help="Let Barlow Twins gradient reach the encoder (removes the "
                        "stop_gradient at agent._loss_fn). Default off matches "
                        "PyTorch reference.")
    # Small-batch knobs for the overfit run
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override cfg.batch_size (production default 16).")
    p.add_argument("--seq_len", type=int, default=None,
                   help="Override cfg.seq_len (production default 64).")
    p.add_argument("--lr", type=float, default=None,
                   help="Override cfg.lr (production default 4e-5).")
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
    p.add_argument("--log_video_episodes", type=int, default=3,
                   help="Number of eval episodes to log as W&B videos")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="W&B project for eval video logging")
    p.add_argument("--wandb_name", type=str, default=None)
    return p
