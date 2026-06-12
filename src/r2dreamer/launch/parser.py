"""Shared argparse helpers for the launch package."""

import argparse


def _add_basic_train_args(p: argparse.ArgumentParser) -> None:
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
    p.add_argument("--mlp_layers", type=int, default=None,
                   help="Depth of the VGGT MLP encoders (wp_cp + aggregator): number "
                        "of hidden Dense->RMSNorm->SiLU blocks before the linear readout. "
                        "None keeps the config default (1). The experiment runs pass 3 to "
                        "match R2Dreamer's native encoder.mlp.layers (3D-52). Only valid "
                        "for VGGT MLP encoders; CNN/dense-WP conv encoders require the "
                        "default value (1).")


def _add_val_train_args(p: argparse.ArgumentParser) -> None:
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


def _add_resume_video_train_args(p: argparse.ArgumentParser) -> None:
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


def _add_overfit_train_args(p: argparse.ArgumentParser) -> None:
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


def _add_loss_override_train_args(p: argparse.ArgumentParser) -> None:
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
    p.add_argument("--train_ratio", type=int, default=None,
                   help="Override cfg.train_ratio (production default 512). Lower "
                        "values bound train_step count for short smoke runs.")
    p.add_argument("--buffer_capacity", "--buffer-capacity", type=int, default=None,
                   help="Override cfg.buffer_capacity for replay-capacity ablations.")


def _add_latent_decoder_train_args(p: argparse.ArgumentParser) -> None:
    # --- Latent-size ablation (3D-50) ---
    p.add_argument("--latent_preset", choices=["small", "default", "large"],
                   default="default",
                   help="Latent-size ablation preset (3D-50). Explicit "
                        "--deter_size/--stoch_classes/--stoch_discrete win over it.")
    p.add_argument("--deter_size", type=int, default=None,
                   help="Override cfg.deter_size (explicit; wins over --latent_preset).")
    p.add_argument("--stoch_classes", type=int, default=None,
                   help="Override cfg.stoch_classes (explicit; wins over --latent_preset).")
    p.add_argument("--stoch_discrete", type=int, default=None,
                   help="Override cfg.stoch_discrete (explicit; wins over --latent_preset).")
    # --- Co-trained decoder (3D-51) ---
    p.add_argument("--decoder", action="store_true",
                   help="Co-train a ConvDecoder + add a reconstruction loss (3D-51).")
    p.add_argument("--scale_decoder", type=float, default=None,
                   help="Override cfg.scale_decoder (reconstruction loss weight).")
    # --- Hybrid VGGT-branch MLP knobs ---
    p.add_argument("--mlp_vggt_hidden", type=int, default=None,
                   help="Override cfg.mlp_vggt_hidden (hybrid WP/CP MLP width).")
    p.add_argument("--mlp_vggt_layers", type=int, default=None,
                   help="Override cfg.mlp_vggt_layers (hybrid WP/CP MLP depth).")


def _add_token_transformer_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vggt_token_transformer_layers", type=int, default=None,
                   help="Override cfg.vggt_token_transformer_layers for "
                        "vggt_agg_token_transformer.")
    p.add_argument("--vggt_token_transformer_heads", type=int, default=None,
                   help="Override cfg.vggt_token_transformer_heads for "
                        "vggt_agg_token_transformer.")
    p.add_argument("--vggt_token_projection_dim", type=int, default=None,
                   help="Override cfg.vggt_token_projection_dim before token attention.")
    p.add_argument("--vggt_token_transformer_mlp_ratio", type=int, default=None,
                   help="Override cfg.vggt_token_transformer_mlp_ratio.")
    p.add_argument("--vggt_drop_register_tokens", action="store_true",
                   help="Drop the 4 VGGT register tokens in the token Transformer "
                        "ablation. Default keeps registers for 3D-75.")


def _build_parser_train() -> argparse.ArgumentParser:
    """Build CLI parser for train(). Union of flags from all r2dreamer entrypoints."""
    p = argparse.ArgumentParser(add_help=True)
    _add_basic_train_args(p)
    _add_val_train_args(p)
    _add_resume_video_train_args(p)
    _add_overfit_train_args(p)
    _add_loss_override_train_args(p)
    _add_latent_decoder_train_args(p)
    _add_token_transformer_train_args(p)
    return p


def _build_parser_eval() -> argparse.ArgumentParser:
    """Build CLI parser for evaluate()."""
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--encoder", type=str, default=None,
                   choices=[
                       "cnn", "vggt", "vggt_aggregator_mlp",
                       "vggt_agg_token_transformer", "vggt_wp_dense_cnn",
                       "vggt_wp_cp_64", "hybrid",
                   ])
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
