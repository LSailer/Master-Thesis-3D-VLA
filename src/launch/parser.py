"""The launch package's single argparse parser.

One parser covers every workflow: the run-level selection (which mode, env,
adapter, curriculum), the training knobs and the evaluation knobs. A flag the
running workflow does not read simply stays at its default, which is what lets
``src.main`` parse the command line once and dispatch on ``--mode`` instead of
handing an unparsed remainder from one parser to the next.

Boolean flags are plain ``store_true``: the Slurm launcher renders a YAML
``key: true`` as the bare ``--flag`` and omits the line for ``false``
(``scripts/slurm/launch.py``), so no flag ever receives a boolean value token.
"""

import argparse

from src.adapters import ADAPTERS
from src.configs.config import LATENT_PRESETS

ENVS = ("habitat", "crafter")
MODES = ("train", "eval")


def build_parser() -> argparse.ArgumentParser:
    """Build the one parser every entry point uses.

    Flags are grouped by what they drive, not by which workflow reads them: a
    training knob is defined here even on an evaluation launch, so a mistyped
    workflow combination fails at parse time instead of being ignored.

    Returns:
        The parser. ``--mode`` selects the workflow and defaults to ``train``,
        so a bare launch trains. The mode value doubles as the Habitat episode
        split, which is why the eval workflow is named ``eval`` and not
        ``evaluate``.
    """
    p = argparse.ArgumentParser(
        description="Run the 3D ObjectNav pipeline from a single entry point."
    )

    # --- Which workflow, on which env, with which variant ---
    p.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=list(MODES),
        help="Which workflow to run: train (default) rolls out the train "
        "episode set and updates the agent, eval scores a checkpoint on the "
        "eval episode set. The value is also the Habitat episode split.",
    )
    p.add_argument("--env", default="habitat", choices=list(ENVS))
    # Default is the appearance-only control baseline.
    p.add_argument("--adapter", default="rgb", choices=sorted(ADAPTERS.keys()))
    # L1 is the entry level; crafter simply ignores the value, so a default is
    # safe for every env (decision: no curriculum validator).
    p.add_argument(
        "--curriculum",
        type=str,
        default="L1",
        choices=["L1", "L2", "L3", "L4"],
        help="Habitat curriculum level name (L1..L4). Ignored by crafter.",
    )

    p.add_argument(
        "--render_resolution",
        type=int,
        default=None,
        help="Override the env render resolution. None lets the adapter decide "
        "(RENDER_RESOLUTION: 518 for the VGGT arms, 64 for the RGB baseline). "
        "A hard default here would shadow every adapter's own declaration.",
    )
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_episode_steps",
        type=int,
        default=500,
        help="Per-episode step cap, enforced by the env in both modes.",
    )

    # --- Training budget, logging, W&B ---
    p.add_argument("--steps", type=int, default=2_400_000)
    p.add_argument("--prefill", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument("--checkpoint_every", type=int, default=50_000)
    p.add_argument("--wandb_project", type=str, default="3d-vla-objectnav")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated tags (appended to preset defaults)",
    )
    p.add_argument(
        "--mlp_layers",
        type=int,
        default=None,
        help="Depth of the composite encoder's MLP branch: number of hidden "
        "Dense->RMSNorm->SiLU blocks before the linear readout. None keeps "
        "the branch default (1). The experiment runs pass 3 to match "
        "R2Dreamer's native encoder.mlp.layers (3D-52). Only affects "
        "variants that route a field to the MLP branch.",
    )

    # --- Knobs a single adapter variant consumes (its ``RUN_FLAGS`` claim
    # them). These reach no config: an unclaimed one would be a flag the run
    # ignores, so ``src.main.make_adapter`` rejects it rather than starting the
    # job. Default ``None``, because that is how "unset" is recognized there.
    p.add_argument(
        "--pointcloud_dump_steps",
        type=str,
        default=None,
        help="For rgb_house_voxels*: comma-separated adapter steps (e.g. "
        "'500000,1000000') at which every non-empty house map is saved as a "
        "binary PLY under <output_dir>/pointcloud_dumps/step_<N>/<scene>/. An "
        "extra snapshot is written at the end of the first episode. Comma "
        "string rather than a list because the SLURM launcher renders scalars "
        "only. Diagnostics; unset disables.",
    )

    # --- Resume, train-time video, exploration ---
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="W&B run-id to reattach to (resume='must')",
    )
    p.add_argument(
        "--video_log_every",
        type=int,
        default=0,
        help="Log one Habitat episode video every N train steps",
    )
    p.add_argument(
        "--video_log_episodes",
        type=int,
        default=0,
        help="Number of Habitat train episodes to record per interval",
    )
    p.add_argument(
        "--act_entropy",
        type=float,
        default=3e-2,
        help="Actor entropy coefficient. 3e-2 is the Habitat 4-action ObjectNav "
        "baseline; the DreamerV3 paper default 3e-4 (tuned for 17-action "
        "Crafter) collapses the policy here.",
    )

    # --- Diagnostic overfit-one-batch branch (Karpathy step 3) ---
    p.add_argument(
        "--overfit_one_batch",
        action="store_true",
        help="After prefill, freeze a single sampled batch and call "
        "train_step on it for --overfit_steps iterations instead of entering "
        "the run loop. Disables env rollouts and checkpointing.",
    )
    p.add_argument(
        "--overfit_steps",
        type=int,
        default=1000,
        help="Number of gradient steps to run on the frozen batch "
        "when --overfit_one_batch is set.",
    )
    p.add_argument(
        "--overfit_batch_size",
        type=int,
        default=1,
        help="B for the frozen overfit batch (default 1).",
    )
    p.add_argument(
        "--overfit_seq_len",
        type=int,
        default=8,
        help="T for the frozen overfit batch (default 8).",
    )
    p.add_argument(
        "--overfit_min_loss_drop",
        type=float,
        default=0.20,
        help="Fail --overfit_one_batch unless total_loss drops by this "
        "fraction over the frozen-batch run (default 0.20).",
    )

    # --- Loss-scale overrides (Protocol C). None => keep config default. ---
    p.add_argument(
        "--actor_loss_weight",
        type=float,
        default=None,
        help="Override cfg.scale_policy. Set 0 to disable actor loss.",
    )
    p.add_argument(
        "--value_loss_weight",
        type=float,
        default=None,
        help="Override cfg.scale_value. Set 0 to disable critic loss.",
    )
    p.add_argument(
        "--repval_loss_weight",
        type=float,
        default=None,
        help="Override cfg.scale_repval. Set 0 to disable replay value loss.",
    )
    # Protocol D toggle
    p.add_argument(
        "--barlow_grad_to_encoder",
        action="store_true",
        help="Let Barlow Twins gradient reach the encoder (removes the "
        "stop_gradient at agent._loss_fn). Default off matches "
        "PyTorch reference.",
    )
    # Small-batch knobs for the overfit run
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override cfg.batch_size (production default 16).",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Override cfg.seq_len (production default 64).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override cfg.lr (production default 4e-5).",
    )
    p.add_argument(
        "--train_ratio",
        type=int,
        default=None,
        help="Override cfg.train_ratio (production default 512). Lower "
        "values bound train_step count for short smoke runs.",
    )
    p.add_argument(
        "--buffer_capacity",
        "--buffer-capacity",
        type=int,
        default=None,
        help="Override cfg.buffer_capacity for replay-capacity ablations.",
    )

    # --- Latent-size ablation (3D-50) ---
    p.add_argument(
        "--latent_preset",
        choices=tuple(LATENT_PRESETS),
        default="12m",
        help="Model-size preset from the R2-Dreamer table. Scales RSSM shape, "
        "CNN depth, and head MLP width. Explicit "
        "--deter_size/--stoch_classes/--stoch_discrete override the RSSM "
        "shape.",
    )
    p.add_argument(
        "--deter_size",
        type=int,
        default=None,
        help="Override cfg.deter_size (explicit; wins over --latent_preset).",
    )
    p.add_argument(
        "--stoch_classes",
        type=int,
        default=None,
        help="Override cfg.stoch_classes (explicit; wins over --latent_preset).",
    )
    p.add_argument(
        "--stoch_discrete",
        type=int,
        default=None,
        help="Override cfg.stoch_discrete (explicit; wins over --latent_preset).",
    )

    # --- Debug decoder probe (3D-51) ---
    p.add_argument(
        "--decoder",
        action="store_true",
        help="Train a stop-gradient ConvDecoder probe for reconstruction "
        "logging (3D-51).",
    )
    p.add_argument(
        "--scale_decoder",
        type=float,
        default=None,
        help="Override cfg.scale_decoder (decoder-only reconstruction weight).",
    )

    # --- Precision ---
    p.add_argument(
        "--compute_dtype",
        choices=["float32", "bfloat16", "bf16", "float16", "fp16"],
        default=None,
        help="Override cfg.compute_dtype for large encoder activations. "
        "Use bfloat16 for full-token memory ablations.",
    )
    p.add_argument(
        "--full_bf16",
        action="store_true",
        help="Run the whole JAX model (encoders, RSSM, heads) in "
        "cfg.compute_dtype instead of only the token transformer - mixed "
        "precision with float32 master params and float32-pinned logits.",
    )

    # --- Evaluation ---
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--random",
        action="store_true",
        help="Use a random agent instead of a checkpoint",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Eval episode budget: the run loop stops after this many "
        "episodes in eval mode.",
    )
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--render_topdown", action="store_true")
    p.add_argument(
        "--log_video_episodes",
        type=int,
        default=0,
        help="Number of eval episodes to log as W&B videos",
    )
    return p
