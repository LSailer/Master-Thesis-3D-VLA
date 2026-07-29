"""Trainer-owned R2Dreamer loop, replay, logging, and resume configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainerConfig:
    """Controls the training loop outside ``R2DreamerAgent``.

    This config owns replay capacity, sequence sampling, prefill, W&B, resume,
    and diagnostic loops. It intentionally does not own neural network
    architecture, optimizer, or loss fields.
    """

    output_dir: str = "output/runs/r2dreamer"
    total_steps: int = 10_000_000
    seed: int = 43

    # --- Replay / sampling cadence ---
    buffer_capacity: int = 500_000
    prefill_steps: int = 5000
    batch_size: int = 16
    seq_len: int = 64
    train_ratio: int = 512

    # --- Logging/checkpointing ---
    log_every: int = 250
    checkpoint_every: int = 50_000

    # When True, a fully completed run hard-exits the process (os._exit(0))
    # after the final checkpoint, MANIFEST, and W&B are flushed — skipping
    # habitat_sim's GL teardown, which SIGABRTs ("no current context") on some
    # magnum builds and would otherwise poison the exit code of a successful
    # run. Set by the SLURM launcher (env R2DREAMER_HARD_EXIT_ON_FINISH=1);
    # left False for notebook/test callers so they keep the normal close() path
    # and real failures still surface a non-zero exit.
    hard_exit_on_finish: bool = False

    # WandB (None = disabled)
    wandb_project: str | None = "3d-vla-objectnav"
    wandb_name: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["r2dreamer"])
    # Resume an existing W&B run (e.g. "87u0l6dy"). Requires the run to exist.
    wandb_id: str | None = None
    video_log_every: int = 0
    video_log_episodes: int = 0

    # Resume from checkpoint (.pkl produced by save_checkpoint). When set,
    # restores agent.{params, opt_state, slow_critic_params, ema_state} and
    # offsets the train loop to start at the checkpoint's step.
    resume_from: str | None = None

    # --- Karpathy step-3 diagnostic: overfit a single sampled batch ---
    # When True, the run does the normal prefill, then samples one batch
    # (overfit_batch_size, overfit_seq_len) once, freezes it, and runs
    # agent.train_step on that same batch for overfit_steps iterations.
    # No env rollouts, no validation, no checkpointing.
    overfit_one_batch: bool = False
    overfit_steps: int = 1000
    overfit_batch_size: int = 1
    overfit_seq_len: int = 8
    overfit_min_loss_drop: float = 0.20
