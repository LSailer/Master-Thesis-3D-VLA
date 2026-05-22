"""Offline R2Dreamer training entry point for the 3D-26 ablation.

Trains the R2Dreamer world model + actor + critic on a pre-computed offline
buffer (`data/offline_buffer/`) — no live env, no VGGT forward pass during
training. The agent ingests `z_*.npz` (wp_cp 4116-d or aggregator 3072-d)
through `OfflineBufferDataset` and calls `agent.train_step` on each sampled
sequence batch.

Fairness contract (from issue 3D-26):
    Only `encoder_type` and `train_seed` may differ across the 6 runs.
    `batch_size=16`, `seq_len=64`, `train_ratio=512`,
    `imagination_horizon=15` are shared. The Aggregator variant's
    in-code overrides (batch=4, seq_len=32, ...) are deliberately
    overridden back to these defaults here.

Usage:
    python scripts/r2dreamer/train_offline_ablation.py \\
        --encoder wp_cp --seed 0 --steps 500000 \\
        --buffer-dir data/offline_buffer \\
        --output-dir output/3d26/wp_cp-seed0 \\
        --wandb-project 3d-vla-objectnav-offline-ablation \\
        --wandb-name wp_cp-seed0

Smoke test (NaN-free 100 grad steps, no W&B):
    python scripts/r2dreamer/train_offline_ablation.py \\
        --encoder wp_cp --seed 0 --steps 100 \\
        --buffer-dir <path> --output-dir /tmp/smoke \\
        --no-wandb --skip-heldout-eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Shared hyperparameters from the 3D-26 spec. Both variants use these — they
# explicitly override the Aggregator variant's smaller-batch defaults so the
# only difference between runs is the encoder + seed.
SHARED_AGENT_HYPERPARAMS = {
    "batch_size": 16,
    "seq_len": 64,
    "imagination_horizon": 15,
    "train_ratio": 512,
    "buffer_capacity": 0,  # unused in offline mode but read by ReplayBuffer init
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--encoder",
        required=True,
        choices=["wp_cp", "aggregator"],
        help="Encoder kind. wp_cp -> z_wp_cp.npz (4116-d); aggregator -> z_aggregator.npz (3072-d).",
    )
    p.add_argument("--seed", type=int, required=True, help="Training seed (one of {0,1,2}).")
    p.add_argument("--steps", type=int, default=500_000, help="Total grad steps.")
    p.add_argument("--buffer-dir", type=str, default="data/offline_buffer")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument(
        "--heldout-eval-every", type=int, default=10_000,
        help="Run held-out eval_loss every N steps. 0 disables in-loop eval.",
    )
    p.add_argument(
        "--heldout-eval-batches", type=int, default=16,
        help="Number of held-out batches averaged per in-loop eval.",
    )
    p.add_argument("--wandb-project", type=str, default="3d-vla-objectnav-offline-ablation")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="3d-26,offline-ablation")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--skip-heldout-eval",
        action="store_true",
        help="Skip both in-loop and final held-out eval (smoke testing).",
    )
    return p


def _encoder_spec(encoder_kind: str) -> dict[str, Any]:
    """Map --encoder to (module_cls, obs_shape, encoder_type) without running JAX."""
    from src.r2dreamer.world_model import encoders as wm_encoders

    if encoder_kind == "wp_cp":
        return {
            "module_cls": wm_encoders.VGGTEncoder,
            "obs_shape": (4116,),
            "encoder_type": "vggt",
        }
    if encoder_kind == "aggregator":
        return {
            "module_cls": wm_encoders.VGGTAggregatorMLPEncoder,
            "obs_shape": (3072,),
            "encoder_type": "vggt_aggregator_mlp",
        }
    raise ValueError(f"unknown encoder: {encoder_kind!r}")


def _save_checkpoint(agent: Any, step: int, output_dir: Path) -> Path:
    import jax
    import jax.numpy as jnp

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:09d}.pkl"
    data = {
        "step": step,
        "params": jax.tree.map(np.array, agent.params),
        "opt_state": jax.tree.map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
            agent.opt_state,
        ),
        "slow_critic_params": jax.tree.map(np.array, agent.slow_critic_params),
        "ema_state": jax.tree.map(np.array, agent.ema_state),
    }
    with path.open("wb") as f:
        pickle.dump(data, f)
    return path


def _convert_batch(batch: dict, num_actions: int) -> dict:
    """Replay-buffer batch -> agent train-step format (same as Trainer.convert_batch)."""
    import jax

    return {
        "obs": batch["obs"],
        "actions": jax.nn.one_hot(batch["actions"], num_actions),
        "rewards": batch["rewards"],
        "is_first": batch["is_first"],
        "is_last": batch["dones"],
        "is_terminal": batch["terminals"],
    }


def _aggregate_heldout(
    agent: Any, heldout_dataset: Any, num_actions: int, rng_key: Any, max_batches: int,
) -> dict[str, float]:
    """Run eval_loss on a deterministic slice of the heldout set."""
    import jax

    sums: dict[str, float] = {}
    n_batches = 0
    for raw in heldout_dataset.iter_heldout_batches(
        SHARED_AGENT_HYPERPARAMS["batch_size"],
        SHARED_AGENT_HYPERPARAMS["seq_len"],
        max_batches=max_batches,
    ):
        rng_key, sub = jax.random.split(rng_key)
        metrics = agent.eval_loss(_convert_batch(raw, num_actions), sub)
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n_batches += 1
    if n_batches == 0:
        return {}
    return {f"heldout/{k}": v / n_batches for k, v in sums.items()}


def _flush_csv(path: Path, rows: list[tuple[int, str, float]]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["step", "metric", "value"])
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defer JAX import until after argparse so --help works without a GPU.
    import jax

    from src.buffer.offline_buffer_dataset import OfflineBufferDataset
    from src.r2dreamer.agent import R2DreamerAgent
    from src.r2dreamer.config import R2DreamerConfig

    print(f"3D-26 offline R2Dreamer: encoder={args.encoder} seed={args.seed} steps={args.steps}")
    print(f"  buffer_dir={args.buffer_dir}")
    print(f"  output_dir={output_dir}")
    print(f"  JAX devices: {jax.devices()}")

    train_ds = OfflineBufferDataset(
        args.buffer_dir, encoder_kind=args.encoder, split="train", seed=args.seed,
    )
    heldout_ds = None
    if not args.skip_heldout_eval:
        heldout_ds = OfflineBufferDataset(
            args.buffer_dir, encoder_kind=args.encoder, split="heldout", seed=args.seed,
        )

    enc = _encoder_spec(args.encoder)
    if tuple(train_ds.obs_shape) != enc["obs_shape"]:
        raise ValueError(
            f"buffer obs_shape={train_ds.obs_shape} does not match encoder "
            f"{args.encoder} expected {enc['obs_shape']}"
        )

    agent_config = R2DreamerConfig(
        encoder_type=enc["encoder_type"],
        encoder_module_cls=enc["module_cls"],
        obs_shape=enc["obs_shape"],
        num_actions=4,  # Habitat objectnav: stop/forward/left/right
        seed=args.seed,
        total_steps=args.steps,
        log_every=args.log_every,
        logdir=str(output_dir),
        **SHARED_AGENT_HYPERPARAMS,
    )

    cfg_snapshot = asdict(agent_config) if is_dataclass(agent_config) else dict(vars(agent_config))
    # Drop the unpicklable Flax class object before dumping.
    cfg_snapshot.pop("encoder_module_cls", None)
    cfg_snapshot.update(
        {
            "issue": "3D-26",
            "buffer_dir": str(args.buffer_dir),
            "encoder_choice": args.encoder,
            "train_split_size": train_ds.size,
            "heldout_split_size": heldout_ds.size if heldout_ds is not None else None,
            "buffer_metadata": {
                "code_sha": train_ds.metadata.code_sha,
                "checkpoint_sha256": train_ds.metadata.checkpoint_sha256,
                "collect_seed": train_ds.metadata.collect_seed,
                "status": train_ds.metadata.status,
                "n_completed_steps": train_ds.metadata.n_completed_steps,
                "num_episodes": train_ds.metadata.num_episodes,
                "heldout_start_episode": train_ds.metadata.heldout_start_episode,
            },
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(cfg_snapshot, indent=2))

    init_rng, train_rng = jax.random.split(jax.random.PRNGKey(args.seed))
    agent = R2DreamerAgent(agent_config, init_rng)

    wandb_run = None
    wandb_module = None
    if not args.no_wandb:
        import wandb

        wandb_module = wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=cfg_snapshot,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
        )

    csv_path = output_dir / "metrics.csv"
    rows_buffer: list[tuple[int, str, float]] = []

    print(f"Starting offline training for {args.steps} grad steps ...")
    t0 = time.time()
    last_log_time = t0

    try:
        for step in range(args.steps):
            train_rng, batch_rng, step_rng = jax.random.split(train_rng, 3)
            batch = _convert_batch(
                train_ds.sample(
                    SHARED_AGENT_HYPERPARAMS["batch_size"],
                    SHARED_AGENT_HYPERPARAMS["seq_len"],
                ),
                num_actions=agent_config.num_actions,
            )
            metrics = agent.train_step(batch, step_rng)

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                fps = (step + 1) / max(elapsed, 1e-6)
                step_fps = args.log_every / max(time.time() - last_log_time, 1e-6)
                last_log_time = time.time()
                total = float(metrics.get("total_loss", 0.0))
                dyn = float(metrics.get("loss/dyn", 0.0))
                rep = float(metrics.get("loss/rep", 0.0))
                rew = float(metrics.get("loss/rew", 0.0))
                pol = float(metrics.get("loss/policy", 0.0))
                nan_count = float(metrics.get("nan_skipped", 0.0))
                print(
                    f"[step {step:>8d}/{args.steps}] "
                    f"total={total:.4f} dyn={dyn:.4f} rep={rep:.4f} "
                    f"rew={rew:.4f} policy={pol:.4f} "
                    f"nan_skipped={nan_count:.0f} fps={step_fps:.1f} "
                    f"(avg {fps:.1f})",
                    flush=True,
                )
                rows_buffer.extend((step, k, float(v)) for k, v in metrics.items())
                if wandb_module is not None:
                    wandb_module.log({k: float(v) for k, v in metrics.items()}, step=step)
                _flush_csv(csv_path, rows_buffer)
                rows_buffer.clear()

                if not np.isfinite(total):
                    raise RuntimeError(
                        f"Non-finite total_loss at step {step}: {total}. "
                        "Aborting before checkpoint corruption."
                    )

            if (step + 1) % args.checkpoint_every == 0:
                ckpt_path = _save_checkpoint(agent, step + 1, output_dir)
                print(f"Checkpoint: {ckpt_path}")

            if (
                heldout_ds is not None
                and args.heldout_eval_every > 0
                and (step + 1) % args.heldout_eval_every == 0
            ):
                train_rng, eval_rng = jax.random.split(train_rng)
                heldout_metrics = _aggregate_heldout(
                    agent, heldout_ds, agent_config.num_actions, eval_rng,
                    max_batches=args.heldout_eval_batches,
                )
                if heldout_metrics:
                    rows_buffer.extend(
                        (step, k, v) for k, v in heldout_metrics.items()
                    )
                    if wandb_module is not None:
                        wandb_module.log(heldout_metrics, step=step)
                    print(
                        f"[step {step:>8d}] heldout "
                        f"total={heldout_metrics.get('heldout/total_loss', 0.0):.4f} "
                        f"dyn={heldout_metrics.get('heldout/loss/dyn', 0.0):.4f} "
                        f"rep={heldout_metrics.get('heldout/loss/rep', 0.0):.4f}"
                    )
                    _flush_csv(csv_path, rows_buffer)
                    rows_buffer.clear()
    finally:
        _flush_csv(csv_path, rows_buffer)

    final_ckpt = _save_checkpoint(agent, args.steps, output_dir)
    print(f"Final checkpoint: {final_ckpt}")

    if heldout_ds is not None:
        train_rng, eval_rng = jax.random.split(train_rng)
        final_heldout = _aggregate_heldout(
            agent, heldout_ds, agent_config.num_actions, eval_rng,
            max_batches=max(args.heldout_eval_batches, 64),
        )
        if final_heldout:
            (output_dir / "heldout_final.json").write_text(json.dumps(final_heldout, indent=2))
            _flush_csv(csv_path, [(args.steps, k, v) for k, v in final_heldout.items()])
            if wandb_module is not None:
                wandb_module.log({f"final/{k}": v for k, v in final_heldout.items()}, step=args.steps)
            print("Final held-out metrics:")
            for k, v in final_heldout.items():
                print(f"  {k}={v:.6f}")

    if wandb_module is not None and wandb_run is not None:
        wandb_module.finish()

    print(f"Done. total_time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
