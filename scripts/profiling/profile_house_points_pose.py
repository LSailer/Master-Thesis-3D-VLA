"""Phase-timing profiler for the house-points-pose training loop.

Answers "where do the ~213-230 ms/step of the production-shape compare runs
(jobs 5736907/5736908) go?". Unlike ``profile_training.py`` (which re-implements
the loop for the cnn/vggt encoders), this profiler runs the REAL production
stack: it builds the trainer through ``launch_run`` exactly like ``run.py``
does, and instruments the instance by wrapping its hot-path methods before
``Trainer.run()`` executes. Measured phases per env step:

    agent_act        agent.act — world-model inference + policy (syncs on int)
    env_step         Habitat sim step + render
    obs_transform    full ObsAdapter.transform (sum of the three below + glue)
      vggt_extract   VGGT feature extraction (blocked on output arrays)
      house_add      HouseContextPoseBuffer.add (blocked on store array)
      house_snapshot fixed-shape (max_points, 6) snapshot for the agent
    replay_add       host-side replay-buffer append
    replay_sample    replay batch sampling
    replay_augment   attaching the latest house snapshot to the batch
    train_step       agent.train_step (syncs via float() metric casts)

``train_step`` runs ``train_ratio / (batch_size * seq_len)`` times per env
step, so the report also prints an amortized per-env-step accounting that
should reconcile with the ``perf/ms_per_step_interval`` seen in real runs.

Run on a GPU node (sbatch via scripts/slurm/configs/profile_house_points_pose.yaml).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.shared.profiling import (
    block_until_ready_tree,
    init_phase_times,
    render_phase_table,
    summarize_values_ms,
    write_json,
)

PHASES = (
    "agent_act",
    "env_step",
    "obs_transform",
    "vggt_extract",
    "house_add",
    "house_snapshot",
    "replay_add",
    "replay_sample",
    "replay_augment",
    "train_step",
)

# Phases whose amortized sum should reconcile with perf/ms_per_step_interval.
# obs_transform already contains vggt_extract/house_add/house_snapshot.
PER_STEP_PHASES = ("agent_act", "env_step", "obs_transform", "replay_add")


def _timed_method(orig, phase_times, phase, block=None):
    """Wrap a bound method with a wall-clock timer.

    Args:
      orig: The bound method to wrap.
      phase_times: Millisecond accumulator from ``init_phase_times``.
      phase: Phase name to record under.
      block: Optional callable applied to the result to force device sync
        before the timer stops (JAX dispatch is asynchronous).

    Returns:
      The wrapped callable, signature-compatible with ``orig``.
    """

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig(*args, **kwargs)
        if block is not None:
            block(result)
        phase_times[phase].append((time.perf_counter() - t0) * 1000.0)
        return result

    return wrapped


def _block_vggt_output(out) -> None:
    """Force-sync the VGGT output fields the adapter consumes."""
    from src.r2dreamer.adapters.house_points_adapter import _vggt_output_field

    for field in ("world_points", "confidence", "camera_pose", "extrinsics"):
        value = _vggt_output_field(out, field)
        if value is not None:
            block_until_ready_tree(value)


def _instrument(trainer, phase_times) -> None:
    """Install timing wrappers on a fully constructed Trainer instance.

    Args:
      trainer: The production ``Trainer`` about to enter ``run()``.
      phase_times: Millisecond accumulator shared with the report step.
    """
    from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer

    agent, env, adapter, replay = (
        trainer.agent,
        trainer.env,
        trainer.obs_adapter,
        trainer.buffer,
    )

    agent.act = _timed_method(agent.act, phase_times, "agent_act")
    agent.train_step = _timed_method(agent.train_step, phase_times, "train_step")
    env.step = _timed_method(env.step, phase_times, "env_step")

    adapter.transform = _timed_method(
        adapter.transform, phase_times, "obs_transform"
    )
    extractor = getattr(adapter, "_extractor", None)
    if extractor is not None:
        extractor.extract = _timed_method(
            extractor.extract,
            phase_times,
            "vggt_extract",
            block=_block_vggt_output,
        )
    if hasattr(adapter, "_house_context_snapshot"):
        adapter._house_context_snapshot = _timed_method(
            adapter._house_context_snapshot,
            phase_times,
            "house_snapshot",
            block=block_until_ready_tree,
        )
    if hasattr(adapter, "augment_replay_batch"):
        adapter.augment_replay_batch = _timed_method(
            adapter.augment_replay_batch, phase_times, "replay_augment"
        )

    # House buffers are created lazily per scene — wrap at class level so
    # every instance is covered. ``add`` returns the device store array.
    orig_add = HouseContextPoseBuffer.add

    def timed_add(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = orig_add(self, *args, **kwargs)
        block_until_ready_tree(result)
        phase_times["house_add"].append((time.perf_counter() - t0) * 1000.0)
        return result

    HouseContextPoseBuffer.add = timed_add

    replay.add = _timed_method(replay.add, phase_times, "replay_add")
    replay.sample = _timed_method(
        replay.sample, phase_times, "replay_sample", block=block_until_ready_tree
    )


def _steady_stats(values: list[float], tail_frac: float) -> dict[str, float]:
    """Summarize the steady-state tail of one phase's samples.

    Args:
      values: Millisecond samples in call order (JIT warmup included).
      tail_frac: Fraction of trailing samples treated as steady state.

    Returns:
      ``summarize_values_ms`` stats of the tail, plus ``first_ms`` (the
      warmup-dominated first call) and ``n_total``.
    """
    if not values:
        stats = summarize_values_ms(values)
        stats["first_ms"] = 0.0
        stats["n_total"] = 0
        return stats
    tail = values[max(1, int(len(values) * (1.0 - tail_frac))):] or values[-1:]
    stats = summarize_values_ms(tail)
    stats["first_ms"] = values[0]
    stats["n_total"] = len(values)
    return stats


def _report(phase_times, args) -> dict:
    """Build and print the per-phase and amortized per-step report."""
    stats = {phase: _steady_stats(phase_times[phase], args.tail_frac) for phase in PHASES}

    table = render_phase_table(
        stats,
        PHASES,
        (
            ("mean_ms", "mean_ms", ".3f"),
            ("p50_ms", "p50_ms", ".3f"),
            ("p95_ms", "p95_ms", ".3f"),
            ("first_ms", "first_ms", ".1f"),
            ("n_total", "n_total", ".0f"),
        ),
    )

    train_per_env_step = args.train_ratio / (args.batch_size * args.seq_len)
    amortized = {
        phase: stats[phase]["mean_ms"] for phase in PER_STEP_PHASES
    }
    amortized["train_step (x%.2f)" % train_per_env_step] = (
        stats["train_step"]["mean_ms"] * train_per_env_step
    )
    amortized["replay_sample (x%.2f)" % train_per_env_step] = (
        stats["replay_sample"]["mean_ms"] * train_per_env_step
    )
    amortized["replay_augment (x%.2f)" % train_per_env_step] = (
        stats["replay_augment"]["mean_ms"] * train_per_env_step
    )
    total = sum(amortized.values())

    print()
    print(f"steady-state phase stats (last {args.tail_frac:.0%} of samples):")
    print(table)
    print()
    print("amortized per-env-step accounting (steady-state means):")
    for name, ms in amortized.items():
        print(f"  {name:>28}: {ms:8.2f} ms  ({100.0 * ms / max(total, 1e-9):5.1f}%)")
    print(f"  {'TOTAL':>28}: {total:8.2f} ms/env-step")
    print()
    inner = sum(
        stats[p]["mean_ms"] for p in ("vggt_extract", "house_add", "house_snapshot")
    )
    print(
        f"obs_transform decomposition: total={stats['obs_transform']['mean_ms']:.2f} "
        f"= vggt_extract {stats['vggt_extract']['mean_ms']:.2f} "
        f"+ house_add {stats['house_add']['mean_ms']:.2f} "
        f"+ house_snapshot {stats['house_snapshot']['mean_ms']:.2f} "
        f"+ glue {stats['obs_transform']['mean_ms'] - inner:.2f} ms"
    )
    return {
        "stats": stats,
        "amortized_per_env_step_ms": amortized,
        "amortized_total_ms": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="House-points-pose training-loop phase profiler"
    )
    parser.add_argument(
        "--run_id",
        default="habitat-l1-gnn-house-points-pose",
        help="run.py run id (gnn or vggt house-points-pose variant).",
    )
    parser.add_argument("--prefill_steps", type=int, default=1300)
    parser.add_argument("--acting_steps", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--train_ratio", type=int, default=512)
    parser.add_argument("--render_resolution", type=int, default=518)
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=None,
        help="Override replay capacity (e.g. < prefill to exercise ring wrap).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--tail_frac",
        type=float,
        default=0.5,
        help="Trailing fraction of samples treated as steady state.",
    )
    args = parser.parse_args()

    phase_times = init_phase_times(PHASES)

    from src.r2dreamer.trainer import Trainer

    original_run = Trainer.run

    def run_instrumented(self):
        _instrument(self, phase_times)
        original_run(self)

    Trainer.run = run_instrumented

    from scripts.r2dreamer._run_configs import launch_run

    argv = [
        "--steps", str(args.acting_steps),
        "--prefill", str(args.prefill_steps),
        "--batch_size", str(args.batch_size),
        "--seq_len", str(args.seq_len),
        "--train_ratio", str(args.train_ratio),
        "--render_resolution", str(args.render_resolution),
        "--seed", str(args.seed),
        "--output_dir", args.output_dir,
        "--val_every", "0",
        "--checkpoint_every", "1000000000",
        "--log_every", "100",
    ]
    if args.buffer_capacity is not None:
        argv += ["--buffer_capacity", str(args.buffer_capacity)]
    print(f"Launching {args.run_id} with argv: {argv}")
    trainer = launch_run(args.run_id, argv=argv)

    report = _report(phase_times, args)
    report["config"] = {
        "run_id": args.run_id,
        "prefill_steps": args.prefill_steps,
        "acting_steps": args.acting_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "train_ratio": args.train_ratio,
        "render_resolution": args.render_resolution,
        "seed": args.seed,
        "tail_frac": args.tail_frac,
    }
    adapter = trainer.obs_adapter
    if hasattr(adapter, "diagnostics"):
        report["house_buffer"] = adapter.diagnostics()
    if hasattr(adapter, "growth_history"):
        report["growth_history"] = adapter.growth_history
    report["raw_phase_times_ms"] = phase_times

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = write_json(
        Path(args.output_dir) / f"profile_house_points_pose_{ts}.json", report
    )
    print(f"\nJSON saved to: {json_path}")


if __name__ == "__main__":
    main()
