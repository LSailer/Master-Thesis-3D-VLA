#!/usr/bin/env python3
"""Benchmark env-step/train-step scheduling modes on the real R2Dreamer stack.

Builds env + agent + collector exactly like the production composition root
(``src.r2dreamer.launch.train.train``), selected via a canonical
``scripts/r2dreamer/_run_configs.py`` run id, then runs ONE scheduling mode
(``--mode interleaved|episode|threaded``, see ``scheduling_loops.py``) and
appends its timing stats to ``<out_dir>/MANIFEST.json``.

Judge runs by the manifest ``status`` field, never the exit code — habitat's
GL teardown SIGABRTs on this cluster, so on success the runner hard-exits
after the manifest is flushed.

Usage (one mode per process; see run_scheduling_experiment.sbatch)::

    .venv/bin/python prototyp/train_scheduling/run_scheduling_experiment.py \
        --mode threaded --run_id habitat-l1-cnn \
        --out_dir outputs/prototype/train_scheduling/run-123 \
        --steps 600 --prefill 200 --batch_size 4 --seq_len 16 --train_ratio 16

    # afterwards, combine + compute the overlap gain (CPU-safe, no habitat):
    .venv/bin/python prototyp/train_scheduling/run_scheduling_experiment.py \
        --summarize outputs/prototype/train_scheduling/run-123

Unrecognized flags are forwarded verbatim to the production train parser
(``--seed``, ``--log_every``, encoder knobs, ...). This runner's ``--mode``
shadows the train parser's env-split flag, which therefore stays ``train``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prototyp.train_scheduling.locked_buffer import LockedReplayBuffer
from prototyp.train_scheduling.scheduling_loops import LOOPS, LoopStats

MODES = ("interleaved", "episode", "threaded")
DEFAULT_OUT_ROOT = _REPO_ROOT / "outputs" / "prototype" / "train_scheduling"


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parses runner-owned flags; everything else is forwarded to train's parser.

    Args:
        argv: Raw command-line arguments (without the program name).

    Returns:
        The parsed runner namespace and the leftover train-parser argv.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default=None)
    parser.add_argument(
        "--run_id",
        default="habitat-l1-cnn",
        help="Run id from scripts/r2dreamer/_run_configs.py (env/encoder/"
        "curriculum source of truth). Default is the lightest adapter.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Shared results dir for all modes of one comparison "
        "(default: outputs/prototype/train_scheduling/run-<jobid|timestamp>).",
    )
    parser.add_argument(
        "--summarize",
        metavar="DIR",
        default=None,
        help="Skip running; combine results_<mode>.json in DIR into the "
        "manifest and compute the threaded-vs-interleaved overlap gain.",
    )
    parser.add_argument(
        "--no_hard_exit",
        action="store_true",
        help="Run env teardown instead of os._exit(0) after a successful "
        "habitat run (teardown may SIGABRT on this cluster).",
    )
    return parser.parse_known_args(argv)


def _default_out_dir() -> Path:
    """Returns the default shared results dir for this job/invocation."""
    tag = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUT_ROOT / f"run-{tag}"


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    """Loads the accumulating manifest, or a fresh skeleton if absent."""
    path = out_dir / "MANIFEST.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"experiment": "train_scheduling", "status": "failed", "modes": {}}


def _write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    """Recomputes overall status from per-mode statuses and writes atomically.

    Args:
        out_dir: Shared results dir holding ``MANIFEST.json``.
        manifest: Manifest dict; ``status`` is derived from ``modes``.
    """
    modes = manifest.get("modes", {})
    ok = bool(modes) and all(m.get("status") == "ok" for m in modes.values())
    manifest["status"] = "ok" if ok else "failed"
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = out_dir / "MANIFEST.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "MANIFEST.json")


def _build_experiment(run_id: str, train_argv: list[str], out_dir: Path):
    """Composes env/agent/collector via the production composition helpers.

    Imports are local so ``--summarize`` (and the CPU tests) never touch
    habitat or JAX-GPU code paths.

    Args:
        run_id: Key into ``scripts/r2dreamer/_run_configs.RUN_CONFIGS``.
        train_argv: Flags for the production train parser.
        out_dir: Output dir stamped into the agent config.

    Returns:
        Tuple ``(agent, collector, agent_config, train_args)``.

    Raises:
        KeyError: If ``run_id`` is unknown.
    """
    scripts_dir = _REPO_ROOT / "scripts" / "r2dreamer"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import _run_configs  # noqa: PLC0415 — path-injected canonical registry

    import jax  # noqa: PLC0415

    from src.configs.config import LATENT_PRESETS, R2DreamerConfig  # noqa: PLC0415
    from src.environments.habitat_metrics import HabitatEpisodeMetrics  # noqa: PLC0415
    from src.r2dreamer.agent import R2DreamerAgent  # noqa: PLC0415
    from src.r2dreamer.launch import train as train_mod  # noqa: PLC0415
    from src.r2dreamer.launch.parser import _build_parser_train  # noqa: PLC0415
    from src.r2dreamer.launch.registries import (  # noqa: PLC0415
        encoder_registry,
        env_registry,
    )

    if run_id not in _run_configs.RUN_CONFIGS:
        raise KeyError(
            f"unknown run id {run_id!r}; "
            f"available: {sorted(_run_configs.RUN_CONFIGS)}"
        )
    run_cfg = _run_configs.RUN_CONFIGS[run_id]

    args = _build_parser_train().parse_args(train_argv)
    # The prototype benchmarks the train collector only — never build the val
    # env (env construction is the expensive part of habitat startup).
    args.val_every = 0

    curriculum, curriculum_path = train_mod._effective_curriculum_inputs(
        env=run_cfg["env"],
        args=args,
        curriculum=run_cfg.get("curriculum"),
        env_registry=env_registry,
    )
    enc, adapter, encoder_spec = train_mod._make_encoder_bundle(
        run_cfg["encoder"], args, encoder_registry
    )
    env_instance, _val_env, num_actions = train_mod._make_env_instances(
        env=run_cfg["env"],
        args=args,
        curriculum=curriculum,
        curriculum_path=curriculum_path,
        encoder_spec=encoder_spec,
        env_registry=env_registry,
    )
    agent_overrides = train_mod._agent_overrides_from_args(
        args, encoder_spec, LATENT_PRESETS
    )
    agent_config = train_mod._make_agent_config(
        args=args,
        encoder_spec=encoder_spec,
        num_actions=num_actions,
        output_dir=str(out_dir),
        agent_overrides=agent_overrides,
        config_cls=R2DreamerConfig,
    )
    _, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    agent = R2DreamerAgent(agent_config, init_key)
    collector, _val_collector = train_mod._make_collectors(
        env=run_cfg["env"],
        enc=enc,
        env_instance=env_instance,
        val_env_instance=None,
        agent_config=agent_config,
        adapter=adapter,
        episode_metrics_cls=HabitatEpisodeMetrics,
    )
    return agent, collector, agent_config, args


def _run_mode(
    mode: str, run_id: str, train_argv: list[str], out_dir: Path
) -> LoopStats:
    """Builds the stack, prefills, and times one scheduling mode.

    Args:
        mode: One of ``MODES``.
        run_id: Canonical run id selecting env/encoder/curriculum.
        train_argv: Forwarded train-parser flags (steps, prefill, shapes...).
        out_dir: Results dir (also stamped into the agent config).

    Returns:
        The mode's timing stats.
    """
    import jax  # noqa: PLC0415

    from src.r2dreamer.launch.loops import prefill  # noqa: PLC0415

    agent, collector, acfg, args = _build_experiment(run_id, train_argv, out_dir)
    if mode == "threaded":
        # Wrap-not-edit: the real ReplayBuffer is not thread-safe; serialize
        # add/sample behind one lock (see locked_buffer.py / PROBLEMS.md #2).
        collector.buffer = LockedReplayBuffer(collector.buffer)

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key = prefill(
        collector,
        num_steps=args.prefill,
        num_actions=acfg.num_actions,
        rng_key=rng_key,
    )
    print(
        f"[{mode}] timed section: steps={args.steps} "
        f"B={acfg.batch_size} T={acfg.seq_len} ratio={acfg.train_ratio}",
        flush=True,
    )
    stats = LOOPS[mode](
        agent,
        collector,
        total_steps=args.steps,
        batch_size=acfg.batch_size,
        seq_len=acfg.seq_len,
        train_ratio=acfg.train_ratio,
        rng_key=rng_key,
        log_every=args.log_every,
    )
    return stats


def _summarize(out_dir: Path) -> int:
    """Combines per-mode results and computes the overlap gain.

    Args:
        out_dir: Dir containing ``results_<mode>.json`` files.

    Returns:
        Process exit code (0 when every expected mode reported ok).
    """
    manifest = _load_manifest(out_dir)
    for mode in MODES:
        path = out_dir / f"results_{mode}.json"
        if path.exists():
            manifest["modes"][mode] = json.loads(path.read_text(encoding="utf-8"))

    modes = manifest["modes"]
    inter = modes.get("interleaved", {}).get("stats")
    threaded = modes.get("threaded", {}).get("stats")
    if inter and threaded and threaded["wall_time_s"] > 0:
        manifest["comparison"] = {
            "overlap_gain_vs_interleaved": (
                inter["wall_time_s"] / threaded["wall_time_s"]
            )
        }

    print(f"\n=== train_scheduling summary ({out_dir}) ===")
    header = (
        f"{'mode':<12} {'status':<7} {'wall_s':>8} {'env/s':>8} "
        f"{'train/s':>8} {'train_steps':>11} {'ratio':>7}"
    )
    print(header)
    for mode in MODES:
        entry = modes.get(mode)
        if entry is None:
            print(f"{mode:<12} {'missing':<7}")
            continue
        stats = entry.get("stats") or {}
        print(
            f"{mode:<12} {entry.get('status', '?'):<7} "
            f"{stats.get('wall_time_s', float('nan')):>8.1f} "
            f"{stats.get('env_steps_per_s', float('nan')):>8.1f} "
            f"{stats.get('train_steps_per_s', float('nan')):>8.1f} "
            f"{stats.get('train_steps', 0):>11d} "
            f"{stats.get('achieved_ratio', float('nan')):>7.1f}"
        )
    if "comparison" in manifest:
        gain = manifest["comparison"]["overlap_gain_vs_interleaved"]
        print(f"\noverlap gain (interleaved wall / threaded wall): {gain:.2f}x")

    _write_manifest(out_dir, manifest)
    print(f"MANIFEST status: {manifest['status']}")
    return 0 if manifest["status"] == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point: run one scheduling mode or summarize a finished dir.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (informational only for habitat runs — judge by the
        manifest).
    """
    args, train_argv = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.summarize is not None:
        return _summarize(Path(args.summarize))
    if args.mode is None:
        print("error: --mode is required unless --summarize is given")
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "status": "failed",
        "run_id": args.run_id,
        "train_argv": train_argv,
    }
    try:
        stats = _run_mode(args.mode, args.run_id, train_argv, out_dir)
        entry["status"] = "ok"
        entry["stats"] = stats.to_dict()
        print(
            f"[{args.mode}] done: wall={stats.wall_time_s:.1f}s "
            f"env/s={stats.env_steps_per_s:.1f} "
            f"train/s={stats.train_steps_per_s:.1f} "
            f"achieved_ratio={stats.achieved_ratio:.1f}",
            flush=True,
        )
    except Exception:  # noqa: BLE001 — recorded in the manifest
        entry["error"] = traceback.format_exc()
        traceback.print_exc()

    (out_dir / f"results_{args.mode}.json").write_text(
        json.dumps(entry, indent=2), encoding="utf-8"
    )
    manifest = _load_manifest(out_dir)
    manifest["modes"][args.mode] = entry
    _write_manifest(out_dir, manifest)
    print(f"[{args.mode}] manifest entry status: {entry['status']}", flush=True)

    if entry["status"] == "ok" and not args.no_hard_exit:
        # Habitat GL teardown can SIGABRT after a fully successful run and
        # poison the exit code; everything is flushed, so skip teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0 if entry["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
