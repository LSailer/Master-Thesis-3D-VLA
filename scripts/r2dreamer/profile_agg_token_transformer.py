#!/usr/bin/env python3
"""Smoke/profile the 3D-75 VGGT aggregator-token Transformer path.

Reports the long-budget feasibility fields required by scripts/r2dreamer/AGENTS.md:
env timing, VGGT feature extraction, replay overhead, train-step timing, storage
preflight, and a 2,000,000-step wall-clock estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.replay_buffer import ReplayBuffer
from src.r2dreamer.observation_preparation.vggt_readouts import (
    _flatten_full_aggregator_tokens,
)
from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.encoders import VGGTAggTokenTransformerEncoder
from src.r2dreamer.launch.habitat_setup import make_habitat_env
from src.r2dreamer.obs_batch import ObservationPacker
from src.r2dreamer.trainer import convert_batch
from src.r2dreamer.agent import R2DreamerAgent
from src.shared.profiling import timed


TARGET_STEPS = 2_000_000
TOKEN_COUNT = 1374
TOKEN_DIM = 1024
BYTES_PER_VALUE = 2
REPLAY_BYTES_PER_STEP = TOKEN_COUNT * TOKEN_DIM * BYTES_PER_VALUE


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    return xs_sorted[min(len(xs_sorted) - 1, int(q * len(xs_sorted)))]


def stats(xs: list[float]) -> dict[str, float | int]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean_ms": mean(xs),
        "p50_ms": pct(xs, 0.50),
        "p95_ms": pct(xs, 0.95),
        "total_s": sum(xs) / 1000.0,
    }


def block_tree(x):
    return jax.tree_util.tree_map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
        x,
    )


def gib(n_bytes: int | float) -> float:
    return float(n_bytes) / (1024 ** 3)


def storage_preflight(path: Path, capacity: int, min_free_gb: float) -> dict[str, float | bool | str]:
    usage = shutil.disk_usage(path)
    capacity_bytes = capacity * REPLAY_BYTES_PER_STEP
    slots100k_bytes = 100_000 * REPLAY_BYTES_PER_STEP
    return {
        "path": str(path),
        "free_gib": gib(usage.free),
        "required_for_capacity_gib": gib(capacity_bytes),
        "raw_token_100k_gib": gib(slots100k_bytes),
        "min_free_gib": min_free_gb,
        "passes_min_free_gate": gib(usage.free) >= min_free_gb,
    }


def setup(args):
    print(f"JAX devices: {jax.devices()}", flush=True)

    class _Args:
        render_resolution = 518

    enc = VGGTAggTokenTransformerEncoder.from_train_args(_Args())
    adapter = enc.make_adapter()
    spec = enc.spec()
    print(f"Encoder spec: obs_shape={spec.obs_shape}, encoder_type={spec.encoder_type}", flush=True)
    print(f"Adapter: buffer_shape={adapter.buffer_shape}, buffer_dtype={adapter.buffer_dtype}", flush=True)
    print(f"Agent overrides: {spec.agent_overrides}", flush=True)

    env = make_habitat_env(
        curriculum="L1",
        curriculum_mode="train",
        seed=args.seed,
        render_resolution=spec.env_render_resolution,
    )
    cfg = R2DreamerConfig(
        encoder_type=spec.encoder_type,
        encoder_module_cls=spec.module_cls,
        obs_shape=spec.obs_shape,
        num_actions=4,
        **spec.agent_overrides,
    )
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.train_ratio is not None:
        cfg.train_ratio = args.train_ratio
    rng = jax.random.PRNGKey(args.seed)
    agent = R2DreamerAgent(cfg, rng)
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity)
    return enc, adapter, env, cfg, agent, buffer, rng


def transform_timed(adapter, obs_dict, accum):
    pt = {"vggt_forward": [], "vggt_wrapper": []}
    t0 = time.perf_counter()
    out = adapter._extractor.extract(obs_dict["image"], phase_times=pt)
    features_jax = _flatten_full_aggregator_tokens(out["aggregator_features"])
    block_tree(features_jax)
    accum["vggt_extract_total"].append((time.perf_counter() - t0) * 1000.0)
    if pt["vggt_forward"]:
        accum["vggt_forward_internal"].append(pt["vggt_forward"][0])
    if pt["vggt_wrapper"]:
        accum["vggt_wrapper_internal"].append(pt["vggt_wrapper"][0])

    t1 = time.perf_counter()
    replay_features = np.asarray(features_jax).astype(np.float16)
    agent_features = features_jax.astype(jnp.float32)
    block_tree(agent_features)
    accum["adapter_post"].append((time.perf_counter() - t1) * 1000.0)
    agent_obs = {"features": agent_features, "is_first": obs_dict.get("is_first", False)}
    return replay_features, agent_obs


PHASES = (
    "act", "env_step", "vggt_extract_total", "vggt_forward_internal",
    "vggt_wrapper_internal", "adapter_post", "buffer_add", "buffer_sample",
    "train_step", "total_step",
)


def init_accum() -> dict[str, list[float]]:
    return {k: [] for k in PHASES}


def run_phase(label, n_steps, adapter, env, cfg, agent, buffer, rng, buffer_obs, agent_obs):
    accum = init_accum()
    packer = ObservationPacker(cfg)
    batch_steps = cfg.batch_size * cfg.seq_len
    train_credit = 0.0
    for i in range(n_steps):
        step_t0 = time.perf_counter()
        rng, act_key = jax.random.split(rng)
        with timed(accum, "act"):
            encoder_obs = packer.from_step(agent_obs)
            action = agent.act(encoder_obs, agent_obs["is_first"], act_key)

        with timed(accum, "env_step"):
            next_obs = env.step(int(action))
        next_buffer_obs, next_agent_obs = transform_timed(adapter, next_obs, accum)

        with timed(accum, "buffer_add"):
            buffer.add(buffer_obs, next_obs)

        if next_obs["done"]:
            obs = env.reset()
            if adapter.on_episode_reset:
                adapter.on_episode_reset()
            buffer_obs, agent_obs = transform_timed(adapter, obs, accum)
        else:
            buffer_obs, agent_obs = next_buffer_obs, next_agent_obs

        if buffer.size >= batch_steps:
            train_credit += cfg.train_ratio / batch_steps
            while train_credit >= 1.0:
                with timed(accum, "buffer_sample"):
                    batch = buffer.sample(cfg.batch_size, cfg.seq_len)
                    block_tree(batch)
                batch = convert_batch(batch, cfg.num_actions)
                rng, train_key = jax.random.split(rng)
                with timed(accum, "train_step"):
                    metrics = agent.train_step(batch, train_key)
                    block_tree(metrics)
                train_credit -= 1.0

        accum["total_step"].append((time.perf_counter() - step_t0) * 1000.0)
        print(f"  {label} {i + 1}/{n_steps}", flush=True)
    return accum, rng, buffer_obs, agent_obs


def run_profile(args, adapter, env, cfg, agent, buffer, rng):
    prefill_accum = init_accum()
    warmup_accum = init_accum()
    steady_accum = init_accum()

    obs = env.reset()
    if adapter.on_episode_reset:
        adapter.on_episode_reset()
    buffer_obs, agent_obs = transform_timed(adapter, obs, prefill_accum)

    for i in range(args.prefill):
        action = int(np.random.randint(0, cfg.num_actions))
        with timed(prefill_accum, "env_step"):
            next_obs = env.step(action)
        next_buffer_obs, next_agent_obs = transform_timed(adapter, next_obs, prefill_accum)

        with timed(prefill_accum, "buffer_add"):
            buffer.add(buffer_obs, next_obs)

        if next_obs["done"]:
            obs = env.reset()
            if adapter.on_episode_reset:
                adapter.on_episode_reset()
            buffer_obs, agent_obs = transform_timed(adapter, obs, prefill_accum)
        else:
            buffer_obs, agent_obs = next_buffer_obs, next_agent_obs
        print(f"  prefill {i + 1}/{args.prefill}", flush=True)

    if args.warmup > 0:
        warmup_accum, rng, buffer_obs, agent_obs = run_phase(
            "warmup", args.warmup, adapter, env, cfg, agent, buffer, rng, buffer_obs, agent_obs,
        )
    if args.measure > 0:
        steady_accum, rng, buffer_obs, agent_obs = run_phase(
            "measure", args.measure, adapter, env, cfg, agent, buffer, rng, buffer_obs, agent_obs,
        )
    return {"prefill": prefill_accum, "warmup": warmup_accum, "steady": steady_accum}


def summarize(accum, cfg, storage, target_steps: int, max_hours: float):
    total = stats(accum["total_step"])
    mean_step_ms = float(total.get("mean_ms", 0.0))
    env_sps = 1000.0 / mean_step_ms if mean_step_ms > 0 else 0.0
    estimated_hours = (target_steps / env_sps) / 3600.0 if env_sps > 0 else float("inf")
    train_step = stats(accum["train_step"])
    train_sps = 1000.0 / float(train_step.get("mean_ms", 0.0)) if train_step.get("n", 0) else 0.0
    feasible = (
        bool(storage["passes_min_free_gate"])
        and env_sps > 0
        and estimated_hours <= max_hours
        and train_step.get("n", 0) > 0
    )
    limiting = []
    if not storage["passes_min_free_gate"]:
        limiting.append("storage free-space gate")
    if train_step.get("n", 0) == 0:
        limiting.append("no train_step sample collected")
    if estimated_hours > max_hours:
        limiting.append(f"wall-clock estimate exceeds {max_hours:.1f}h")
    if not limiting:
        limiting.append("none under this smoke estimate")
    return {
        "env_steps_per_sec": env_sps,
        "mean_env_step_wall_ms": mean_step_ms,
        "train_steps_per_sec": train_sps,
        "target_steps": target_steps,
        "estimated_hours": estimated_hours,
        "max_feasible_hours": max_hours,
        "feasible": feasible,
        "limiting_factors": limiting,
        "config": {
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "train_ratio": cfg.train_ratio,
            "buffer_capacity": cfg.buffer_capacity,
            "obs_shape": list(cfg.obs_shape),
            "vggt_token_projection_dim": cfg.vggt_token_projection_dim,
            "vggt_token_transformer_layers": cfg.vggt_token_transformer_layers,
            "vggt_token_transformer_heads": cfg.vggt_token_transformer_heads,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefill", type=int, default=16)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measure", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-steps", type=int, default=TARGET_STEPS)
    p.add_argument("--max-feasible-hours", type=float, default=42.0)
    p.add_argument("--min-free-gb", type=float, default=300.0)
    p.add_argument("--storage-path", type=Path, default=Path("output"))
    p.add_argument("--out", type=Path, default=Path("output/profiling/agg_token_transformer.json"))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--train-ratio", type=int, default=None)
    args = p.parse_args()

    args.storage_path.mkdir(parents=True, exist_ok=True)
    enc, adapter, env, cfg, agent, buffer, rng = setup(args)
    storage = storage_preflight(args.storage_path, cfg.buffer_capacity, args.min_free_gb)
    print("Storage preflight:", json.dumps(storage, indent=2), flush=True)

    profile = run_profile(args, adapter, env, cfg, agent, buffer, rng)
    accum = profile["steady"]
    summary = summarize(accum, cfg, storage, args.target_steps, args.max_feasible_hours)

    print("\nSteady-state phase summary", flush=True)
    for name in accum:
        s = stats(accum[name])
        if s["n"]:
            print(f"{name:24s} n={s['n']:>3} mean_ms={s['mean_ms']:.3f} p95_ms={s['p95_ms']:.3f}", flush=True)
    print("\nFeasibility:", json.dumps(summary, indent=2), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "storage": storage,
        "phases": {
            phase: {k: stats(v) for k, v in phase_accum.items()}
            for phase, phase_accum in profile.items()
        },
        "summary": summary,
        "devices": [str(d) for d in jax.devices()],
    }, indent=2))
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
