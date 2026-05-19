#!/usr/bin/env python3
"""Per-phase wall-clock profile of the aggregator-MLP acting+training loop.

Mirrors the real Trainer.run() control flow (Habitat env + VGGTAggregatorMLPEncoder
adapter + R2DreamerAgent + numpy ring buffer) but wraps every phase in
time.perf_counter() so we see where each env-step actually spends its time.

Phases:
  act               agent.act(...)                       single-step JIT inference
  env_step          env.step(action)                     Habitat render + reward
  vggt_extract      adapter._extractor.extract(image)    full VGGT fwd (aggregator + camera + point heads)
    vggt_forward      (internal phase_times)             transformer block forwards
    vggt_wrapper      (internal phase_times)             head wrap + reshape
  adapter_post      cam/mean/max pools + np.asarray      on-device pool + host copy
  buffer_add        np ring-buffer write                 host-side, microseconds
  buffer_sample     numpy index + jnp.array upload       host→device per train step
  train_step        agent.train_step(batch, key)         encoder + WM + actor + critic + opt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp

from src.r2dreamer.adapters.vggt_adapter import _vggt_aggregator_features
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.launch.curricula import CURRICULA
from src.r2dreamer.encoders import VGGTAggregatorMLPEncoder
from src.r2dreamer.launch.habitat_setup import make_habitat_env
from src.r2dreamer.trainer import convert_batch
from src.buffer.replay_buffer import BufferConfig, ReplayBuffer


def pct(xs, q):
    xs_sorted = sorted(xs)
    if not xs_sorted:
        return 0.0
    return xs_sorted[min(len(xs_sorted) - 1, int(q * len(xs_sorted)))]


def stats(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean_ms": mean(xs),
        "p50_ms": pct(xs, 0.50),
        "p95_ms": pct(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "total_s": sum(xs) / 1000.0,
    }


def block_tree(x):
    return jax.tree_util.tree_map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
        x,
    )


def setup(args):
    print(f"JAX devices: {jax.devices()}", flush=True)

    class _Args:
        render_resolution = 518

    enc = VGGTAggregatorMLPEncoder.from_train_args(_Args())
    adapter = enc.make_adapter()
    spec = enc.spec()
    print(f"Encoder spec: obs_shape={spec.obs_shape}, encoder_type={spec.encoder_type}", flush=True)
    print(f"Adapter: buffer_shape={adapter.buffer_shape}, buffer_dtype={adapter.buffer_dtype}", flush=True)
    print(f"Agent overrides: {spec.agent_overrides}", flush=True)

    env = make_habitat_env(
        curriculum_path=str(CURRICULA["L1"]),
        curriculum_mode="train",
        seed=42,
        render_resolution=spec.env_render_resolution,
    )

    cfg = R2DreamerConfig(
        encoder_type=spec.encoder_type,
        encoder_module_cls=spec.module_cls,
        obs_shape=spec.obs_shape,
        num_actions=4,
        **spec.agent_overrides,
    )
    print(
        f"Train cfg: batch_size={cfg.batch_size}, seq_len={cfg.seq_len}, "
        f"train_ratio={cfg.train_ratio}, buffer_capacity={cfg.buffer_capacity}",
        flush=True,
    )

    rng = jax.random.PRNGKey(42)
    agent = R2DreamerAgent(cfg, rng)
    rng, _ = jax.random.split(rng)

    buffer = ReplayBuffer(BufferConfig(
        capacity=cfg.buffer_capacity,
        obs_shape=adapter.buffer_shape,
        obs_dtype=adapter.buffer_dtype,
        normalize_obs=adapter.normalize_on_sample,
    ))
    return enc, adapter, env, cfg, agent, buffer, rng


def transform_timed(adapter, obs_dict):
    """Manual reimplementation of VGGTObsAdapter.transform that exposes timings.

    Mirrors src/r2dreamer/adapters/vggt_adapter.py:63-78 for the
    feature_kind="aggregator" branch. Returns the same (replay_features,
    agent_obs) tuple plus a timings dict.
    """
    pt = {"vggt_forward": [], "vggt_wrapper": []}
    t_e0 = time.perf_counter()
    out = adapter._extractor.extract(obs_dict["image"], phase_times=pt)
    features_jax = _vggt_aggregator_features(out, adapter._aggregator_feature_shape)
    block_tree(features_jax)
    t_extract_ms = (time.perf_counter() - t_e0) * 1000

    t_p0 = time.perf_counter()
    replay_features = np.asarray(features_jax).astype(np.float32)
    agent_features = features_jax.astype(jnp.float32)
    block_tree(agent_features)
    t_post_ms = (time.perf_counter() - t_p0) * 1000

    agent_obs = {"features": agent_features, "is_first": obs_dict.get("is_first", False)}
    timings = {
        "vggt_extract_total": t_extract_ms,
        "adapter_post": t_post_ms,
        "vggt_forward_internal": pt["vggt_forward"][0] if pt["vggt_forward"] else 0.0,
        "vggt_wrapper_internal": pt["vggt_wrapper"][0] if pt["vggt_wrapper"] else 0.0,
    }
    return replay_features, agent_obs, timings


def run_prefill(adapter, env, buffer, num_actions, n_steps):
    print(f"--- Prefill {n_steps} steps (untimed) ---", flush=True)
    t0 = time.time()
    obs = env.reset()
    if adapter.on_episode_reset:
        adapter.on_episode_reset()
    buffer_obs, _, _ = transform_timed(adapter, obs)
    for i in range(n_steps):
        action = int(np.random.randint(0, num_actions))
        next_obs = env.step(action)
        next_buffer_obs, _, _ = transform_timed(adapter, next_obs)
        success = next_obs.get("success", 0.0) > 0
        buffer.add(buffer_obs, action, next_obs["reward"], next_obs["done"], terminal=success)
        if next_obs["done"]:
            obs = env.reset()
            if adapter.on_episode_reset:
                adapter.on_episode_reset()
            buffer_obs, _, _ = transform_timed(adapter, obs)
        else:
            buffer_obs = next_buffer_obs
        if (i + 1) % 50 == 0:
            print(f"  prefill {i + 1}/{n_steps} elapsed={time.time() - t0:.1f}s", flush=True)
    print(f"  prefill done in {time.time() - t0:.1f}s", flush=True)


def run_measured(label, adapter, env, agent, cfg, buffer, rng, n_steps, batch_steps, accum):
    print(f"--- {label} {n_steps} steps ---", flush=True)
    obs = env.reset()
    if adapter.on_episode_reset:
        adapter.on_episode_reset()
    buffer_obs, agent_obs, _ = transform_timed(adapter, obs)
    train_credit = 0.0
    loop_t0 = time.perf_counter()
    for i in range(n_steps):
        step_t0 = time.perf_counter()

        # act
        rng, act_key = jax.random.split(rng)
        t0 = time.perf_counter()
        action = agent.act(agent_obs, act_key)
        block_tree(action)
        accum["act"].append((time.perf_counter() - t0) * 1000)

        # env step
        t0 = time.perf_counter()
        next_obs = env.step(int(action))
        accum["env_step"].append((time.perf_counter() - t0) * 1000)

        # adapter (vggt extract + post)
        next_buffer_obs, next_agent_obs, tx = transform_timed(adapter, next_obs)
        accum["vggt_extract_total"].append(tx["vggt_extract_total"])
        accum["adapter_post"].append(tx["adapter_post"])
        accum["vggt_forward_internal"].append(tx["vggt_forward_internal"])
        accum["vggt_wrapper_internal"].append(tx["vggt_wrapper_internal"])

        # buffer.add
        success = next_obs.get("success", 0.0) > 0
        t0 = time.perf_counter()
        buffer.add(buffer_obs, int(action), next_obs["reward"], next_obs["done"], terminal=success)
        accum["buffer_add"].append((time.perf_counter() - t0) * 1000)

        if next_obs["done"]:
            obs = env.reset()
            if adapter.on_episode_reset:
                adapter.on_episode_reset()
            buffer_obs, agent_obs, _ = transform_timed(adapter, obs)
        else:
            buffer_obs = next_buffer_obs
            agent_obs = next_agent_obs

        # train credit -> one or more train steps
        if buffer.size >= batch_steps:
            train_credit += cfg.train_ratio / batch_steps
            while train_credit >= 1.0:
                t0 = time.perf_counter()
                batch = buffer.sample(cfg.batch_size, cfg.seq_len)
                block_tree(batch)
                accum["buffer_sample"].append((time.perf_counter() - t0) * 1000)

                batch = convert_batch(batch, cfg.num_actions)
                rng, tkey = jax.random.split(rng)
                t0 = time.perf_counter()
                metrics = agent.train_step(batch, tkey)
                block_tree(metrics)
                accum["train_step"].append((time.perf_counter() - t0) * 1000)
                train_credit -= 1.0

        accum["total_step"].append((time.perf_counter() - step_t0) * 1000)

        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - loop_t0
            fps = (i + 1) / elapsed
            print(f"  {label} {i + 1}/{n_steps} elapsed={elapsed:.1f}s fps={fps:.2f}", flush=True)
    return rng


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefill", type=int, default=200,
                   help="Buffer prefill steps (untimed)")
    p.add_argument("--warmup", type=int, default=20,
                   help="Timed warmup steps (reported separately, captures JIT compile)")
    p.add_argument("--measure", type=int, default=80,
                   help="Timed steady-state steps")
    p.add_argument("--out", type=str,
                   default="output/profiling/pipeline_aggregator_mlp.json")
    args = p.parse_args()

    overall_t0 = time.time()
    enc, adapter, env, cfg, agent, buffer, rng = setup(args)
    batch_steps = cfg.batch_size * cfg.seq_len

    run_prefill(adapter, env, buffer, cfg.num_actions, args.prefill)

    warmup_accum = {k: [] for k in (
        "act", "env_step", "vggt_extract_total", "vggt_forward_internal",
        "vggt_wrapper_internal", "adapter_post", "buffer_add",
        "buffer_sample", "train_step", "total_step",
    )}
    steady_accum = {k: [] for k in warmup_accum}

    rng = run_measured("WARMUP", adapter, env, agent, cfg, buffer, rng,
                       args.warmup, batch_steps, warmup_accum)
    rng = run_measured("MEASURE", adapter, env, agent, cfg, buffer, rng,
                       args.measure, batch_steps, steady_accum)

    # --- Summary ---
    print("\n=========================================================", flush=True)
    print(" Steady-state per-phase breakdown (after warmup)", flush=True)
    print("=========================================================", flush=True)
    print(f"{'Phase':<26} {'n':>4} {'mean_ms':>9} {'p50_ms':>8} {'p95_ms':>8} {'total_s':>9}", flush=True)
    print("-" * 70, flush=True)
    for k in ["act", "env_step", "vggt_extract_total",
              "vggt_forward_internal", "vggt_wrapper_internal",
              "adapter_post", "buffer_add", "buffer_sample",
              "train_step", "total_step"]:
        s = stats(steady_accum[k])
        if s["n"] == 0:
            continue
        print(
            f"{k:<26} {s['n']:>4d} {s['mean_ms']:>9.3f} {s['p50_ms']:>8.3f} "
            f"{s['p95_ms']:>8.3f} {s['total_s']:>9.3f}",
            flush=True,
        )

    total_steady = stats(steady_accum["total_step"])
    if total_steady["n"] > 0:
        fps = 1000.0 / total_steady["mean_ms"]
        print(f"\nSteady-state mean fps: {fps:.2f} (1000/{total_steady['mean_ms']:.1f} ms)", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "devices": [str(d) for d in jax.devices()],
        "config": {
            "prefill": args.prefill,
            "warmup": args.warmup,
            "measure": args.measure,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "train_ratio": cfg.train_ratio,
            "obs_shape": list(cfg.obs_shape),
            "batch_steps": batch_steps,
        },
        "warmup": {k: stats(v) for k, v in warmup_accum.items()},
        "steady": {k: stats(v) for k, v in steady_accum.items()},
        "total_wallclock_s": time.time() - overall_t0,
    }
    out_path.write_text(json.dumps(blob, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
