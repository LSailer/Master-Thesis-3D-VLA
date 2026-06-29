"""Profile flat vs modality-aware replay layouts.

This isolates the 3D-72 replay change without running Habitat or online VGGT:

* ``flat_hybrid`` emulates the old replay layout:
  one float32 vector ``[rgb64_normalized_flat | wp_cp]``.
* ``modal_hybrid`` uses the new replay layout:
  ``{"image": uint8 RGB64, "wp_cp": float32}``.

Both feed the same HybridEncoder after the agent packs observations at the JAX
boundary, so timing differences are attributable to replay layout and packing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _summ(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p90_ms": _pct(values, 90),
        "p95_ms": _pct(values, 95),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def _block_tree(tree: Any) -> None:
    import jax

    for value in jax.tree_util.tree_leaves(tree):
        try:
            value.block_until_ready()
        except AttributeError:
            pass


def _frame(action: int = 0, reward: float = 0.0, done: bool = False):
    from src.environments.observation import ObservationFrame

    return ObservationFrame(
        image=np.empty((0,), dtype=np.uint8),
        is_first=False,
        previous_action=action,
        reward=reward,
        done=done,
    )


def _make_buffer(kind: str, capacity: int, rng: np.random.Generator):
    from src.buffer.replay_buffer import ReplayBuffer, ReplayTransition

    buffer = ReplayBuffer(capacity=capacity, num_actions=4)
    actions = rng.integers(0, 4, size=capacity, dtype=np.int32)
    rewards = rng.standard_normal(capacity).astype(np.float32)

    def add_transition(index: int, obs: Any) -> None:
        buffer.add(
            ReplayTransition.from_frame(
                obs,
                _frame(action=int(actions[index]), reward=float(rewards[index])),
            )
        )

    if kind == "wpcp":
        features = rng.standard_normal((capacity, 4116), dtype=np.float32)
        for idx, obs in enumerate(features):
            add_transition(idx, obs)
    elif kind == "flat_hybrid":
        image = rng.integers(0, 256, size=(capacity, 3, 64, 64), dtype=np.uint8)
        wp_cp = rng.standard_normal((capacity, 4116), dtype=np.float32)
        rgb = image.astype(np.float32).reshape(capacity, -1) / 255.0
        flat = np.concatenate([rgb, wp_cp], axis=-1).astype(np.float32)
        for idx, obs in enumerate(flat):
            add_transition(idx, obs)
    elif kind == "modal_hybrid":
        image = rng.integers(0, 256, size=(capacity, 3, 64, 64), dtype=np.uint8)
        wp_cp = rng.standard_normal((capacity, 4116), dtype=np.float32)
        for idx in range(capacity):
            add_transition(idx, {"image": image[idx], "wp_cp": wp_cp[idx]})
    else:
        raise ValueError(kind)

    return buffer


def _make_agent(kind: str, seed: int):
    import jax

    from src.r2dreamer.agent import R2DreamerAgent
    from src.configs.config import R2DreamerConfig
    from src.r2dreamer.encoders.mlp import HybridEncoder, MLPEncoder

    common = dict(
        num_actions=4,
        batch_size=16,
        seq_len=64,
        train_ratio=512,
        buffer_capacity=100_000,
        seed=seed,
        decoder=False,
    )
    if kind == "wpcp_mlp3":
        cfg = R2DreamerConfig(
            encoder_type="vggt",
            encoder_module_cls=MLPEncoder,
            obs_shape=(4116,),
            vggt_mlp_layers=3,
            **common,
        )
    elif kind == "hybrid":
        cfg = R2DreamerConfig(
            encoder_type="hybrid",
            encoder_module_cls=HybridEncoder,
            obs_shape=(16404,),
            **common,
        )
    else:
        raise ValueError(kind)
    return cfg, R2DreamerAgent(cfg, jax.random.PRNGKey(seed))


def _measure_sample_convert(buffer, convert_batch, iters: int, warmup: int) -> dict[str, float]:
    times: list[float] = []
    for i in range(iters + warmup):
        t0 = time.perf_counter()
        batch = convert_batch(buffer.sample(16, 64), 4)
        _block_tree(batch)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    return _summ(times)


def _measure_train_step(agent, batch, iters: int, warmup: int) -> dict[str, float]:
    import jax

    _block_tree(batch)
    times: list[float] = []
    for i in range(iters + warmup):
        key = jax.random.PRNGKey(1000 + i)
        t0 = time.perf_counter()
        metrics = agent.train_step(batch, key)
        _ = metrics.get("total_loss", 0.0)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    return _summ(times)


def _measure_combined(buffer, agent, convert_batch, iters: int, warmup: int) -> dict[str, float]:
    import jax

    times: list[float] = []
    for i in range(iters + warmup):
        key = jax.random.PRNGKey(2000 + i)
        t0 = time.perf_counter()
        batch = convert_batch(buffer.sample(16, 64), 4)
        metrics = agent.train_step(batch, key)
        _ = metrics.get("total_loss", 0.0)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    return _summ(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--sample-iters", type=int, default=40)
    parser.add_argument("--train-iters", type=int, default=15)
    parser.add_argument("--combined-iters", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=str, default="output/profiles/modal_replay_profile.json")
    args = parser.parse_args()

    cwd = Path.cwd()
    sys.path.insert(0, str(cwd))

    import jax
    from src.r2dreamer.trainer import convert_batch

    rng = np.random.default_rng(72)
    buffers = {
        kind: _make_buffer(kind, args.capacity, rng)
        for kind in ("wpcp", "flat_hybrid", "modal_hybrid")
    }
    agents = {
        "wpcp_mlp3": _make_agent("wpcp_mlp3", seed=72),
        "flat_hybrid": _make_agent("hybrid", seed=73),
        "modal_hybrid": _make_agent("hybrid", seed=74),
    }

    results: dict[str, Any] = {
        "meta": {
            "cwd": str(cwd),
            "jax_devices": [str(d) for d in jax.devices()],
            "capacity": args.capacity,
            "sample_iters": args.sample_iters,
            "train_iters": args.train_iters,
            "combined_iters": args.combined_iters,
            "warmup": args.warmup,
            "batch_size": 16,
            "seq_len": 64,
            "batch_env_steps": 1024,
            "storage_bytes_per_transition": {
                "wpcp": 4116 * 4,
                "flat_hybrid": 16404 * 4,
                "modal_hybrid": (3 * 64 * 64) + (4116 * 4),
            },
        },
        "runs": {},
    }

    for kind, buffer in buffers.items():
        results["runs"].setdefault(kind, {})["sample_convert"] = _measure_sample_convert(
            buffer, convert_batch, args.sample_iters, args.warmup,
        )

    prepared_batches = {
        "wpcp_mlp3": convert_batch(buffers["wpcp"].sample(16, 64), 4),
        "flat_hybrid": convert_batch(buffers["flat_hybrid"].sample(16, 64), 4),
        "modal_hybrid": convert_batch(buffers["modal_hybrid"].sample(16, 64), 4),
    }
    for label, (_, agent) in agents.items():
        results["runs"].setdefault(label, {})["train_step"] = _measure_train_step(
            agent, prepared_batches[label], args.train_iters, args.warmup,
        )

    combined = {
        "wpcp_mlp3": ("wpcp", "wpcp_mlp3"),
        "flat_hybrid": ("flat_hybrid", "flat_hybrid"),
        "modal_hybrid": ("modal_hybrid", "modal_hybrid"),
    }
    for label, (buffer_kind, agent_kind) in combined.items():
        _, agent = agents[agent_kind]
        results["runs"].setdefault(label, {})["sample_convert_train"] = _measure_combined(
            buffers[buffer_kind], agent, convert_batch, args.combined_iters, args.warmup,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
