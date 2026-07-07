"""Microbenchmark: JAX ``train_step`` cost with and without the ``full_bf16`` gate.

``full_bf16`` only changes the JAX model compute (encoders, RSSM, heads,
optimizer math) — it does not touch the torch VGGT extractor (~132 ms/step)
or the host-side replay sampler (~59 ms/step amortized). So a full training
probe would bury the effect under those fixed costs. This entrypoint isolates
what the gate actually changes: it builds the hybrid house-points-pose agent
at production shape, feeds a synthetic ``ReplayBatch``, and times the JIT
``train_step`` for the gate off vs. on. It never imports habitat, so it runs
on the shared ``.venv`` even when the habitat/numpy stack is unimportable.

Run on a GPU node with the worktree-local interpreter (no ``uv`` sync):

    .venv/bin/python -m scripts.profiling.profile_full_bf16_step \
        --batch 16 --seq 64 --iters 50
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from src.buffer.replay_buffer import ReplayBatch
from src.configs.agent_config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.encoders.constants import (
    HOUSE_CONTEXT_MAX_POINTS,
    HOUSE_POINT_DIM,
)
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
    HYBRID_IMAGE_KEY,
)

CAMERA_POSE_DIM = 9


def build_agent(full_bf16: bool, num_actions: int) -> R2DreamerAgent:
    """Build the prod-shape hybrid agent with the ``full_bf16`` gate set.

    Args:
      full_bf16: Whether to enable the model-wide bfloat16 compute gate.
      num_actions: Discrete action count (habitat objectnav uses 4).

    Returns:
      An initialized ``R2DreamerAgent`` using production encoder defaults.
    """
    cfg = R2DreamerConfig(
        encoder_type="vggt_hybrid_house_points_pose",
        obs_shape={
            HYBRID_IMAGE_KEY: (3, 64, 64),
            CAMERA_POSE_KEY: (CAMERA_POSE_DIM,),
            HOUSE_CONTEXT_KEY: (HOUSE_CONTEXT_MAX_POINTS, HOUSE_POINT_DIM),
            HOUSE_CONTEXT_SIZE_KEY: (),
        },
        num_actions=num_actions,
        compute_dtype="bfloat16",
        full_bf16=full_bf16,
        warmup_steps=0,
    )
    return R2DreamerAgent(cfg, jax.random.PRNGKey(0))


def make_batch(rng, batch: int, seq: int, num_actions: int, house_size: int):
    """Build a synthetic production-shape replay batch.

    Args:
      rng: PRNG key.
      batch: Batch dimension B.
      seq: Sequence length T.
      num_actions: Discrete action count for the one-hot actions.
      house_size: Valid-point count written into the size field (does not
        change compute — all snapshot rows run through the point MLP).

    Returns:
      A ``ReplayBatch`` with the hybrid house-points-pose observation layout.
    """
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    return ReplayBatch(
        obs={
            HYBRID_IMAGE_KEY: jax.random.uniform(k1, (batch, seq, 3, 64, 64)),
            CAMERA_POSE_KEY: jax.random.normal(
                k2, (batch, seq, CAMERA_POSE_DIM)
            ).astype(jnp.float16),
            HOUSE_CONTEXT_KEY: jax.random.normal(
                k3, (HOUSE_CONTEXT_MAX_POINTS, HOUSE_POINT_DIM)
            ).astype(jnp.float16),
            HOUSE_CONTEXT_SIZE_KEY: jnp.asarray(house_size, dtype=jnp.int32),
        },
        actions=jax.nn.one_hot(
            jax.random.randint(k4, (batch, seq), 0, num_actions), num_actions
        ),
        rewards=jax.random.normal(k4, (batch, seq)),
        is_first=jnp.zeros((batch, seq)).at[:, 0].set(1.0),
        is_episode_end=jnp.zeros((batch, seq)),
    )


def time_train_step(agent, batch, iters: int, warmup: int) -> list[float]:
    """Time ``iters`` JIT train steps after ``warmup`` compile/warm iterations.

    Args:
      agent: Built agent to step.
      batch: Fixed synthetic batch reused every step.
      iters: Number of timed steps.
      warmup: Number of untimed warmup steps (covers JIT compilation).

    Returns:
      Per-step wall-clock times in milliseconds.
    """
    for i in range(warmup):
        m = agent.train_step(batch, jax.random.PRNGKey(1000 + i))
        jax.block_until_ready(m)
    times = []
    for i in range(iters):
        t0 = time.perf_counter()
        m = agent.train_step(batch, jax.random.PRNGKey(i))
        jax.block_until_ready(m)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def summarize(label: str, times: list[float]) -> float:
    """Print and return the median of a per-step timing list.

    Args:
      label: Row label for the printout.
      times: Per-step times in milliseconds.

    Returns:
      The median step time in milliseconds.
    """
    med = statistics.median(times)
    p10 = statistics.quantiles(times, n=10)[0]
    p90 = statistics.quantiles(times, n=10)[-1]
    print(
        f"{label:>14}: median {med:7.2f} ms  p10 {p10:7.2f}  p90 {p90:7.2f}  "
        f"(n={len(times)})"
    )
    return med


def main() -> None:
    """Run the gate-off vs gate-on train_step benchmark and print the delta."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--house-size", type=int, default=200_000)
    args = parser.parse_args()

    print(f"devices: {jax.devices()}")
    print(
        f"shape: B={args.batch} T={args.seq} "
        f"house_points={HOUSE_CONTEXT_MAX_POINTS} "
        f"iters={args.iters} warmup={args.warmup}"
    )
    batch = make_batch(
        jax.random.PRNGKey(7), args.batch, args.seq, args.num_actions, args.house_size
    )

    results = {}
    for full_bf16 in (False, True):
        agent = build_agent(full_bf16, args.num_actions)
        # Confirm the gate actually took effect on the encoder output.
        embed = agent.encoder_mod.apply(agent.params["encoder"], batch.obs)
        label = "full_bf16" if full_bf16 else "float32"
        print(f"{label} encoder embed dtype: {embed.dtype}")
        times = time_train_step(agent, batch, args.iters, args.warmup)
        results[label] = summarize(label, times)

    base, gated = results["float32"], results["full_bf16"]
    delta = gated - base
    pct = 100.0 * delta / base if base else 0.0
    print(
        f"\ntrain_step delta (full_bf16 - float32): {delta:+.2f} ms/step "
        f"({pct:+.1f}%)"
    )


if __name__ == "__main__":
    main()
