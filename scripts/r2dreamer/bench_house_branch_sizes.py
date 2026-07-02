"""GPU bench: house-branch encoder cost vs snapshot size N.

Decides between "single static N + masked pooling" and the bucketed
split-boundary design (scratchpad/experiments/split_boundary_encoder): if the
house branch is cheap even at N = 256k-8.4M, one static shape wins on
simplicity.

Replicates HousePointsCameraEncoder's house branch dims (encoders/mlp.py:
Dense 6->256->256 per point, [mean ‖ max] pool, project to 1024; float32 math
as in production) plus masked pooling, and times, per N:

    fwd      - policy-path cost (embedding only)
    fwd+bwd  - train-path cost (value_and_grad w.r.t. params)

The branch runs once per batch in production (singleton broadcast), so these
numbers are per-train-step / per-env-step additions.

Run on a GPU node:  uv run python scripts/r2dreamer/bench_house_branch_sizes.py
"""

from __future__ import annotations

import functools
import statistics
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

POINT_DIM = 6
HIDDEN = 256
EMBED_DIM = 1024
SIZES = [4096, 32768, 65536, 262144, 1048576, 2097152, 8388608]
ITERS = 30
WARMUP = 5


class Params(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    wp: jax.Array
    bp: jax.Array


def init_params(key: jax.Array) -> Params:
    ks = jax.random.split(key, 3)
    return Params(
        w1=0.1 * jax.random.normal(ks[0], (POINT_DIM, HIDDEN), jnp.float32),
        b1=jnp.zeros((HIDDEN,), jnp.float32),
        w2=0.1 * jax.random.normal(ks[1], (HIDDEN, HIDDEN), jnp.float32),
        b2=jnp.zeros((HIDDEN,), jnp.float32),
        wp=0.1 * jax.random.normal(ks[2], (2 * HIDDEN, EMBED_DIM), jnp.float32),
        bp=jnp.zeros((EMBED_DIM,), jnp.float32),
    )


def house_branch(params: Params, snapshot: jax.Array, size: jax.Array) -> jax.Array:
    """Masked per-point MLP + [mean ‖ max] pool -> (EMBED_DIM,)."""
    n = snapshot.shape[0]
    mask = (jnp.arange(n) < size)[:, None]
    x = snapshot.astype(jnp.float32)
    x = jax.nn.silu(x @ params.w1 + params.b1)
    x = jax.nn.silu(x @ params.w2 + params.b2)
    denom = jnp.maximum(size, 1).astype(jnp.float32)
    mean = (x * mask).sum(axis=0) / denom
    maxp = jnp.where(mask, x, -jnp.inf).max(axis=0)
    return jnp.concatenate([mean, maxp]) @ params.wp + params.bp


@jax.jit
def fwd(params: Params, snapshot: jax.Array, size: jax.Array) -> jax.Array:
    return house_branch(params, snapshot, size)


@jax.jit
def fwd_bwd(params: Params, snapshot: jax.Array, size: jax.Array):
    def loss(p: Params) -> jax.Array:
        return jnp.mean(house_branch(p, snapshot, size) ** 2)

    return jax.value_and_grad(loss)(params)


def timed(fn, *args) -> float:
    xs = []
    for i in range(ITERS + WARMUP):
        t = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        if i >= WARMUP:
            xs.append((time.perf_counter() - t) * 1e3)
    return statistics.median(xs)


def mem_mb() -> float:
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use", 0) / 2**20


def main() -> None:
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"house branch dims: {POINT_DIM}->{HIDDEN}->{HIDDEN}->pool->{EMBED_DIM}  "
          f"float32, fill=50% (worst padding within a bucket)\n")
    params = init_params(jax.random.PRNGKey(0))

    print(f"{'N':>9} {'fwd_ms':>8} {'fwd+bwd_ms':>11} {'peak_mem_MB':>12}")
    for n in SIZES:
        snapshot = jax.random.normal(
            jax.random.PRNGKey(n), (n, POINT_DIM), jnp.float16
        )
        size = jnp.asarray(n // 2, jnp.int32)  # half-full bucket
        f_ms = timed(fwd, params, snapshot, size)
        fb_ms = timed(fwd_bwd, params, snapshot, size)
        print(f"{n:9d} {f_ms:8.2f} {fb_ms:11.2f} {mem_mb():12.0f}")

    print("\ncontext: VGGT extract is ~64 ms/env-step; train_step is the "
          "dominant train cost. The house branch runs once per batch.")


if __name__ == "__main__":
    main()
