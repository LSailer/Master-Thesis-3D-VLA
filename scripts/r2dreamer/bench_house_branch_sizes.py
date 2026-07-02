"""GPU bench: house-branch encoder cost vs snapshot size N.

Decides between "single static N + masked pooling" and the bucketed
split-boundary design (scratchpad/experiments/split_boundary_encoder): if the
house branch is cheap even at N = 256k-8.4M, one static shape wins on
simplicity.

Times the REAL production module — ``HousePointsCameraEncoder`` from
encoders/mlp.py (per-point Dense 6->256->256 with RMSNorm/SiLU, masked
[mean ‖ max] pool, project to 1024, fused with the camera branch) — per N:

    fwd      - policy-path cost (embedding only)
    fwd+bwd  - train-path cost (value_and_grad w.r.t. params)

The camera branch (a 9-dim MLP) rides along at negligible cost, so the
numbers are house-branch dominated. The branch runs once per batch in
production (singleton broadcast), so these are per-train-step / per-env-step
additions.

Run on a GPU node:  uv run python scripts/r2dreamer/bench_house_branch_sizes.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp

from src.r2dreamer.encoders.constants import HOUSE_POINT_DIM
from src.r2dreamer.encoders.mlp import HousePointsCameraEncoder
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
)

SIZES = [4096, 32768, 65536, 262144, 1048576, 2097152, 8388608]
ITERS = 30
WARMUP = 5

ENCODER = HousePointsCameraEncoder()


def obs_for(n: int) -> dict[str, jax.Array]:
    """Half-full snapshot (worst padding within a bucket) plus one camera pose."""
    return {
        CAMERA_POSE_KEY: jnp.zeros((1, ENCODER.camera_pose_dim), jnp.float16),
        HOUSE_CONTEXT_KEY: jax.random.normal(
            jax.random.PRNGKey(n), (n, HOUSE_POINT_DIM), jnp.float16
        ),
        HOUSE_CONTEXT_SIZE_KEY: jnp.asarray(n // 2, jnp.int32),
    }


@jax.jit
def fwd(variables, obs: dict[str, jax.Array]) -> jax.Array:
    return ENCODER.apply(variables, obs)


@jax.jit
def fwd_bwd(variables, obs: dict[str, jax.Array]):
    def loss(v) -> jax.Array:
        return jnp.mean(ENCODER.apply(v, obs) ** 2)

    return jax.value_and_grad(loss)(variables)


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
    print(
        f"module: HousePointsCameraEncoder "
        f"({HOUSE_POINT_DIM}->{ENCODER.point_hidden}x{ENCODER.point_layers}"
        f"->pool->{ENCODER.embed_dim}), fill=50%\n"
    )
    # Dense params are shape-independent of N, so one init serves every size.
    variables = ENCODER.init(jax.random.PRNGKey(0), obs_for(SIZES[0]))

    print(f"{'N':>9} {'fwd_ms':>8} {'fwd+bwd_ms':>11} {'peak_mem_MB':>12}")
    for n in SIZES:
        obs = obs_for(n)
        f_ms = timed(fwd, variables, obs)
        fb_ms = timed(fwd_bwd, variables, obs)
        print(f"{n:9d} {f_ms:8.2f} {fb_ms:11.2f} {mem_mb():12.0f}")

    print("\ncontext: VGGT extract is ~64 ms/env-step; train_step is the "
          "dominant train cost. The house branch runs once per batch.")


if __name__ == "__main__":
    main()
