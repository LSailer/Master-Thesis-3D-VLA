"""Embedding stability under live-buffer growth: flatten vs masked pooling.

Simulates the HouseContextPoseBuffer snapshot pipeline: a house-like point
cloud is ingested in exploration order; at several fill levels the buffer is
snapshotted with the production even-stride rule
``idx = floor(arange(max_points) * size / max_points)`` (see
src/buffer/house_context_pose_buffer.py::_house_context_snapshot).

Three fixed random-weight encoders embed each snapshot:
  pool_mlp      point-MLP (2x Dense(256)+SiLU) -> masked mean+max -> Dense(1024)
                (same shape as HousePointsCameraEncoder._house_embedding)
  flat_raw      even-stride subsample to 4096 pts in insertion order,
                flatten (24576,) -> Dense(1024)
  flat_sorted   same, but points sorted by voxel key before flattening

Metric: cosine similarity of each snapshot's embedding to the FINAL full-map
embedding, i.e. "how much does the world-model input churn while the map is
still growing?". Random weights measure representational stability, not task
performance (random features approximately preserve geometry).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

MAX_POINTS = 262_144
FLAT_POINTS = 4_096
EMBED = 1_024
HIDDEN = 256
VOXEL = 0.01

key = jax.random.PRNGKey(0)


def make_house_cloud(key, n_target=4_000_000):
    """Samples a house-like cloud (floor + walls + furniture boxes), 1cm voxels.

    Args:
      key: PRNG key.
      n_target: Approximate number of surviving voxel-deduped points.

    Returns:
      (N,6) float32 xyzrgb in insertion (exploration) order.
    """
    ks = jax.random.split(key, 8)
    n_raw = int(n_target * 1.35)
    # Floor 12x10m, walls, and 12 furniture boxes.
    n_floor = n_raw // 3
    floor = jnp.stack(
        [
            jax.random.uniform(ks[0], (n_floor,)) * 12.0,
            jax.random.uniform(ks[1], (n_floor,)) * 10.0,
            jnp.zeros(n_floor),
        ],
        axis=-1,
    )
    n_wall = n_raw // 3
    t = jax.random.uniform(ks[2], (n_wall,))
    z = jax.random.uniform(ks[3], (n_wall,)) * 2.5
    side = jax.random.randint(ks[4], (n_wall,), 0, 4)
    wx = jnp.where(side == 0, t * 12.0, jnp.where(side == 1, t * 12.0, jnp.where(side == 2, 0.0, 12.0)))
    wy = jnp.where(side == 0, 0.0, jnp.where(side == 1, 10.0, t * 10.0))
    walls = jnp.stack([wx, wy, z], axis=-1)
    n_furn = n_raw - n_floor - n_wall
    centers = jax.random.uniform(ks[5], (12, 3)) * jnp.array([11.0, 9.0, 0.0]) + jnp.array([0.5, 0.5, 0.4])
    fi = jax.random.randint(ks[6], (n_furn,), 0, 12)
    offs = (jax.random.uniform(ks[7], (n_furn, 3)) - 0.5) * jnp.array([1.2, 1.2, 0.8])
    furn = centers[fi] + offs
    xyz = jnp.concatenate([floor, walls, furn], axis=0)
    rgb = jnp.clip(xyz / jnp.array([12.0, 10.0, 2.5]), 0.0, 1.0)  # color ~ position proxy
    # Voxel dedup at 1cm like the production buffer.
    vox = jnp.floor(xyz / VOXEL).astype(jnp.int32)
    hkey = vox[:, 0] * 73_856_093 ^ vox[:, 1] * 19_349_663 ^ vox[:, 2] * 83_492_791
    _, first = jnp.unique(hkey, return_index=True, size=xyz.shape[0], fill_value=-1)
    first = first[first >= 0]
    pts = jnp.concatenate([xyz, rgb], axis=-1)[first]
    # Exploration order: sweep by angle around the house centre + jitter.
    ang = jnp.arctan2(pts[:, 1] - 5.0, pts[:, 0] - 6.0)
    ang = ang + 0.15 * jax.random.normal(jax.random.PRNGKey(9), ang.shape)
    return pts[jnp.argsort(ang)].astype(jnp.float32)


def snapshot(pts, size):
    """Production even-stride snapshot to fixed (MAX_POINTS,6) + valid count."""
    size = min(size, pts.shape[0])
    if size <= MAX_POINTS:
        out = jnp.zeros((MAX_POINTS, 6), jnp.float32).at[:size].set(pts[:size])
        return out, size
    idx = jnp.floor(jnp.arange(MAX_POINTS) * (size / MAX_POINTS)).astype(jnp.int32)
    return pts[:size][idx], MAX_POINTS


kw = jax.random.split(jax.random.PRNGKey(42), 6)
w1 = (jax.random.normal(kw[0], (6, HIDDEN)) / jnp.sqrt(6)).astype(jnp.bfloat16)
w2 = (jax.random.normal(kw[1], (HIDDEN, HIDDEN)) / jnp.sqrt(HIDDEN)).astype(jnp.bfloat16)
w3 = (jax.random.normal(kw[2], (2 * HIDDEN, EMBED)) / jnp.sqrt(2 * HIDDEN)).astype(jnp.bfloat16)
wf = (jax.random.normal(kw[3], (FLAT_POINTS * 6, EMBED)) / jnp.sqrt(FLAT_POINTS * 6)).astype(jnp.bfloat16)


def pool_mlp(snap, size):
    x = jax.nn.silu(snap.astype(jnp.bfloat16) @ w1)
    x = jax.nn.silu(x @ w2)
    valid = (jnp.arange(MAX_POINTS) < size)[:, None]
    mean = (x * valid).sum(0) / jnp.maximum(size, 1)
    mx = jnp.where(valid, x, -jnp.inf).max(0)
    return (jnp.concatenate([mean, mx]) @ w3).astype(jnp.float32)


def flat(snap, size, sort):
    idx = jnp.floor(jnp.arange(FLAT_POINTS) * (min(size, MAX_POINTS) / FLAT_POINTS)).astype(jnp.int32)
    sub = snap[idx]
    if sort:
        vox = jnp.floor(sub[:, :3] / VOXEL).astype(jnp.int32)
        order = jnp.lexsort((vox[:, 2], vox[:, 1], vox[:, 0]))
        sub = sub[order]
    return (sub.astype(jnp.bfloat16).reshape(-1) @ wf).astype(jnp.float32)


def cos(a, b):
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8))


pts = make_house_cloud(key)
total = pts.shape[0]
print(f"house cloud: {total} voxel-deduped points")
sizes = [s for s in [100_000, 262_144, 500_000, 1_000_000, 2_000_000, total] if s <= total]

ref = {name: None for name in ("pool_mlp", "flat_raw", "flat_sorted")}
rows = []
for s in sizes:
    snap, valid = snapshot(pts, s)
    e = {
        "pool_mlp": pool_mlp(snap, valid),
        "flat_raw": flat(snap, valid, sort=False),
        "flat_sorted": flat(snap, valid, sort=True),
    }
    rows.append((s, e))

final = rows[-1][1]
print(f"\ncosine similarity to final full-map embedding ({total} pts):")
print(f"{'buffer size':>12s} {'pool_mlp':>9s} {'flat_raw':>9s} {'flat_sorted':>12s}")
for s, e in rows:
    print(
        f"{s:>12d} {cos(e['pool_mlp'], final['pool_mlp']):>9.4f}"
        f" {cos(e['flat_raw'], final['flat_raw']):>9.4f}"
        f" {cos(e['flat_sorted'], final['flat_sorted']):>12.4f}"
    )

print("\nconsecutive-snapshot similarity (input churn step-to-step):")
for (s0, e0), (s1, e1) in zip(rows, rows[1:]):
    print(
        f"{s0:>9d}->{s1:<9d} pool={cos(e0['pool_mlp'], e1['pool_mlp']):.4f}"
        f" flat_raw={cos(e0['flat_raw'], e1['flat_raw']):.4f}"
        f" flat_sorted={cos(e0['flat_sorted'], e1['flat_sorted']):.4f}"
    )

print("\nparameter counts:")
print(f"  pool_mlp head (prod shape) : {6*HIDDEN + HIDDEN*HIDDEN + 2*HIDDEN*EMBED:,}")
print(f"  flatten 4096 pts -> 1024   : {FLAT_POINTS*6*EMBED:,}")
print(f"  flatten 262144 pts -> 1024 : {MAX_POINTS*6*EMBED:,} (infeasible)")
