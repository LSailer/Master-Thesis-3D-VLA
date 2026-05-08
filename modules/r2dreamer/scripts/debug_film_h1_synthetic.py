#!/usr/bin/env python3
"""Fast H100 sanity check for VGGT-FiLM H1 ablation instrumentation.

This avoids slow Habitat/VGGT environment rollout and directly runs one JAX
train_step per FiLM ablation on synthetic VGGT-shaped batches. It verifies that:
- --film_ablation-equivalent config values compile and execute on GPU,
- grad/obs_* metrics are emitted,
- film/gamma/beta value diagnostics are emitted.

It is a code-path/debug check, not performance evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from modules.r2dreamer.agent import R2DreamerAgent
from modules.r2dreamer.config import R2DreamerConfig

FEATURE_DIM = 4116
OUT = Path("output/debug-film-h1-ablation/fast_synthetic_metrics.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())

variants = {
    "h1_none": dict(film_ablation="none"),
    "h1_zero_pose": dict(film_ablation="zero_pose"),
    "h1_zero_wp": dict(film_ablation="zero_wp"),
    "h4_pose_skip": dict(film_ablation="none", film_pose_skip=True),
    "h3_gamma_noise": dict(film_ablation="none", film_gamma_init_std=0.01),
    "h2_capacity_256_512": dict(film_ablation="none", film_channels=256, film_hidden=512),
}

results = {}
for name, overrides in variants.items():
    print(f"\n=== synthetic train_step variant={name} overrides={overrides} ===", flush=True)
    cfg = R2DreamerConfig(
        encoder_type="vggt_film_v1",
        obs_shape=(FEATURE_DIM,),
        num_actions=4,
        batch_size=2,
        seq_len=8,
        imagination_horizon=3,
        **overrides,
    )
    rng = jax.random.PRNGKey(123)
    agent = R2DreamerAgent(cfg, rng)
    B, T = cfg.batch_size, cfg.seq_len

    # Structured synthetic VGGT-like observation: nonzero world-points and pose.
    obs = jax.random.normal(rng, (B, T, FEATURE_DIM), dtype=jnp.float32)
    actions = jax.nn.one_hot(jnp.zeros((B, T), dtype=jnp.int32), cfg.num_actions)
    batch = {
        "obs": obs,
        "actions": actions,
        "rewards": jnp.zeros((B, T), dtype=jnp.float32),
        "is_first": jnp.zeros((B, T), dtype=jnp.float32).at[:, 0].set(1.0),
        "is_last": jnp.zeros((B, T), dtype=jnp.float32),
        "is_terminal": jnp.zeros((B, T), dtype=jnp.float32),
    }
    _, train_key = jax.random.split(rng)
    row = {}
    for update_idx in range(3):
        rng, train_key = jax.random.split(rng)
        metrics = agent.train_step(batch, train_key)
        keep = [
            "grad/obs_wp_norm",
            "grad/obs_pose_norm",
            "grad/obs_pose_to_wp_ratio",
            "film/gamma_minus_1_abs_mean",
            "film/gamma_actual_mean",
            "film/gamma_actual_std",
            "film/beta_abs_mean",
            "film/beta_rms",
            "total_loss",
        ]
        row = {k: float(metrics[k]) for k in keep if k in metrics}
        row["update_idx"] = update_idx
    results[name] = row
    for k, v in row.items():
        print(f"{k}: {v}")

OUT.write_text(json.dumps(results, indent=2, sort_keys=True))
print(f"\nWrote {OUT}")
