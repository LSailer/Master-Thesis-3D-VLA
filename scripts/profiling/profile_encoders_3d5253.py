"""Ad-hoc profiler: where does the time go for wp_cp (MLP) vs wp_dense (CNN)?

Isolates the two cost centers on a GPU node, with no Habitat dependency:
  1. VGGT extract()  -> the per-frame ACTING cost (shared by both readouts)
  2. agent.train_step() -> the per-gradient-step TRAINING cost (differs by encoder)
  3. encoder forward alone -> the encoder's share of train_step

Each variant uses its real batch/seq/depth so the numbers map onto the live runs:
  wp_cp        : vggt, obs (4116,),        B=16 T=64, vggt_mlp_layers=3
  wp_dense_cnn : vggt_wp_dense_cnn, obs (3,518,518), B=4 T=32
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.numpy as jnp

from src.configs.config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.shared.profiling import make_synthetic_rgb_frame, measure_ms


def timeit(fn, n=20, warmup=3):
    return measure_ms(fn, n=n, warmup=warmup)


def make_batch(cfg):
    B, T = cfg.batch_size, cfg.seq_len
    return {
        "obs": jnp.zeros((B, T, *cfg.obs_shape), jnp.float32),
        "actions": jax.nn.one_hot(jnp.zeros((B, T), jnp.int32), cfg.num_actions),
        "rewards": jnp.zeros((B, T)),
        "is_first": jnp.zeros((B, T)).at[:, 0].set(1.0),
        "is_episode_end": jnp.zeros((B, T)),
    }


def profile_train_step(name, enc_type, obs_shape, B, T, **kw):
    print(f"\n[{name}] building agent (enc={enc_type}, obs={obs_shape}, B={B}, T={T}, {kw}) ...", flush=True)
    cfg = R2DreamerConfig(encoder_type=enc_type, obs_shape=obs_shape,
                          num_actions=4, batch_size=B, seq_len=T,
                          imagination_horizon=15, **kw)
    agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))
    batch = make_batch(cfg)
    key = {"k": jax.random.PRNGKey(1)}

    def step():
        key["k"], sub = jax.random.split(key["k"])
        m = agent.train_step(batch, sub)
        float(m["total_loss"])  # force device sync

    print(f"[{name}] compiling + timing train_step ...", flush=True)
    mean_ms, std_ms = timeit(step, n=15, warmup=3)

    # encoder forward alone, over the B*T items a train_step encodes
    BT = B * T
    x = jnp.zeros((BT, *obs_shape), jnp.float32)
    enc_apply = jax.jit(lambda p, o: agent.encoder_mod.apply(p, o))
    ep = agent.params["encoder"]

    def enc_fwd():
        jax.block_until_ready(enc_apply(ep, x))

    enc_mean, enc_std = timeit(enc_fwd, n=30, warmup=3)
    print(f"[{name}] train_step   = {mean_ms:8.1f} ± {std_ms:5.1f} ms")
    print(f"[{name}] encoder_fwd  = {enc_mean:8.2f} ± {enc_std:4.2f} ms  (over B*T={BT} items)")
    return name, mean_ms, enc_mean, BT


def profile_vggt_extract():
    print("\n[vggt] loading InfiniteVGGT extractor ...", flush=True)
    from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor
    ext = JAXVGGTFeatureExtractor(total_budget=200_000,
                                  budgets_static=tuple([8333] * 24),
                                  compute_heads=True)
    rgb = make_synthetic_rgb_frame(0)
    ext.reset()
    # warm a few frames so we time steady-state (frame>0), not the first-frame graph
    for _ in range(3):
        ext.extract(rgb)["world_points"].block_until_ready()

    def ex_wpcp():
        ext.extract(rgb)["world_points"].block_until_ready()

    def ex_dense():
        ext.extract(rgb, return_dense=True)["dense_world_points"].block_until_ready()

    m1, s1 = timeit(ex_wpcp, n=20, warmup=2)
    m2, s2 = timeit(ex_dense, n=20, warmup=2)
    print(f"[vggt] extract (wp_cp readout)   = {m1:8.1f} ± {s1:5.1f} ms / frame")
    print(f"[vggt] extract (dense readout)   = {m2:8.1f} ± {s2:5.1f} ms / frame")
    return m1, m2


def main() -> None:
    print("devices:", jax.devices(), flush=True)
    ex_wpcp_ms, ex_dense_ms = profile_vggt_extract()
    r_wpcp = profile_train_step("wp_cp", "vggt", (4116,), 16, 64, vggt_mlp_layers=3)
    r_dense = profile_train_step("wp_dense", "vggt_wp_dense_cnn", (3, 518, 518), 4, 32)

    print("\n================ SUMMARY (mean ms) ================")
    print(f"{'phase':<22}{'wp_cp':>14}{'wp_dense':>14}")
    print(f"{'VGGT extract/frame':<22}{ex_wpcp_ms:>14.1f}{ex_dense_ms:>14.1f}")
    print(f"{'encoder fwd':<22}{r_wpcp[2]:>14.2f}{r_dense[2]:>14.2f}")
    print(f"{'train_step':<22}{r_wpcp[1]:>14.1f}{r_dense[1]:>14.1f}")
    print("---------------------------------------------------")
    print(f"train_step ratio wp_dense/wp_cp = {r_dense[1]/max(r_wpcp[1],1e-6):.2f}x")
    print(f"encoder fwd ratio  wp_dense/wp_cp = {r_dense[2]/max(r_wpcp[2],1e-6):.2f}x")
    print("Note: train_step encodes B*T items (wp_cp=1024, wp_dense=128);")
    print("the conv is per-item far heavier, so wp_dense is slower despite fewer items.")


if __name__ == "__main__":
    main()
