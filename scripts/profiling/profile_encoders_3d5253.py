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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

import jax
import jax.numpy as jnp

from src.configs.config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.shared.profiling import measure_ms
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor


def _make_synthetic_rgb_frame(seed: int, size: int = 518) -> np.ndarray:
    """Return a deterministic ``(3, size, size)`` uint8 RGB frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(3, size, size), dtype=np.uint8)


def timeit(fn, n=20, warmup=3):
    """Time ``fn`` and return mean and standard deviation in milliseconds.

    Args:
        fn: Nullary callable to benchmark.
        n: Number of timed iterations.
        warmup: Untimed warmup iterations before measurement.

    Returns:
        ``(mean_ms, std_ms)`` from :func:`measure_ms`.
    """
    return measure_ms(fn, n=n, warmup=warmup)


def make_batch(cfg):
    """Build a zero-filled synthetic replay batch for ``cfg`` shapes.

    Args:
        cfg: Agent config supplying ``batch_size``, ``seq_len``, and ``obs_shape``.

    Returns:
        Batch dict with ``obs``, ``actions``, ``rewards``, ``is_first``,
        and ``is_episode_end`` arrays.
    """
    batch_size, seq_len = cfg.batch_size, cfg.seq_len
    return {
        "obs": jnp.zeros((batch_size, seq_len, *cfg.obs_shape), jnp.float32),
        "actions": jax.nn.one_hot(
            jnp.zeros((batch_size, seq_len), jnp.int32), cfg.num_actions
        ),
        "rewards": jnp.zeros((batch_size, seq_len)),
        "is_first": jnp.zeros((batch_size, seq_len)).at[:, 0].set(1.0),
        "is_episode_end": jnp.zeros((batch_size, seq_len)),
    }


def _time_encoder_forward(agent, obs_shape, num_items, name):
    """Time a JIT-compiled encoder forward pass over ``num_items`` observations.

    Args:
        agent: Initialized :class:`R2DreamerAgent`.
        obs_shape: Per-item observation shape tuple.
        num_items: Number of flattened batch*seq items to encode.
        name: Label printed in timing output.

    Returns:
        Mean encoder forward time in milliseconds.
    """
    obs = jnp.zeros((num_items, *obs_shape), jnp.float32)
    enc_apply = jax.jit(agent.encoder_mod.apply)
    enc_params = agent.params["encoder"]

    def enc_fwd():
        jax.block_until_ready(enc_apply(enc_params, obs))

    enc_mean, enc_std = timeit(enc_fwd, n=30, warmup=3)
    print(
        f"[{name}] encoder_fwd  = {enc_mean:8.2f} ± {enc_std:4.2f} ms"
        f"  (over B*T={num_items} items)"
    )
    return enc_mean


def profile_train_step(name, enc_type, obs_shape, batch_size, seq_len, **kw):
    """Profile ``train_step`` and encoder forward for one encoder variant.

    Args:
        name: Label printed in timing output.
        enc_type: ``encoder_type`` passed to :class:`R2DreamerConfig`.
        obs_shape: Observation shape tuple for the encoder.
        batch_size: Batch size.
        seq_len: Sequence length.
        **kw: Extra keyword args forwarded to :class:`R2DreamerConfig`.

    Returns:
        ``(name, train_step_ms, encoder_fwd_ms, num_items)`` where
        ``num_items`` is ``batch_size * seq_len``.
    """
    print(
        f"\n[{name}] building agent (enc={enc_type}, obs={obs_shape}, "
        f"B={batch_size}, T={seq_len}, {kw}) ...",
        flush=True,
    )
    cfg = R2DreamerConfig(
        encoder_type=enc_type,
        obs_shape=obs_shape,
        num_actions=4,
        batch_size=batch_size,
        seq_len=seq_len,
        imagination_horizon=15,
        **kw,
    )
    agent = R2DreamerAgent(cfg, jax.random.PRNGKey(0))
    batch = make_batch(cfg)
    key = {"k": jax.random.PRNGKey(1)}

    def step():
        key["k"], sub = jax.random.split(key["k"])
        metrics = agent.train_step(batch, sub)
        float(metrics["total_loss"])  # force device sync

    print(f"[{name}] compiling + timing train_step ...", flush=True)
    mean_ms, std_ms = timeit(step, n=15, warmup=3)
    print(f"[{name}] train_step   = {mean_ms:8.1f} ± {std_ms:5.1f} ms")

    num_items = batch_size * seq_len
    enc_mean = _time_encoder_forward(agent, obs_shape, num_items, name)
    return name, mean_ms, enc_mean, num_items


def profile_vggt_extract():
    """Time steady-state VGGT ``extract`` for wp_cp and dense readouts.

    Returns:
        ``(wp_cp_ms, dense_ms)`` mean per-frame extract times in milliseconds.
    """
    print("\n[vggt] loading InfiniteVGGT extractor ...", flush=True)
    ext = JAXVGGTFeatureExtractor(
        total_budget=200_000,
        budgets_static=tuple([8333] * 24),
        compute_heads=True,
    )
    rgb = _make_synthetic_rgb_frame(0)
    ext.reset()
    # warm a few frames so we time steady-state (frame>0), not the first-frame graph
    for _ in range(3):
        ext.extract(rgb).world_points.block_until_ready()

    def ex_wpcp():
        ext.extract(rgb).world_points.block_until_ready()

    def ex_dense():
        ext.extract(rgb, return_dense=True).world_points.block_until_ready()

    m1, s1 = timeit(ex_wpcp, n=20, warmup=2)
    m2, s2 = timeit(ex_dense, n=20, warmup=2)
    print(f"[vggt] extract (wp_cp readout)   = {m1:8.1f} ± {s1:5.1f} ms / frame")
    print(f"[vggt] extract (dense readout)   = {m2:8.1f} ± {s2:5.1f} ms / frame")
    return m1, m2


def main() -> None:
    """Run VGGT extract and encoder train-step benchmarks and print a summary."""
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
