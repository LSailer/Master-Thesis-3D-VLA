"""JAX R2-Dreamer vs PyTorch R2-Dreamer parity training.

Same data, same batch order (seed 42). Saves per-step metrics to JSON for
the parity report notebook.

Public entry: run(train_steps, output_path, argv)
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
EXT = os.path.join(ROOT, "external", "r2dreamer")

from modules.r2dreamer.launch.parity.batch_utils import (
    SEED, WARMUP_STEPS, BATCH_SIZE, SEQ_LEN, OBS_SHAPE_CHW,
    NUM_ACTIONS, collect_crafter_data, precompute_batch_starts,
    _convert_batch, make_batch_torch, make_pytorch_config, make_crafter_spaces,
)

LOG_EVERY = 100
NUM_COLLECT = 20_000


def _train_step_pytorch(agent, data, initial):
    import torch
    from torch.amp import autocast
    torch.compiler.cudagraph_mark_step_begin()
    p_data = agent.preprocess(data)
    agent._update_slow_target()
    with autocast(device_type=agent.device.type, dtype=torch.float16):
        (stoch, deter), mets = agent._cal_grad(p_data, initial)
    agent._scaler.unscale_(agent._optimizer)
    agent._agc(agent._named_params.values())
    agent._scaler.step(agent._optimizer)
    agent._scaler.update()
    agent._scheduler.step()
    agent._optimizer.zero_grad(set_to_none=True)
    return mets, stoch, deter


def _run_jax(transitions, all_starts, train_steps, outpath):
    import jax
    from modules.r2dreamer.config import R2DreamerConfig
    from modules.r2dreamer.agent import R2DreamerAgent
    from modules.r2dreamer.world_model.encoders import ConvEncoder

    cfg = R2DreamerConfig(
        obs_shape=OBS_SHAPE_CHW, num_actions=NUM_ACTIONS,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        encoder_module_cls=ConvEncoder,
    )
    rng = jax.random.PRNGKey(SEED)
    rng, init_key = jax.random.split(rng)
    agent = R2DreamerAgent(cfg, init_key)

    for i in range(WARMUP_STEPS):
        rng, k = jax.random.split(rng)
        batch = _convert_batch(transitions, all_starts[i])
        _ = agent.train_step(batch, k)
        jax.block_until_ready(agent.params)

    rows = []
    t_start = time.perf_counter()
    for i in range(train_steps):
        rng, k = jax.random.split(rng)
        batch = _convert_batch(transitions, all_starts[WARMUP_STEPS + i])
        t0 = time.perf_counter()
        metrics = agent.train_step(batch, k)
        jax.block_until_ready(agent.params)
        t1 = time.perf_counter()

        if (i + 1) % LOG_EVERY == 0:
            row = {"step": i + 1, "step_ms": (t1 - t0) * 1000}
            row.update({k: float(v) for k, v in metrics.items()})
            rows.append(row)

        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t_start
            sps = (i + 1) / elapsed
            eta = (train_steps - i - 1) / sps / 3600
            print(f"  [JAX] step {i+1}/{train_steps} | loss={metrics.get('total_loss',0):.2f} | "
                  f"dyn={metrics.get('loss/dyn',0):.2f} | {sps:.1f} sps | ETA {eta:.1f}h")

    elapsed = time.perf_counter() - t_start
    print(f"  [JAX] Done in {elapsed/60:.1f} min ({train_steps/elapsed:.1f} sps)")

    with open(outpath, "w") as f:
        json.dump(rows, f)
    print(f"  Saved {len(rows)} metrics rows to {outpath}")
    del agent
    gc.collect()


def _run_pytorch(transitions, all_starts, train_steps, outpath, device):
    import torch
    sys.path.insert(0, EXT)
    from dreamer import Dreamer

    cfg = make_pytorch_config(device)
    obs_space, act_space = make_crafter_spaces()
    agent = Dreamer(cfg, obs_space, act_space).to(device)

    for i in range(WARMUP_STEPS):
        data = make_batch_torch(transitions, all_starts[i], device)
        stoch0 = torch.zeros(BATCH_SIZE, cfg.rssm.stoch, cfg.rssm.discrete, device=device)
        deter0 = torch.zeros(BATCH_SIZE, cfg.rssm.deter, device=device)
        _ = _train_step_pytorch(agent, data, (stoch0, deter0))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    rows = []
    t_start = time.perf_counter()
    for i in range(train_steps):
        data = make_batch_torch(transitions, all_starts[WARMUP_STEPS + i], device)
        stoch0 = torch.zeros(BATCH_SIZE, cfg.rssm.stoch, cfg.rssm.discrete, device=device)
        deter0 = torch.zeros(BATCH_SIZE, cfg.rssm.deter, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mets, _, _ = _train_step_pytorch(agent, data, (stoch0, deter0))
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if (i + 1) % LOG_EVERY == 0:
            row = {"step": i + 1, "step_ms": (t1 - t0) * 1000}
            row.update({k: float(v) if isinstance(v, torch.Tensor) else float(v)
                        for k, v in mets.items()})
            rows.append(row)

        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t_start
            sps = (i + 1) / elapsed
            eta = (train_steps - i - 1) / sps / 3600
            loss = float(mets.get("opt/loss", 0))
            dyn = float(mets.get("loss/dyn", 0))
            print(f"  [PT]  step {i+1}/{train_steps} | loss={loss:.2f} | "
                  f"dyn={dyn:.2f} | {sps:.1f} sps | ETA {eta:.1f}h")

    elapsed = time.perf_counter() - t_start
    print(f"  [PT]  Done in {elapsed/60:.1f} min ({train_steps/elapsed:.1f} sps)")

    with open(outpath, "w") as f:
        json.dump(rows, f)
    print(f"  Saved {len(rows)} metrics rows to {outpath}")

    del agent
    torch.cuda.empty_cache()
    gc.collect()


def run(*, train_steps=100_000, output_path=None, argv=None):
    """Run JAX vs PyTorch r2dreamer parity training and save metrics JSON.

    Args:
        train_steps: number of train steps per framework.
        output_path: directory for output JSON files; defaults to ROOT/output/methods/parity.
        argv: argument list (default: sys.argv[1:]).

    Returns:
        dict with keys "jax_path" and "pytorch_path" (str paths to JSON files).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=train_steps)
    parser.add_argument("--jax-only", action="store_true")
    parser.add_argument("--pytorch-only", action="store_true")
    args = parser.parse_args(argv)

    import torch
    import jax
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"PyTorch {torch.__version__}, JAX {jax.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: train_steps={args.train_steps}, seed={SEED}\n")

    print("Collecting Crafter data...")
    transitions = collect_crafter_data(NUM_COLLECT, seed=SEED)
    total_steps = WARMUP_STEPS + args.train_steps
    all_starts = precompute_batch_starts(total_steps, transitions, SEED)
    print(f"Collected {len(transitions)} transitions, {len(all_starts)} batch starts\n")

    outdir = output_path or os.path.join(ROOT, "output", "parity")
    os.makedirs(outdir, exist_ok=True)

    jax_path = os.path.join(outdir, "jax_metrics.json")
    pytorch_path = os.path.join(outdir, "pytorch_metrics.json")

    if not args.pytorch_only:
        print("=== R2-Dreamer (JAX) ===")
        _run_jax(transitions, all_starts, args.train_steps, jax_path)
        print()

    if not args.jax_only:
        print("=== R2-Dreamer (PyTorch) ===")
        _run_pytorch(transitions, all_starts, args.train_steps, pytorch_path, device)
        print()

    print("Done! Results in", outdir)
    return {"jax_path": jax_path, "pytorch_path": pytorch_path}
