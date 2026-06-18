"""3-way R2-Dreamer benchmark: JAX R2 vs PyTorch R2 vs PyTorch DreamerV3.

Trains each variant for train_steps on identical Crafter replay data,
measures timing/memory/losses, evaluates policies, and saves all results
to output/methods/comparisons/r2dreamer_benchmark.json.

Public entry: run(train_steps, eval_episodes, output_path, argv)
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

from src.r2dreamer.launch.parity.batch_utils import (
    SEED,
    WARMUP_STEPS,
    BATCH_SIZE,
    SEQ_LEN,
    OBS_SHAPE_CHW,
    NUM_ACTIONS,
    collect_crafter_data,
    precompute_batch_starts,
    _convert_batch,
    make_batch_torch,
    make_pytorch_config,
    make_crafter_spaces,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
EXT = os.path.join(ROOT, "external", "r2dreamer")

NUM_COLLECT = 10_000
EVAL_MAX_STEPS = 500


def _train_step_pytorch(agent, data, initial):
    import torch
    from torch.amp import autocast

    torch.compiler.cudagraph_mark_step_begin()
    p_data = agent.preprocess(data)
    agent._update_slow_target()
    metrics = {}
    with autocast(device_type=agent.device.type, dtype=torch.float16):
        (stoch, deter), mets = agent._cal_grad(p_data, initial)
    agent._scaler.unscale_(agent._optimizer)
    agent._agc(agent._named_params.values())
    agent._scaler.step(agent._optimizer)
    agent._scaler.update()
    agent._scheduler.step()
    agent._optimizer.zero_grad(set_to_none=True)
    metrics.update(mets)
    return metrics, stoch, deter


def _eval_jax_agent(agent, num_episodes, max_steps, rng_key):
    import jax
    from src.environments.crafter import CrafterEnv

    env = CrafterEnv(size=(64, 64), seed=123)
    rewards, lengths = [], []
    for _ in range(num_episodes):
        obs = env.reset()
        total = 0.0
        for step in range(max_steps):
            rng_key, ak = jax.random.split(rng_key)
            action = agent.act(obs, ak, training=False)
            next_obs = env.step(action)
            total += next_obs["reward"]
            if next_obs["done"]:
                break
            obs = next_obs
        rewards.append(total)
        lengths.append(step + 1)
    env.close()
    return rewards, lengths


def _eval_pytorch_agent(agent, num_episodes, max_steps, device):
    import torch
    from src.environments.crafter import CrafterEnv

    agent.eval()
    env = CrafterEnv(size=(64, 64), seed=123)
    rewards, lengths = [], []
    for _ in range(num_episodes):
        obs_dict = env.reset()
        stoch, deter = agent.rssm.initial(1)
        stoch, deter = stoch.to(device), deter.to(device)
        prev_action = torch.zeros(1, NUM_ACTIONS, device=device)
        total = 0.0
        for step in range(max_steps):
            with torch.no_grad():
                image_hwc = obs_dict["image"].transpose(1, 2, 0)
                image_t = (
                    torch.tensor(image_hwc[None], dtype=torch.float32, device=device)
                    / 255.0
                )
                is_first = torch.tensor(
                    [obs_dict["is_first"]], dtype=torch.bool, device=device
                )
                embed = agent._frozen_encoder({"image": image_t.unsqueeze(1)}).squeeze(
                    1
                )
                stoch, deter, _ = agent._frozen_rssm.obs_step(
                    stoch, deter, prev_action, embed, is_first
                )
                feat = agent._frozen_rssm.get_feat(stoch, deter)
                action = agent._frozen_actor(feat).mode
                prev_action = action
            action_int = int(action[0].argmax().cpu())
            next_obs = env.step(action_int)
            total += next_obs["reward"]
            if next_obs["done"]:
                break
            obs_dict = next_obs
        rewards.append(total)
        lengths.append(step + 1)
    env.close()
    agent.train()
    return rewards, lengths


def _run_variant_jax(transitions, all_starts, train_steps):
    import jax
    from src.r2dreamer.config import R2DreamerConfig
    from src.r2dreamer.agent import R2DreamerAgent
    from src.r2dreamer.world_model.encoders import ConvEncoder

    cfg = R2DreamerConfig(
        obs_shape=OBS_SHAPE_CHW,
        num_actions=NUM_ACTIONS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        encoder_module_cls=ConvEncoder,
    )
    rng = jax.random.PRNGKey(SEED)
    rng, init_key = jax.random.split(rng)
    agent = R2DreamerAgent(cfg, init_key)
    param_count = sum(x.size for x in jax.tree.leaves(agent.params))

    for i in range(WARMUP_STEPS):
        rng, k = jax.random.split(rng)
        batch = _convert_batch(transitions, all_starts[i])
        _ = agent.train_step(batch, k)
        jax.block_until_ready(agent.params)

    metrics_history = []
    step_times = []
    for i in range(train_steps):
        rng, k = jax.random.split(rng)
        batch = _convert_batch(transitions, all_starts[WARMUP_STEPS + i])
        t0 = time.perf_counter()
        metrics = agent.train_step(batch, k)
        jax.block_until_ready(agent.params)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        metrics_history.append(metrics)
        if (i + 1) % 500 == 0:
            print(
                f"  [JAX] step {i + 1}/{train_steps} | loss={metrics.get('total_loss', 0):.2f} | "
                f"dyn={metrics.get('loss/dyn', 0):.2f} | {(t1 - t0) * 1000:.1f} ms"
            )

    peak_mem = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    times = np.array(step_times)
    mask = times < 5 * np.median(times)

    return {
        "params": int(param_count),
        "mean_step_ms": float(np.mean(times[mask]) * 1000),
        "std_step_ms": float(np.std(times[mask]) * 1000),
        "steps_per_sec": float(train_steps / sum(step_times)),
        "peak_gpu_gb": float(peak_mem),
        "total_time_s": float(sum(step_times)),
        "metrics_history": [
            {k: float(v) for k, v in m.items()} for m in metrics_history
        ],
        "agent": agent,
    }


def _run_variant_pytorch(transitions, all_starts, train_steps, rep_loss, device):
    import torch

    sys.path.insert(0, EXT)
    from dreamer import Dreamer

    cfg = make_pytorch_config(device, rep_loss)
    obs_space, act_space = make_crafter_spaces()
    agent = Dreamer(cfg, obs_space, act_space).to(device)
    param_count = sum(p.numel() for p in agent._named_params.values())
    label = f"PT-{rep_loss}"

    for i in range(WARMUP_STEPS):
        data = make_batch_torch(transitions, all_starts[i], device)
        stoch0 = torch.zeros(
            BATCH_SIZE, cfg.rssm.stoch, cfg.rssm.discrete, device=device
        )
        deter0 = torch.zeros(BATCH_SIZE, cfg.rssm.deter, device=device)
        _ = _train_step_pytorch(agent, data, (stoch0, deter0))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    metrics_history = []
    step_times = []
    for i in range(train_steps):
        data = make_batch_torch(transitions, all_starts[WARMUP_STEPS + i], device)
        stoch0 = torch.zeros(
            BATCH_SIZE, cfg.rssm.stoch, cfg.rssm.discrete, device=device
        )
        deter0 = torch.zeros(BATCH_SIZE, cfg.rssm.deter, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mets, _, _ = _train_step_pytorch(agent, data, (stoch0, deter0))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        metrics_history.append(
            {
                k: float(v) if isinstance(v, torch.Tensor) else float(v)
                for k, v in mets.items()
            }
        )
        if (i + 1) % 500 == 0:
            loss = float(mets.get("opt/loss", 0))
            dyn = float(mets.get("loss/dyn", 0))
            print(
                f"  [{label}] step {i + 1}/{train_steps} | loss={loss:.2f} | "
                f"dyn={dyn:.2f} | {(t1 - t0) * 1000:.1f} ms"
            )

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    return {
        "params": int(param_count),
        "mean_step_ms": float(np.mean(step_times) * 1000),
        "std_step_ms": float(np.std(step_times) * 1000),
        "steps_per_sec": float(train_steps / sum(step_times)),
        "peak_gpu_gb": float(peak_mem),
        "total_time_s": float(sum(step_times)),
        "metrics_history": metrics_history,
        "agent": agent,
    }


def run(*, train_steps=4_000, eval_episodes=10, output_path=None, argv=None):
    """Run 3-way benchmark and save results JSON.

    Args:
        train_steps: number of train steps per variant.
        eval_episodes: number of episodes for policy evaluation.
        output_path: directory for output JSON; defaults to ROOT/output/methods/comparisons.
        argv: argument list (default: sys.argv[1:]).

    Returns:
        dict with per-variant timing/memory/eval stats (mirrors JSON structure).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=train_steps)
    parser.add_argument("--eval-episodes", type=int, default=eval_episodes)
    parser.add_argument(
        "--skip-pytorch", action="store_true", help="Only run JAX variant"
    )
    args = parser.parse_args(argv)

    import torch
    import jax

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"PyTorch {torch.__version__}, JAX {jax.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Config: train_steps={args.train_steps}, eval_episodes={args.eval_episodes}, seed={SEED}\n"
    )

    print("Collecting Crafter data...")
    transitions = collect_crafter_data(NUM_COLLECT, seed=SEED)
    all_starts = precompute_batch_starts(
        WARMUP_STEPS + args.train_steps, transitions, SEED
    )
    print(f"Collected {len(transitions)} transitions, {len(all_starts)} batch starts\n")

    results = {}

    print("=== R2-Dreamer (JAX) ===")
    jax_result = _run_variant_jax(transitions, all_starts, args.train_steps)
    agent_jax = jax_result.pop("agent")
    results["R2-Dreamer (JAX)"] = jax_result
    print(
        f"  Done: {jax_result['mean_step_ms']:.1f} ms/step, {jax_result['peak_gpu_gb']:.2f} GB\n"
    )

    agent_r2_pt = None
    agent_dv3_pt = None

    if not args.skip_pytorch:
        print("=== R2-Dreamer (PyTorch) ===")
        r2_result = _run_variant_pytorch(
            transitions, all_starts, args.train_steps, "r2dreamer", device
        )
        agent_r2_pt = r2_result.pop("agent")
        results["R2-Dreamer (PyTorch)"] = r2_result
        print(
            f"  Done: {r2_result['mean_step_ms']:.1f} ms/step, {r2_result['peak_gpu_gb']:.2f} GB\n"
        )

        print("=== DreamerV3 (PyTorch) ===")
        dv3_result = _run_variant_pytorch(
            transitions, all_starts, args.train_steps, "dreamer", device
        )
        agent_dv3_pt = dv3_result.pop("agent")
        results["DreamerV3 (PyTorch)"] = dv3_result
        print(
            f"  Done: {dv3_result['mean_step_ms']:.1f} ms/step, {dv3_result['peak_gpu_gb']:.2f} GB\n"
        )

    print("=== Policy Evaluation ===")
    rng_eval = jax.random.PRNGKey(123)
    jax_rewards, jax_lengths = _eval_jax_agent(
        agent_jax, args.eval_episodes, EVAL_MAX_STEPS, rng_eval
    )
    results["R2-Dreamer (JAX)"]["eval_rewards"] = jax_rewards
    results["R2-Dreamer (JAX)"]["eval_lengths"] = jax_lengths
    print(f"  JAX:    reward={np.mean(jax_rewards):.2f} +/- {np.std(jax_rewards):.2f}")

    del agent_jax
    gc.collect()

    if not args.skip_pytorch:
        r2_rewards, r2_lengths = _eval_pytorch_agent(
            agent_r2_pt, args.eval_episodes, EVAL_MAX_STEPS, device
        )
        results["R2-Dreamer (PyTorch)"]["eval_rewards"] = r2_rewards
        results["R2-Dreamer (PyTorch)"]["eval_lengths"] = r2_lengths
        print(
            f"  R2-PT:  reward={np.mean(r2_rewards):.2f} +/- {np.std(r2_rewards):.2f}"
        )

        dv3_rewards, dv3_lengths = _eval_pytorch_agent(
            agent_dv3_pt, args.eval_episodes, EVAL_MAX_STEPS, device
        )
        results["DreamerV3 (PyTorch)"]["eval_rewards"] = dv3_rewards
        results["DreamerV3 (PyTorch)"]["eval_lengths"] = dv3_lengths
        print(
            f"  DV3-PT: reward={np.mean(dv3_rewards):.2f} +/- {np.std(dv3_rewards):.2f}"
        )

        del agent_r2_pt, agent_dv3_pt
        torch.cuda.empty_cache()
        gc.collect()

    outdir = output_path or os.path.join(ROOT, "output", "comparison")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "r2dreamer_benchmark.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    print(
        f"\n{'Variant':<25} {'Params':>10} {'ms/step':>10} {'GPU (GB)':>10} {'Reward':>10}"
    )
    print("-" * 70)
    for name, r in results.items():
        rew = np.mean(r.get("eval_rewards", [0]))
        print(
            f"{name:<25} {r['params']:>10,} {r['mean_step_ms']:>8.1f}ms {r['peak_gpu_gb']:>8.2f}GB {rew:>8.2f}"
        )

    return results
