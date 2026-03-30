"""3-way R2-Dreamer benchmark: JAX R2 vs PyTorch R2 vs PyTorch DreamerV3.

Trains each variant for TRAIN_STEPS on identical Crafter replay data,
measures timing/memory/losses, evaluates policies, and saves all results
to output/comparison/r2dreamer_benchmark.json.

Usage:
    uv run python modules/r2dreamer/scripts/run_benchmark.py [--train-steps 4000] [--eval-episodes 10]
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXT = os.path.join(ROOT, "external", "r2dreamer")
sys.path.insert(0, ROOT)
sys.path.insert(0, EXT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
WARMUP_STEPS = 3
BATCH_SIZE = 16
SEQ_LEN = 64
NUM_ACTIONS = 17
OBS_SHAPE_CHW = (3, 64, 64)
OBS_SHAPE_HWC = (64, 64, 3)
NUM_COLLECT = 10_000
EVAL_MAX_STEPS = 500

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_crafter_data(num_steps, seed=42):
    from modules.envs.crafter import CrafterEnv
    env = CrafterEnv(size=(64, 64), seed=seed)
    transitions = []
    obs = env.reset()
    for _ in range(num_steps):
        action = np.random.randint(0, NUM_ACTIONS)
        next_obs = env.step(action)
        transitions.append({
            "image_chw": obs["image"].copy(),
            "image_hwc": obs["image"].transpose(1, 2, 0).copy(),
            "action": action,
            "reward": next_obs["reward"],
            "is_first": obs["is_first"],
            "is_last": next_obs["done"],
            "is_terminal": next_obs["done"],
        })
        obs = env.reset() if next_obs["done"] else next_obs
    env.close()
    return transitions


def precompute_batch_starts(num_steps, transitions, seed):
    rng = np.random.RandomState(seed)
    max_start = len(transitions) - SEQ_LEN
    return [rng.randint(0, max_start, size=BATCH_SIZE) for _ in range(num_steps)]


# ---------------------------------------------------------------------------
# Batch builders
# ---------------------------------------------------------------------------


def make_batch_jax(transitions, starts):
    import jax.numpy as jnp
    B, T = len(starts), SEQ_LEN
    obs = np.zeros((B, T, *OBS_SHAPE_CHW), dtype=np.float32)
    actions = np.zeros((B, T, NUM_ACTIONS), dtype=np.float32)
    rewards = np.zeros((B, T), dtype=np.float32)
    is_first = np.zeros((B, T), dtype=np.float32)
    is_last = np.zeros((B, T), dtype=np.float32)
    is_terminal = np.zeros((B, T), dtype=np.float32)
    for i, s in enumerate(starts):
        for t in range(T):
            tr = transitions[s + t]
            obs[i, t] = tr["image_chw"].astype(np.float32) / 255.0
            actions[i, t, tr["action"]] = 1.0
            rewards[i, t] = tr["reward"]
            is_first[i, t] = float(tr["is_first"])
            is_last[i, t] = float(tr["is_last"])
            is_terminal[i, t] = float(tr["is_terminal"])
    return {
        "obs": jnp.array(obs), "actions": jnp.array(actions),
        "rewards": jnp.array(rewards), "is_first": jnp.array(is_first),
        "is_last": jnp.array(is_last), "is_terminal": jnp.array(is_terminal),
    }


def make_batch_torch(transitions, starts, device="cuda"):
    import torch
    from tensordict import TensorDict
    B, T = len(starts), SEQ_LEN
    obs = np.zeros((B, T, *OBS_SHAPE_HWC), dtype=np.uint8)
    actions = np.zeros((B, T, NUM_ACTIONS), dtype=np.float32)
    rewards = np.zeros((B, T, 1), dtype=np.float32)
    is_first = np.zeros((B, T, 1), dtype=np.float32)
    is_last = np.zeros((B, T, 1), dtype=np.float32)
    is_terminal = np.zeros((B, T, 1), dtype=np.float32)
    for i, s in enumerate(starts):
        for t in range(T):
            tr = transitions[s + t]
            obs[i, t] = tr["image_hwc"]
            actions[i, t, tr["action"]] = 1.0
            rewards[i, t, 0] = tr["reward"]
            is_first[i, t, 0] = float(tr["is_first"])
            is_last[i, t, 0] = float(tr["is_last"])
            is_terminal[i, t, 0] = float(tr["is_terminal"])
    return TensorDict({
        "image": torch.tensor(obs, device=device),
        "action": torch.tensor(actions, device=device),
        "reward": torch.tensor(rewards, device=device),
        "is_first": torch.tensor(is_first, dtype=torch.bool, device=device),
        "is_last": torch.tensor(is_last, dtype=torch.bool, device=device),
        "is_terminal": torch.tensor(is_terminal, dtype=torch.bool, device=device),
    }, batch_size=(B, T))


# ---------------------------------------------------------------------------
# PyTorch config (resolved, no Hydra interpolations)
# ---------------------------------------------------------------------------


def make_pytorch_config(device, rep_loss="r2dreamer"):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "act_entropy": 3e-4, "kl_free": 1.0, "imag_horizon": 15, "horizon": 333,
        "lamb": 0.95, "compile": False, "log_grads": False, "device": device,
        "rep_loss": rep_loss,
        "lr": 4e-5, "agc": 0.3, "pmin": 1e-3, "eps": 1e-20,
        "beta1": 0.9, "beta2": 0.999, "warmup": 1000,
        "slow_target_update": 1, "slow_target_fraction": 0.02,
        "loss_scales": {
            "barlow": 0.05, "infonce": 1.0, "recon": 1.0, "rew": 1.0,
            "con": 1.0, "dyn": 1.0, "rep": 0.1, "policy": 1.0,
            "value": 1.0, "repval": 0.3, "swav": 1.0, "temp": 1.0, "norm": 1.0,
        },
        "r2dreamer": {"lambd": 5e-4},
        "rssm": {
            "stoch": 32, "deter": 2048, "hidden": 256, "discrete": 16,
            "img_layers": 2, "obs_layers": 1, "dyn_layers": 1, "blocks": 8,
            "act": "SiLU", "norm": True, "unimix_ratio": 0.01,
            "initial": "learned", "device": device,
        },
        "encoder": {
            "mlp_keys": "$^", "cnn_keys": "image",
            "mlp": {"shape": None, "layers": 3, "units": 256, "act": "SiLU",
                    "norm": True, "device": device, "outscale": None,
                    "symlog_inputs": True, "name": "mlp_encoder"},
            "cnn": {"act": "SiLU", "norm": True, "kernel_size": 5,
                    "minres": 4, "depth": 16, "mults": [2, 3, 4, 4]},
        },
        "decoder": {
            "mlp_keys": "$^", "cnn_keys": "image",
            "mlp_dist": {"name": "symlog_mse"}, "cnn_dist": {"name": "mse"},
            "mlp": {"shape": None, "layers": 3, "units": 256, "act": "SiLU",
                    "norm": True, "dist": {"name": "identity"}, "device": device,
                    "outscale": 1.0, "symlog_inputs": False, "name": "mlp_decoder"},
            "cnn": {"depth": 16, "units": 256, "bspace": 8, "mults": [2, 3, 4, 4],
                    "act": "SiLU", "norm": True, "kernel_size": 5, "minres": 4,
                    "outscale": 1.0},
        },
        "reward": {"shape": [255], "layers": 1, "units": 256, "act": "SiLU",
                   "norm": True, "dist": {"name": "symexp_twohot", "bin_num": 255},
                   "outscale": 0.0, "device": device, "symlog_inputs": False, "name": "reward"},
        "cont": {"shape": [1], "layers": 1, "units": 256, "act": "SiLU",
                 "norm": True, "dist": {"name": "binary"},
                 "outscale": 1.0, "device": device, "symlog_inputs": False, "name": "cont"},
        "actor": {"shape": None, "layers": 3, "units": 256, "act": "SiLU",
                  "norm": True, "device": device,
                  "dist": {"cont": {"name": "bounded_normal", "min_std": 0.1, "max_std": 1.0},
                           "disc": {"name": "onehot", "unimix_ratio": 0.01},
                           "multi_disc": {"name": "multi_onehot", "unimix_ratio": 0.01}},
                  "outscale": 0.01, "symlog_inputs": False, "name": "actor"},
        "critic": {"shape": [255], "layers": 3, "units": 256, "act": "SiLU",
                   "norm": True, "device": device,
                   "dist": {"name": "symexp_twohot", "bin_num": 255},
                   "outscale": 0.0, "symlog_inputs": False, "name": "value"},
    })


def make_crafter_spaces():
    import gymnasium as gym
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, OBS_SHAPE_HWC, dtype=np.uint8),
    })
    act_space = gym.spaces.Box(low=0, high=1, shape=(NUM_ACTIONS,), dtype=np.float32)
    act_space.discrete = True
    return obs_space, act_space


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_step_pytorch(agent, data, initial):
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_jax_agent(agent, num_episodes, max_steps, rng_key):
    import jax
    from modules.envs.crafter import CrafterEnv
    env = CrafterEnv(size=(64, 64), seed=123)
    rewards, lengths = [], []
    for ep in range(num_episodes):
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


def eval_pytorch_agent(agent, num_episodes, max_steps, device):
    import torch
    from modules.envs.crafter import CrafterEnv
    agent.eval()
    env = CrafterEnv(size=(64, 64), seed=123)
    rewards, lengths = [], []
    for ep in range(num_episodes):
        obs_dict = env.reset()
        stoch, deter = agent.rssm.initial(1)
        stoch, deter = stoch.to(device), deter.to(device)
        prev_action = torch.zeros(1, NUM_ACTIONS, device=device)
        total = 0.0
        for step in range(max_steps):
            with torch.no_grad():
                image_hwc = obs_dict["image"].transpose(1, 2, 0)
                image_t = torch.tensor(image_hwc[None], dtype=torch.float32, device=device) / 255.0
                is_first = torch.tensor([obs_dict["is_first"]], dtype=torch.bool, device=device)
                embed = agent._frozen_encoder({"image": image_t.unsqueeze(1)}).squeeze(1)
                stoch, deter, _ = agent._frozen_rssm.obs_step(stoch, deter, prev_action, embed, is_first)
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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_variant_jax(transitions, all_starts, train_steps):
    import jax
    import jax.numpy as jnp
    from modules.r2dreamer.config import R2DreamerConfig
    from modules.r2dreamer.agent import R2DreamerAgent

    cfg = R2DreamerConfig(obs_shape=OBS_SHAPE_CHW, num_actions=NUM_ACTIONS,
                          batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    rng = jax.random.PRNGKey(SEED)
    rng, init_key = jax.random.split(rng)
    agent = R2DreamerAgent(cfg, init_key)
    param_count = sum(x.size for x in jax.tree.leaves(agent.params))

    # Warmup
    for i in range(WARMUP_STEPS):
        rng, k = jax.random.split(rng)
        batch = make_batch_jax(transitions, all_starts[i])
        _ = agent.train_step(batch, k)
        jax.block_until_ready(agent.params)

    # Train
    metrics_history = []
    step_times = []
    for i in range(train_steps):
        rng, k = jax.random.split(rng)
        batch = make_batch_jax(transitions, all_starts[WARMUP_STEPS + i])
        t0 = time.perf_counter()
        metrics = agent.train_step(batch, k)
        jax.block_until_ready(agent.params)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        metrics_history.append(metrics)
        if (i + 1) % 500 == 0:
            print(f"  [JAX] step {i+1}/{train_steps} | loss={metrics.get('total_loss',0):.2f} | "
                  f"dyn={metrics.get('loss/dyn',0):.2f} | {(t1-t0)*1000:.1f} ms")

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
        "metrics_history": [{k: float(v) for k, v in m.items()} for m in metrics_history],
        "agent": agent,
    }


def run_variant_pytorch(transitions, all_starts, train_steps, rep_loss, device):
    import torch
    from dreamer import Dreamer

    cfg = make_pytorch_config(device, rep_loss)
    obs_space, act_space = make_crafter_spaces()
    agent = Dreamer(cfg, obs_space, act_space).to(device)
    param_count = sum(p.numel() for p in agent._named_params.values())
    label = f"PT-{rep_loss}"

    # Warmup
    for i in range(WARMUP_STEPS):
        data = make_batch_torch(transitions, all_starts[i], device)
        stoch0 = torch.zeros(BATCH_SIZE, cfg.rssm.stoch, cfg.rssm.discrete, device=device)
        deter0 = torch.zeros(BATCH_SIZE, cfg.rssm.deter, device=device)
        _ = train_step_pytorch(agent, data, (stoch0, deter0))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Train
    metrics_history = []
    step_times = []
    for i in range(train_steps):
        data = make_batch_torch(transitions, all_starts[WARMUP_STEPS + i], device)
        stoch0 = torch.zeros(BATCH_SIZE, cfg.rssm.stoch, cfg.rssm.discrete, device=device)
        deter0 = torch.zeros(BATCH_SIZE, cfg.rssm.deter, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mets, _, _ = train_step_pytorch(agent, data, (stoch0, deter0))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        metrics_history.append({k: float(v) if isinstance(v, torch.Tensor) else float(v)
                                for k, v in mets.items()})
        if (i + 1) % 500 == 0:
            loss = float(mets.get("opt/loss", 0))
            dyn = float(mets.get("loss/dyn", 0))
            print(f"  [{label}] step {i+1}/{train_steps} | loss={loss:.2f} | "
                  f"dyn={dyn:.2f} | {(t1-t0)*1000:.1f} ms")

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--skip-pytorch", action="store_true", help="Only run JAX variant")
    args = parser.parse_args()

    import torch
    import jax
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"PyTorch {torch.__version__}, JAX {jax.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: train_steps={args.train_steps}, eval_episodes={args.eval_episodes}, seed={SEED}\n")

    # Collect data
    print("Collecting Crafter data...")
    transitions = collect_crafter_data(NUM_COLLECT, seed=SEED)
    all_starts = precompute_batch_starts(WARMUP_STEPS + args.train_steps, transitions, SEED)
    print(f"Collected {len(transitions)} transitions, {len(all_starts)} batch starts\n")

    results = {}

    # JAX R2-Dreamer
    print("=== R2-Dreamer (JAX) ===")
    jax_result = run_variant_jax(transitions, all_starts, args.train_steps)
    agent_jax = jax_result.pop("agent")
    results["R2-Dreamer (JAX)"] = jax_result
    print(f"  Done: {jax_result['mean_step_ms']:.1f} ms/step, {jax_result['peak_gpu_gb']:.2f} GB\n")

    if not args.skip_pytorch:
        # PyTorch R2-Dreamer
        print("=== R2-Dreamer (PyTorch) ===")
        r2_result = run_variant_pytorch(transitions, all_starts, args.train_steps, "r2dreamer", device)
        agent_r2_pt = r2_result.pop("agent")
        results["R2-Dreamer (PyTorch)"] = r2_result
        print(f"  Done: {r2_result['mean_step_ms']:.1f} ms/step, {r2_result['peak_gpu_gb']:.2f} GB\n")

        # PyTorch DreamerV3
        print("=== DreamerV3 (PyTorch) ===")
        dv3_result = run_variant_pytorch(transitions, all_starts, args.train_steps, "dreamer", device)
        agent_dv3_pt = dv3_result.pop("agent")
        results["DreamerV3 (PyTorch)"] = dv3_result
        print(f"  Done: {dv3_result['mean_step_ms']:.1f} ms/step, {dv3_result['peak_gpu_gb']:.2f} GB\n")

    # Evaluation
    print("=== Policy Evaluation ===")
    rng_eval = jax.random.PRNGKey(123)
    jax_rewards, jax_lengths = eval_jax_agent(agent_jax, args.eval_episodes, EVAL_MAX_STEPS, rng_eval)
    results["R2-Dreamer (JAX)"]["eval_rewards"] = jax_rewards
    results["R2-Dreamer (JAX)"]["eval_lengths"] = jax_lengths
    print(f"  JAX:    reward={np.mean(jax_rewards):.2f} +/- {np.std(jax_rewards):.2f}")

    del agent_jax
    gc.collect()

    if not args.skip_pytorch:
        r2_rewards, r2_lengths = eval_pytorch_agent(agent_r2_pt, args.eval_episodes, EVAL_MAX_STEPS, device)
        results["R2-Dreamer (PyTorch)"]["eval_rewards"] = r2_rewards
        results["R2-Dreamer (PyTorch)"]["eval_lengths"] = r2_lengths
        print(f"  R2-PT:  reward={np.mean(r2_rewards):.2f} +/- {np.std(r2_rewards):.2f}")

        dv3_rewards, dv3_lengths = eval_pytorch_agent(agent_dv3_pt, args.eval_episodes, EVAL_MAX_STEPS, device)
        results["DreamerV3 (PyTorch)"]["eval_rewards"] = dv3_rewards
        results["DreamerV3 (PyTorch)"]["eval_lengths"] = dv3_lengths
        print(f"  DV3-PT: reward={np.mean(dv3_rewards):.2f} +/- {np.std(dv3_rewards):.2f}")

        del agent_r2_pt, agent_dv3_pt
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    outdir = os.path.join(ROOT, "output", "comparison")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "r2dreamer_benchmark.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    # Summary
    print(f"\n{'Variant':<25} {'Params':>10} {'ms/step':>10} {'GPU (GB)':>10} {'Reward':>10}")
    print("-" * 70)
    for name, r in results.items():
        rew = np.mean(r.get("eval_rewards", [0]))
        print(f"{name:<25} {r['params']:>10,} {r['mean_step_ms']:>8.1f}ms {r['peak_gpu_gb']:>8.2f}GB {rew:>8.2f}")


if __name__ == "__main__":
    main()
