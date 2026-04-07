"""100k-step Crafter training: JAX R2-Dreamer vs PyTorch R2-Dreamer.

Same data, same batch order. Saves per-step metrics to JSON for the
parity report notebook.

Usage:
    uv run python modules/r2dreamer/scripts/run_parity_training.py [--train-steps 100000]
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

SEED = 42
WARMUP_STEPS = 3
BATCH_SIZE = 16
SEQ_LEN = 64
NUM_ACTIONS = 17
OBS_SHAPE_CHW = (3, 64, 64)
OBS_SHAPE_HWC = (64, 64, 3)
NUM_COLLECT = 20_000
LOG_EVERY = 100


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


def train_step_pytorch(agent, data, initial):
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


def run_jax(transitions, all_starts, train_steps, outpath):
    import jax
    import jax.numpy as jnp
    from modules.r2dreamer.config import R2DreamerConfig
    from modules.r2dreamer.agent import R2DreamerAgent

    cfg = R2DreamerConfig(
        obs_shape=OBS_SHAPE_CHW, num_actions=NUM_ACTIONS,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
    )
    rng = jax.random.PRNGKey(SEED)
    rng, init_key = jax.random.split(rng)
    agent = R2DreamerAgent(cfg, init_key)

    # Warmup JIT
    for i in range(WARMUP_STEPS):
        rng, k = jax.random.split(rng)
        batch = make_batch_jax(transitions, all_starts[i])
        _ = agent.train_step(batch, k)
        jax.block_until_ready(agent.params)

    # Train
    rows = []
    t_start = time.perf_counter()
    for i in range(train_steps):
        rng, k = jax.random.split(rng)
        batch = make_batch_jax(transitions, all_starts[WARMUP_STEPS + i])
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


def run_pytorch(transitions, all_starts, train_steps, outpath, device):
    import torch
    from dreamer import Dreamer

    cfg = make_pytorch_config(device)
    obs_space, act_space = make_crafter_spaces()
    agent = Dreamer(cfg, obs_space, act_space).to(device)

    # Warmup
    for i in range(WARMUP_STEPS):
        data = make_batch_torch(transitions, all_starts[i], device)
        stoch0 = torch.zeros(BATCH_SIZE, cfg.rssm.stoch, cfg.rssm.discrete, device=device)
        deter0 = torch.zeros(BATCH_SIZE, cfg.rssm.deter, device=device)
        _ = train_step_pytorch(agent, data, (stoch0, deter0))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Train
    rows = []
    t_start = time.perf_counter()
    for i in range(train_steps):
        data = make_batch_torch(transitions, all_starts[WARMUP_STEPS + i], device)
        stoch0 = torch.zeros(BATCH_SIZE, cfg.rssm.stoch, cfg.rssm.discrete, device=device)
        deter0 = torch.zeros(BATCH_SIZE, cfg.rssm.deter, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mets, _, _ = train_step_pytorch(agent, data, (stoch0, deter0))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--jax-only", action="store_true")
    parser.add_argument("--pytorch-only", action="store_true")
    args = parser.parse_args()

    import torch
    import jax
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"PyTorch {torch.__version__}, JAX {jax.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: train_steps={args.train_steps}, seed={SEED}\n")

    # Collect data
    print("Collecting Crafter data...")
    transitions = collect_crafter_data(NUM_COLLECT, seed=SEED)
    total_steps = WARMUP_STEPS + args.train_steps
    rng = np.random.RandomState(SEED)
    max_start = len(transitions) - SEQ_LEN
    all_starts = [rng.randint(0, max_start, size=BATCH_SIZE) for _ in range(total_steps)]
    print(f"Collected {len(transitions)} transitions, {len(all_starts)} batch starts\n")

    outdir = os.path.join(ROOT, "output", "parity")
    os.makedirs(outdir, exist_ok=True)

    if not args.pytorch_only:
        print("=== R2-Dreamer (JAX) ===")
        run_jax(transitions, all_starts, args.train_steps,
                os.path.join(outdir, "jax_metrics.json"))
        print()

    if not args.jax_only:
        print("=== R2-Dreamer (PyTorch) ===")
        run_pytorch(transitions, all_starts, args.train_steps,
                    os.path.join(outdir, "pytorch_metrics.json"), device)
        print()

    print("Done! Results in output/parity/")


if __name__ == "__main__":
    main()
