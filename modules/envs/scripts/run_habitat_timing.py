"""Habitat timing benchmark: measure training step throughput for 3 model variants.

Uses SYNTHETIC data to isolate model training speed from environment speed.
Outputs JSON to output/comparison/habitat_timing.json.

Variants:
  r2dreamer_jax      -- JAX R2DreamerAgent (always runs)
  r2dreamer_pytorch  -- PyTorch r2dreamer with rep_loss=r2dreamer
  dreamerv3_pytorch  -- PyTorch r2dreamer with rep_loss=dreamer
"""

import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO_ROOT)
R2DREAMER_DIR = os.path.join(REPO_ROOT, "external", "r2dreamer")

# Shared benchmark parameters
OBS_SHAPE = (3, 64, 64)   # CHW (JAX) / will be transposed to HWC for PyTorch
NUM_ACTIONS = 4
BATCH_SIZE = 16
SEQ_LEN = 64
WARMUP_STEPS = 3
DEFAULT_STEPS = 30


# ---------------------------------------------------------------------------
# JAX R2-Dreamer benchmark
# ---------------------------------------------------------------------------

def benchmark_jax_r2dreamer(steps=DEFAULT_STEPS, obs_shape=OBS_SHAPE,
                             num_actions=NUM_ACTIONS, batch_size=BATCH_SIZE,
                             seq_len=SEQ_LEN):
    """Benchmark JAX R2-Dreamer."""
    import jax
    import jax.numpy as jnp
    from modules.r2dreamer.agent import R2DreamerAgent
    from modules.r2dreamer.config import R2DreamerConfig

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    cfg = R2DreamerConfig(
        obs_shape=obs_shape,
        num_actions=num_actions,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    agent = R2DreamerAgent(cfg, init_key)

    def make_batch():
        return {
            "obs": jnp.array(
                np.random.rand(batch_size, seq_len, *obs_shape).astype(np.float32)
            ),
            "actions": jnp.array(
                np.eye(num_actions, dtype=np.float32)[
                    np.random.randint(0, num_actions, (batch_size, seq_len))
                ]
            ),
            "rewards": jnp.array(
                np.random.randn(batch_size, seq_len).astype(np.float32)
            ),
            "is_first": jnp.array(
                np.zeros((batch_size, seq_len), dtype=np.float32)
            ),
            "is_last": jnp.array(
                np.zeros((batch_size, seq_len), dtype=np.float32)
            ),
            "is_terminal": jnp.array(
                np.zeros((batch_size, seq_len), dtype=np.float32)
            ),
        }

    # Warmup (JIT compilation)
    print(f"Warming up JAX R2-Dreamer ({WARMUP_STEPS} steps)...")
    for _ in range(WARMUP_STEPS):
        rng_key, train_key = jax.random.split(rng_key)
        agent.train_step(make_batch(), train_key)
    print("Warmup done.")

    # Timed benchmark
    print(f"Benchmarking JAX R2-Dreamer ({steps} steps)...")
    step_times = []
    for _ in range(steps):
        rng_key, train_key = jax.random.split(rng_key)
        batch = make_batch()
        jax.block_until_ready(agent.params)
        t0 = time.perf_counter()
        metrics = agent.train_step(batch, train_key)
        jax.block_until_ready(metrics)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    # Peak GPU memory
    try:
        mem_bytes = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
        peak_gpu_gb = mem_bytes / 1e9
    except Exception:
        peak_gpu_gb = float("nan")

    mean_t = float(np.mean(step_times))
    std_t = float(np.std(step_times))
    print(f"JAX R2-Dreamer: {mean_t:.4f} +/- {std_t:.4f} s/step  "
          f"({1/mean_t:.1f} steps/s)  peak GPU: {peak_gpu_gb:.2f} GB")

    return {
        "mean_step_time": mean_t,
        "std_step_time": std_t,
        "steps_per_sec": 1.0 / mean_t,
        "peak_gpu_gb": peak_gpu_gb,
    }


# ---------------------------------------------------------------------------
# PyTorch r2dreamer benchmark
# ---------------------------------------------------------------------------

def benchmark_pytorch_r2dreamer(steps=DEFAULT_STEPS, obs_shape=OBS_SHAPE,
                                 num_actions=NUM_ACTIONS, batch_size=BATCH_SIZE,
                                 seq_len=SEQ_LEN, rep_loss="r2dreamer"):
    """Benchmark PyTorch r2dreamer (rep_loss: 'r2dreamer' or 'dreamer')."""
    import torch
    from torch.amp import autocast

    sys.path.insert(0, R2DREAMER_DIR)
    from omegaconf import OmegaConf
    import gym
    from dreamer import Dreamer

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device: {device_str}")

    # Load the full config via Hydra compose (resolves all interpolations)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    config_dir = os.path.join(R2DREAMER_DIR, "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="configs",
            overrides=[
                "env=crafter",
                "model=size12M",
                f"model.rep_loss={rep_loss}",
                f"device={device_str}",
                "model.compile=False",
            ],
        )
    cfg = cfg.model

    # Obs/act spaces (HWC for PyTorch r2dreamer)
    H, W, C = obs_shape[1], obs_shape[2], obs_shape[0]
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (H, W, C), dtype=np.uint8)
    })
    act_space = gym.spaces.Discrete(num_actions)

    agent = Dreamer(cfg, obs_space, act_space).to(device_str)

    def make_batch():
        """Return a TensorDict-style batch for _cal_grad."""
        from tensordict import TensorDict
        dev = torch.device(device_str)
        # image: (B, T, H, W, C) uint8
        image = torch.from_numpy(
            np.random.randint(0, 256, (batch_size, seq_len, H, W, C), dtype=np.uint8)
        ).to(dev)
        action = torch.from_numpy(
            np.eye(num_actions, dtype=np.float32)[
                np.random.randint(0, num_actions, (batch_size, seq_len))
            ]
        ).to(dev)
        reward = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32, device=dev)
        is_first = torch.zeros(batch_size, seq_len, 1, dtype=torch.bool, device=dev)
        is_last = torch.zeros(batch_size, seq_len, 1, dtype=torch.bool, device=dev)
        is_terminal = torch.zeros(batch_size, seq_len, 1, dtype=torch.bool, device=dev)

        td = TensorDict(
            {
                "image": image,
                "action": action,
                "reward": reward,
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
            },
            batch_size=(batch_size, seq_len),
            device=dev,
        )
        return td

    def make_initial():
        """Return the initial RSSM state."""
        return agent.get_initial_state(batch_size)

    def one_step(data, initial):
        p_data = agent.preprocess(data)
        agent._update_slow_target()
        with autocast(device_type=torch.device(device_str).type, dtype=torch.float16):
            (stoch, deter), _ = agent._cal_grad(p_data, initial)
        agent._scaler.unscale_(agent._optimizer)
        agent._agc(agent._named_params.values())
        agent._scaler.step(agent._optimizer)
        agent._scaler.update()
        agent._scheduler.step()
        agent._optimizer.zero_grad(set_to_none=True)
        return stoch, deter

    # Warmup
    print(f"Warming up PyTorch r2dreamer/{rep_loss} ({WARMUP_STEPS} steps)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP_STEPS):
        data = make_batch()
        initial = (make_initial()["stoch"], make_initial()["deter"])
        one_step(data, initial)
    print("Warmup done.")

    # Timed benchmark
    print(f"Benchmarking PyTorch r2dreamer/{rep_loss} ({steps} steps)...")
    step_times = []
    for _ in range(steps):
        data = make_batch()
        initial = (make_initial()["stoch"], make_initial()["deter"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step(data, initial)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    if torch.cuda.is_available():
        peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_gpu_gb = float("nan")

    mean_t = float(np.mean(step_times))
    std_t = float(np.std(step_times))
    print(f"PyTorch r2dreamer/{rep_loss}: {mean_t:.4f} +/- {std_t:.4f} s/step  "
          f"({1/mean_t:.1f} steps/s)  peak GPU: {peak_gpu_gb:.2f} GB")

    return {
        "mean_step_time": mean_t,
        "std_step_time": std_t,
        "steps_per_sec": 1.0 / mean_t,
        "peak_gpu_gb": peak_gpu_gb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Habitat model timing benchmark")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Number of timed training steps per variant")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--output", type=str,
                        default=os.path.join(REPO_ROOT, "output", "comparison",
                                             "habitat_timing.json"))
    args = parser.parse_args()

    results = {}

    # JAX benchmark (always runs)
    print("\n" + "=" * 60)
    print("Variant 1/3: JAX R2-Dreamer")
    print("=" * 60)
    results["r2dreamer_jax"] = benchmark_jax_r2dreamer(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # PyTorch benchmarks (skip if deps missing)
    for variant, rep_loss in [
        ("r2dreamer_pytorch", "r2dreamer"),
        ("dreamerv3_pytorch", "dreamer"),
    ]:
        print("\n" + "=" * 60)
        print(f"Variant: PyTorch r2dreamer (rep_loss={rep_loss})")
        print("=" * 60)
        try:
            results[variant] = benchmark_pytorch_r2dreamer(
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                rep_loss=rep_loss,
            )
        except Exception as e:
            print(f"PyTorch benchmark ({rep_loss}) skipped: {e}")
            results[variant] = {"error": str(e)}

    # Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Variant':<28} {'mean (s)':>10} {'std (s)':>8} {'steps/s':>9} {'GPU GB':>7}")
    print("-" * 65)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<28}  (skipped: {r['error'][:30]})")
        else:
            print(f"{name:<28} {r['mean_step_time']:>10.4f} {r['std_step_time']:>8.4f}"
                  f" {r['steps_per_sec']:>9.1f} {r['peak_gpu_gb']:>7.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
