"""Run R2-Dreamer (JAX) on Habitat ObjectNav, output metrics to CSV + WandB."""

import argparse
import csv
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from modules.r2dreamer.agent import R2DreamerAgent
from modules.r2dreamer.config import R2DreamerConfig
from modules.dreamerv3.configs import DreamerConfig
from modules.envs.habitat import HabitatObjectNavEnv
from modules.dreamerv3.replay_buffer import ReplayBuffer


def _convert_batch(batch: dict, num_actions: int) -> dict:
    """Convert replay buffer batch to R2DreamerAgent format."""
    actions_onehot = jax.nn.one_hot(batch["actions"], num_actions)
    dones = batch["dones"]
    return {
        "obs": batch["obs"],
        "actions": actions_onehot,
        "rewards": batch["rewards"],
        "is_first": batch["is_first"],
        "is_last": dones,
        "is_terminal": dones,
    }


def _save_checkpoint(agent, step, output_dir):
    """Save agent params, optimizer state, and slow critic to disk."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:09d}.pkl")
    data = {
        "step": step,
        "params": jax.tree.map(np.array, agent.params),
        "opt_state": jax.tree.map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
            agent.opt_state,
        ),
        "slow_critic_params": jax.tree.map(np.array, agent.slow_critic_params),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--prefill", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--checkpoint_every", type=int, default=50_000)
    parser.add_argument("--wandb_project", type=str, default="3d-vla-objectnav")
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "metrics.csv")

    # --- Config ---
    config = R2DreamerConfig(
        obs_shape=(3, 64, 64),
        num_actions=4,  # STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
        total_steps=args.steps,
        prefill_steps=args.prefill,
        buffer_capacity=1_000_000,
        act_entropy=3e-2,
        seed=args.seed,
        log_every=args.log_every,
        logdir=args.output_dir,
    )

    # --- WandB ---
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(config) if hasattr(config, "__dict__") else {},
        tags=["r2dreamer", "habitat", "baseline", "10M"],
    )

    # --- Environment ---
    hab_config = DreamerConfig(
        obs_shape=(3, 64, 64),
        max_episode_steps=500,
        split="train",
        reward_type="geodesic_delta",
    )
    env = HabitatObjectNavEnv(hab_config)

    # --- Agent ---
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, init_key = jax.random.split(rng_key)
    agent = R2DreamerAgent(config, init_key)
    buffer = ReplayBuffer(config)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "metric", "value"])

        # --- Prefill ---
        print(f"Prefilling {config.prefill_steps} steps...")
        obs = env.reset()
        for i in range(config.prefill_steps):
            action = np.random.randint(1, config.num_actions)  # exclude STOP
            next_obs = env.step(action)
            buffer.add(obs["image"], action, next_obs["reward"], next_obs["done"])
            obs = next_obs if not next_obs["done"] else env.reset()

        # --- Training loop ---
        print(f"Training for {config.total_steps} steps...")
        obs = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_count = 0
        t0 = time.time()
        batch_steps = config.batch_size * config.seq_len
        train_credit = 0.0
        metrics = {}

        for step in range(config.total_steps):
            rng_key, act_key = jax.random.split(rng_key)

            action = agent.act(obs, act_key)
            next_obs = env.step(action)
            buffer.add(obs["image"], action, next_obs["reward"], next_obs["done"])
            episode_reward += next_obs["reward"]
            episode_steps += 1

            if next_obs["done"]:
                episode_count += 1
                success = next_obs.get("success", 0.0)
                spl = next_obs.get("spl", 0.0)

                # CSV
                writer.writerow([step, "episode/reward", episode_reward])
                writer.writerow([step, "episode/success", success])
                writer.writerow([step, "episode/spl", spl])
                writer.writerow([step, "episode/steps", episode_steps])
                f.flush()

                # WandB
                wandb.log({
                    "episode/reward": episode_reward,
                    "episode/success": success,
                    "episode/spl": spl,
                    "episode/steps": episode_steps,
                    "episode/count": episode_count,
                }, step=step)

                print(
                    f"[step {step:>8d}] episode {episode_count}: "
                    f"reward={episode_reward:.2f} success={success:.0f} "
                    f"spl={spl:.3f} steps={episode_steps}"
                )

                episode_reward = 0.0
                episode_steps = 0
                obs = env.reset()
            else:
                obs = next_obs

            # --- Train ---
            if buffer.size >= batch_steps:
                train_credit += config.train_ratio / batch_steps
                while train_credit >= 1.0:
                    rng_key, train_key = jax.random.split(rng_key)
                    batch = buffer.sample(config.batch_size, config.seq_len)
                    batch = _convert_batch(batch, config.num_actions)
                    metrics = agent.train_step(batch, train_key)
                    train_credit -= 1.0

                if step % config.log_every == 0 and metrics:
                    for k, v in metrics.items():
                        writer.writerow([step, k, v])
                    f.flush()

                    wandb.log(metrics, step=step)

                    elapsed = time.time() - t0
                    fps = (step + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"[step {step:>8d}/{config.total_steps}] "
                        f"total={metrics.get('total_loss', 0):.3f} "
                        f"dyn={metrics.get('loss/dyn', 0):.3f} "
                        f"rew={metrics.get('loss/rew', 0):.3f} "
                        f"policy={metrics.get('loss/policy', 0):.3f} "
                        f"fps={fps:.0f}"
                    )

            # --- Checkpoint ---
            if (step + 1) % args.checkpoint_every == 0:
                _save_checkpoint(agent, step + 1, args.output_dir)

    _save_checkpoint(agent, config.total_steps, args.output_dir)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Episodes: {episode_count}. Output: {csv_path}")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
