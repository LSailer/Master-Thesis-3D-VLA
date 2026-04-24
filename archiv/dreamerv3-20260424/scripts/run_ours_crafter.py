"""Run our DreamerV3 on Crafter, output metrics to CSV."""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import jax
import numpy as np

from modules.dreamerv3.agent import DreamerAgent
from modules.dreamerv3.configs import DreamerConfig
from modules.envs.crafter import CrafterEnv
from modules.dreamerv3.replay_buffer import ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--prefill", type=int, default=5000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=250)
    args = parser.parse_args()

    config = DreamerConfig(
        obs_shape=(3, 64, 64),
        num_actions=17,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        seed=args.seed,
        log_every=args.log_every,
    )

    env = CrafterEnv(size=(64, 64), seed=args.seed)
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, init_key = jax.random.split(rng_key)
    agent = DreamerAgent(config, init_key)
    buffer = ReplayBuffer(config)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "metric", "value"])

        # Prefill with random actions
        print(f"Prefilling {config.prefill_steps} steps...")
        obs = env.reset()
        for i in range(config.prefill_steps):
            action = np.random.randint(0, config.num_actions)
            next_obs = env.step(action)
            buffer.add(obs["image"], action, next_obs["reward"], next_obs["done"])
            obs = next_obs if not next_obs["done"] else env.reset()

        # Training loop
        print(f"Training for {config.total_steps} steps...")
        obs = env.reset()
        episode_reward = 0.0
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

            if next_obs["done"]:
                episode_count += 1
                writer.writerow([step, "episode/score", episode_reward])
                episode_reward = 0.0
                obs = env.reset()
            else:
                obs = next_obs

            # Train (with train ratio)
            if buffer.size >= batch_steps:
                train_credit += config.train_ratio / batch_steps
                while train_credit >= 1.0:
                    rng_key, train_key = jax.random.split(rng_key)
                    batch = buffer.sample(config.batch_size, config.seq_len)
                    metrics = agent.train_step(batch, train_key)
                    train_credit -= 1.0

                if step % config.log_every == 0 and metrics:
                    for k, v in metrics.items():
                        writer.writerow([step, k, v])
                    f.flush()

                    elapsed = time.time() - t0
                    fps = (step + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"[step {step:>6d}/{config.total_steps}] "
                        f"wm={metrics.get('wm_loss', 0):.3f} "
                        f"dyn={metrics.get('loss/dyn', 0):.3f} "
                        f"img={metrics.get('loss/image', 0):.3f} "
                        f"rew={metrics.get('loss/rew', 0):.3f} "
                        f"fps={fps:.0f}"
                    )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Episodes: {episode_count}. Output: {args.output}")


if __name__ == "__main__":
    main()
