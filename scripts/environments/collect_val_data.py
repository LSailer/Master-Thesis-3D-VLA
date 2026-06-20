"""Collect random-action val episodes and save as .npz for val loss computation.

Usage:
    uv run python scripts/environments/collect_val_data.py \
        --episodes 200 --output data/val_replay/val_200ep.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.environments.habitat import build_habitat_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/val_replay/val_200ep.npz")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_geodesic", type=float, default=5.0)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--obs_size", type=int, default=64)
    args = parser.parse_args()

    env = build_habitat_env(
        obs_shape=(3, args.obs_size, args.obs_size),
        max_episode_steps=args.max_episode_steps,
        split=args.split,
        max_geodesic=args.max_geodesic,
    )

    all_obs, all_actions, all_rewards, all_dones, all_terminals = [], [], [], [], []

    for ep in range(args.episodes):
        obs = env.reset()
        ep_steps = 0
        while True:
            action = np.random.randint(0, 4)
            next_obs = env.step(action)
            all_obs.append(obs.image)  # (C, H, W) uint8
            all_actions.append(action)
            all_rewards.append(next_obs.reward)
            success = next_obs.success > 0
            all_dones.append(next_obs.done)
            all_terminals.append(success)
            ep_steps += 1
            if next_obs.done:
                break
            obs = next_obs

        if (ep + 1) % 10 == 0:
            print(f"[{ep + 1}/{args.episodes}] steps={ep_steps}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        args.output,
        obs=np.array(all_obs, dtype=np.uint8),
        actions=np.array(all_actions, dtype=np.int32),
        rewards=np.array(all_rewards, dtype=np.float32),
        dones=np.array(all_dones, dtype=np.bool_),
        terminals=np.array(all_terminals, dtype=np.bool_),
    )
    total = len(all_obs)
    print(f"Saved {total} steps from {args.episodes} episodes to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
