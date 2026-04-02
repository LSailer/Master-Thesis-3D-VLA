"""Evaluate R2-Dreamer checkpoint on Habitat ObjectNav, save results to JSON."""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import jax
import jax.numpy as jnp
import numpy as np

from modules.r2dreamer.agent import R2DreamerAgent
from modules.r2dreamer.config import R2DreamerConfig
from modules.dreamerv3.configs import DreamerConfig
from modules.envs.habitat import HabitatObjectNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_frames", action="store_true",
                        help="Save RGB frames as numpy arrays (large files)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # --- Load checkpoint ---
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    print(f"Loaded checkpoint from step {ckpt['step']}")

    # --- Agent ---
    config = R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=4)
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    agent = R2DreamerAgent(config, init_key)
    agent.params = jax.tree.map(jnp.array, ckpt["params"])
    agent.slow_critic_params = jax.tree.map(jnp.array, ckpt["slow_critic_params"])

    # --- Environment ---
    hab_config = DreamerConfig(
        obs_shape=(3, 64, 64),
        max_episode_steps=500,
        split="train",
        reward_type="geodesic_delta",
    )
    env = HabitatObjectNavEnv(hab_config)

    # --- Evaluate ---
    ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    results = []

    for ep_idx in range(args.episodes):
        obs = env.reset()
        actions_taken = []
        rewards = []

        for step in range(500):
            rng_key, act_key = jax.random.split(rng_key)
            action = agent.act(obs, act_key, training=False)
            next_obs = env.step(action)
            actions_taken.append(int(action))
            rewards.append(float(next_obs["reward"]))

            if next_obs["done"]:
                obs = next_obs
                break
            obs = next_obs

        ep_result = {
            "episode": ep_idx,
            "steps": len(actions_taken),
            "reward": sum(rewards),
            "success": float(obs.get("success", 0.0)),
            "spl": float(obs.get("spl", 0.0)),
            "actions": actions_taken,
            "action_counts": {
                name: actions_taken.count(idx)
                for idx, name in ACTIONS.items()
            },
        }
        results.append(ep_result)

        print(
            f"Episode {ep_idx}: steps={len(actions_taken):3d}  "
            f"reward={sum(rewards):.2f}  "
            f"success={obs.get('success', 0):.0f}"
        )

    # --- Summary ---
    print(f"\n--- Summary ({args.episodes} episodes) ---")
    print(f"Success: {np.mean([r['success'] for r in results])*100:.1f}%")
    print(f"Mean reward: {np.mean([r['reward'] for r in results]):.2f}")
    print(f"Mean steps: {np.mean([r['steps'] for r in results]):.0f}")

    # --- Save ---
    with open(args.output, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "results": results}, f, indent=2)
    print(f"Results saved to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
