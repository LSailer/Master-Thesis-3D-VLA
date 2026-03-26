"""DreamerV3 deterministic evaluation on HM3D ObjectNav."""

import argparse
import dataclasses
import json
import os

import jax
import numpy as np

from .agent import DreamerAgent
from .configs import DreamerConfig


def evaluate(config: DreamerConfig, checkpoint: str, max_geodesic: float | None,
             num_episodes: int) -> list[dict]:
    from .env_habitat import HabitatObjectNavEnv
    env = HabitatObjectNavEnv(config, max_geodesic=max_geodesic)

    rng_key = jax.random.PRNGKey(config.seed)
    agent = DreamerAgent(config, rng_key)
    agent.load(checkpoint)

    results = []
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward, steps = 0.0, 0
        rng_key, act_key = jax.random.split(rng_key)

        while not obs.get("done", False) and steps < config.max_episode_steps:
            action = agent.act(obs, act_key, training=False)
            obs = env.step(action)
            ep_reward += obs["reward"]
            steps += 1

        results.append({
            "episode": ep,
            "reward": ep_reward,
            "success": float(obs.get("success", 0)),
            "spl": float(obs.get("spl", 0)),
            "steps": steps,
            "category": getattr(env._env.current_episode, "object_category", ""),
        })
        sr_so_far = np.mean([r["success"] for r in results])
        print(f"  ep {ep+1}/{num_episodes}: reward={ep_reward:.2f} "
              f"success={results[-1]['success']:.0f} steps={steps} "
              f"(running SR={sr_so_far:.3f})")

    env.close()

    sr = np.mean([r["success"] for r in results])
    spl = np.mean([r["spl"] for r in results])
    mean_r = np.mean([r["reward"] for r in results])
    print(f"\n{'='*50}")
    print(f"SR={sr:.3f}  SPL={spl:.3f}  mean_reward={mean_r:.2f}  "
          f"({num_episodes} episodes)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_geodesic", type=float, default=None)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--obs_size", type=int, default=64)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = DreamerConfig(
        obs_shape=(3, args.obs_size, args.obs_size),
        max_episode_steps=args.max_episode_steps,
        split=args.split,
        seed=args.seed,
    )

    results = evaluate(config, args.checkpoint, args.max_geodesic, args.num_episodes)

    out_path = args.output or os.path.join(args.checkpoint, "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
