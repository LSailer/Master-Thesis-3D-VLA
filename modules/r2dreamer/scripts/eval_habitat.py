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
from scipy.spatial.transform import Rotation

from modules.r2dreamer.agent import R2DreamerAgent
from modules.r2dreamer.config import R2DreamerConfig
from modules.dreamerv3.configs import DreamerConfig
from modules.envs.habitat import HabitatObjectNavEnv


def _get_agent_heading(env):
    """Extract agent heading (yaw in radians) from habitat sim state."""
    state = env._env.sim.get_agent_state()
    quat = state.rotation
    # habitat quaternion is (w, x, y, z) but scipy expects (x, y, z, w)
    r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    euler = r.as_euler("yxz")
    return float(euler[0])  # yaw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--random", action="store_true",
                        help="Use random agent instead of checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_frames", action="store_true",
                        help="Save RGB frames as numpy arrays (large files)")
    args = parser.parse_args()

    if not args.random and args.checkpoint is None:
        parser.error("--checkpoint is required unless --random is set")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # --- Agent ---
    config = R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=4)
    rng_key = jax.random.PRNGKey(42)

    if args.random:
        agent = None
        print("Using random agent")
    else:
        with open(args.checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        print(f"Loaded checkpoint from step {ckpt['step']}")
        rng_key, init_key = jax.random.split(rng_key)
        agent = R2DreamerAgent(config, init_key)
        agent.params = jax.tree.map(jnp.array, ckpt["params"])
        agent.slow_critic_params = jax.tree.map(jnp.array, ckpt["slow_critic_params"])

    # --- Environment ---
    hab_config = DreamerConfig(
        obs_shape=(3, 64, 64),
        max_episode_steps=500,
        split="val",
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
        trajectory = []
        headings = []

        # Record start position and goal
        start_pos = env._env.sim.get_agent_state().position.tolist()
        goal_positions = []
        for goal in env._env.current_episode.goals:
            if goal.view_points:
                for vp in goal.view_points:
                    goal_positions.append(vp.agent_state.position.tolist())
                    break  # first viewpoint per goal
            else:
                goal_positions.append(goal.position.tolist())
        scene_id = env._env.current_episode.scene_id
        object_category = env._env.current_episode.object_category

        # Record initial position
        trajectory.append(start_pos)
        headings.append(_get_agent_heading(env))

        for step in range(500):
            if agent is not None:
                rng_key, act_key = jax.random.split(rng_key)
                action = agent.act(obs, act_key, training=False)
            else:
                action = np.random.randint(0, config.num_actions)

            next_obs = env.step(action)
            actions_taken.append(int(action))
            rewards.append(float(next_obs["reward"]))

            # Record position after step
            pos = env._env.sim.get_agent_state().position.tolist()
            trajectory.append(pos)
            headings.append(_get_agent_heading(env))

            if next_obs["done"]:
                obs = next_obs
                break
            obs = next_obs

        ep_result = {
            "episode": ep_idx,
            "scene_id": scene_id,
            "object_category": object_category,
            "steps": len(actions_taken),
            "reward": sum(rewards),
            "success": float(obs.get("success", 0.0)),
            "spl": float(obs.get("spl", 0.0)),
            "actions": actions_taken,
            "action_counts": {
                name: actions_taken.count(idx)
                for idx, name in ACTIONS.items()
            },
            "start_position": start_pos,
            "goal_positions": goal_positions,
            "trajectory": trajectory,
            "headings": headings,
        }
        results.append(ep_result)

        print(
            f"Episode {ep_idx}: steps={len(actions_taken):3d}  "
            f"reward={sum(rewards):.2f}  "
            f"success={obs.get('success', 0):.0f}  "
            f"category={object_category}"
        )

    # --- Summary ---
    print(f"\n--- Summary ({args.episodes} episodes) ---")
    print(f"Success: {np.mean([r['success'] for r in results])*100:.1f}%")
    print(f"SPL: {np.mean([r['spl'] for r in results]):.3f}")
    print(f"Mean reward: {np.mean([r['reward'] for r in results]):.2f}")
    print(f"Mean steps: {np.mean([r['steps'] for r in results]):.0f}")

    # --- Save ---
    meta = {"agent": "random" if args.random else args.checkpoint}
    with open(args.output, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"Results saved to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
