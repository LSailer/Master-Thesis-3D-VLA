"""Evaluate R2-Dreamer checkpoint on Habitat ObjectNav, save results to JSON.

Supports optional semantic sensor and top-down map rendering for analysis.
"""

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
from modules.shared.configs import DreamerConfig
from modules.envs.habitat import HabitatObjectNavEnv, sample_navmesh


def _render_topdown(env, trajectory, goal_positions, output_path):
    """Render a top-down map with navmesh, trajectory, and goal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nav = sample_navmesh(env._env, resolution=0.1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Navigable area
    extent = [nav["x_min"], nav["x_max"], nav["z_max"], nav["z_min"]]
    ax.imshow(nav["grid"], extent=extent, cmap="Greys_r", alpha=0.3)

    # Trajectory (x, z from 3D positions)
    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 2], "b-", linewidth=1.5, alpha=0.7)
    ax.plot(traj[0, 0], traj[0, 2], "go", markersize=10, label="Start")
    ax.plot(traj[-1, 0], traj[-1, 2], "rs", markersize=10, label="End")

    # Goal positions
    for i, gp in enumerate(goal_positions):
        ax.plot(gp[0], gp[2], "m*", markersize=15,
                label="Goal" if i == 0 else None)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(os.path.basename(output_path).replace(".png", ""))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split (train, val, val_mini)")
    parser.add_argument("--semantic", action="store_true",
                        help="Enable semantic mesh loading")
    parser.add_argument("--render_topdown", action="store_true",
                        help="Render top-down maps with trajectory overlay")
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
        split=args.split,
        reward_type="geodesic_delta",
    )
    env = HabitatObjectNavEnv(hab_config, semantic=args.semantic)

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
                    pos = vp.agent_state.position
                    goal_positions.append(
                        pos.tolist() if hasattr(pos, "tolist") else list(pos))
                    break  # first viewpoint per goal
            else:
                pos = goal.position
                goal_positions.append(
                    pos.tolist() if hasattr(pos, "tolist") else list(pos))
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

        # Render top-down map
        if args.render_topdown:
            topdown_dir = os.path.join(
                os.path.dirname(args.output) or ".", "topdown")
            os.makedirs(topdown_dir, exist_ok=True)
            topdown_path = os.path.join(
                topdown_dir, f"episode_{ep_idx:03d}.png")
            _render_topdown(env, trajectory, goal_positions, topdown_path)

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
