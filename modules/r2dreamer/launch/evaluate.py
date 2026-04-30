"""Public evaluate() entry point for the r2dreamer launcher."""

from __future__ import annotations

import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from modules.r2dreamer.launch.parser import _build_parser_eval
from modules.r2dreamer.launch.registries import env_registry
from modules.r2dreamer.launch.curricula import CURRICULA
from modules.r2dreamer.agent import R2DreamerAgent
from modules.r2dreamer.config import R2DreamerConfig
from modules.envs.habitat import sample_navmesh


def _render_topdown(env, trajectory, goal_positions, output_path):
    """Render a top-down map with navmesh, trajectory, and goal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nav = sample_navmesh(env._env, resolution=0.1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    extent = [nav["x_min"], nav["x_max"], nav["z_max"], nav["z_min"]]
    ax.imshow(nav["grid"], extent=extent, cmap="Greys_r", alpha=0.3)

    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 2], "b-", linewidth=1.5, alpha=0.7)
    ax.plot(traj[0, 0], traj[0, 2], "go", markersize=10, label="Start")
    ax.plot(traj[-1, 0], traj[-1, 2], "rs", markersize=10, label="End")

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
    r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    euler = r.as_euler("yxz")
    return float(euler[0])


def evaluate(
    *,
    env: str,
    encoder: str,
    curriculum: str | None = None,
    checkpoint: str | None = None,
    output_dir: str | None = None,
    argv: list[str] | None = None,
) -> dict:
    """Resolve (env, encoder, curriculum) via registries; parse CLI; run eval loop.

    Kwargs (checkpoint, output_dir) are shim-supplied defaults — CLI flags override.

    Returns metrics dict with 'results' and 'meta' keys.
    """
    from modules.r2dreamer.launch.registries import encoder_registry
    from modules.shared.configs import DreamerConfig
    from modules.envs.habitat import HabitatObjectNavEnv

    parser = _build_parser_eval()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # CLI --encoder overrides shim kwarg if user passed it explicitly.
    eff_encoder = args.encoder if args.encoder is not None else encoder
    if eff_encoder not in encoder_registry:
        raise KeyError(f"Unknown encoder {eff_encoder!r}. Available: {list(encoder_registry)}")

    # --- Resolve checkpoint ---
    eff_checkpoint = args.checkpoint if args.checkpoint is not None else checkpoint
    if not args.random and eff_checkpoint is None:
        raise ValueError(
            "checkpoint must be set via evaluate(..., checkpoint=...) or --checkpoint"
        )

    # --- Resolve curriculum path ---
    if args.curriculum_path is not None:
        curriculum_path = args.curriculum_path
    elif curriculum is not None:
        if curriculum not in CURRICULA:
            raise KeyError(f"Unknown curriculum {curriculum!r}. Available: {list(CURRICULA)}")
        curriculum_path = str(CURRICULA[curriculum])
    else:
        curriculum_path = None

    # --- Resolve output dir ---
    eff_output_dir = args.output_dir if args.output_dir is not None else output_dir
    if eff_output_dir is None:
        raise ValueError("output_dir must be set via evaluate(..., output_dir=...) or --output_dir")

    os.makedirs(eff_output_dir, exist_ok=True)
    output_path = os.path.join(eff_output_dir, "eval_results.json")

    # --- Build env ---
    if env not in env_registry:
        raise KeyError(f"Unknown env {env!r}. Available: {list(env_registry)}")

    if eff_encoder == "vggt":
        render_resolution = args.render_resolution if args.render_resolution is not None else 518
    else:
        render_resolution = args.render_resolution if args.render_resolution is not None else 64
    hab_config = DreamerConfig(
        obs_shape=(3, render_resolution, render_resolution),
        max_episode_steps=500,
        split=args.split,
        reward_type="geodesic_delta",
    )
    env_instance = HabitatObjectNavEnv(
        hab_config,
        semantic=args.semantic,
        curriculum_path=curriculum_path,
        curriculum_mode="eval",
    )

    # --- Build encoder + adapter ---
    encoder_cls = encoder_registry[eff_encoder]
    if eff_encoder == "vggt":
        enc = encoder_cls(resolution=render_resolution)
    else:
        enc = encoder_cls()
    adapter = enc.make_adapter()

    # --- Build agent ---
    if eff_encoder == "vggt":
        from modules.r2dreamer.adapters import VGGT_FEATURE_DIM
        config = R2DreamerConfig(
            encoder_type="vggt",
            obs_shape=(VGGT_FEATURE_DIM,),
            num_actions=4,
        )
    else:
        config = R2DreamerConfig(obs_shape=(3, 64, 64), num_actions=4)
    rng_key = jax.random.PRNGKey(args.seed)

    if args.random:
        agent = None
        print("Using random agent")
    else:
        with open(eff_checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        print(f"Loaded checkpoint from step {ckpt['step']}")
        rng_key, init_key = jax.random.split(rng_key)
        agent = R2DreamerAgent(config, init_key)
        agent.params = jax.tree.map(jnp.array, ckpt["params"])
        agent.slow_critic_params = jax.tree.map(jnp.array, ckpt["slow_critic_params"])

    # --- Evaluate ---
    ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    results = []

    for ep_idx in range(args.episodes):
        obs = env_instance.reset()
        if adapter.on_episode_reset:
            adapter.on_episode_reset()
        _, agent_obs = adapter.transform(obs)
        actions_taken = []
        rewards = []
        trajectory = []
        headings = []

        start_pos = env_instance._env.sim.get_agent_state().position.tolist()
        goal_positions = []
        for goal in env_instance._env.current_episode.goals:
            if goal.view_points:
                for vp in goal.view_points:
                    pos = vp.agent_state.position
                    goal_positions.append(
                        pos.tolist() if hasattr(pos, "tolist") else list(pos))
                    break
            else:
                pos = goal.position
                goal_positions.append(
                    pos.tolist() if hasattr(pos, "tolist") else list(pos))
        scene_id = env_instance._env.current_episode.scene_id
        object_category = env_instance._env.current_episode.object_category

        trajectory.append(start_pos)
        headings.append(_get_agent_heading(env_instance))

        for _step in range(500):
            if agent is not None:
                rng_key, act_key = jax.random.split(rng_key)
                action = agent.act(agent_obs, act_key, training=False)
            else:
                action = np.random.randint(0, config.num_actions)

            next_obs = env_instance.step(action)
            _, next_agent_obs = adapter.transform(next_obs)
            actions_taken.append(int(action))
            rewards.append(float(next_obs["reward"]))

            pos = env_instance._env.sim.get_agent_state().position.tolist()
            trajectory.append(pos)
            headings.append(_get_agent_heading(env_instance))

            if next_obs["done"]:
                obs = next_obs
                break
            obs = next_obs
            agent_obs = next_agent_obs

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

        if args.render_topdown:
            topdown_dir = os.path.join(eff_output_dir, "topdown")
            os.makedirs(topdown_dir, exist_ok=True)
            topdown_path = os.path.join(topdown_dir, f"episode_{ep_idx:03d}.png")
            _render_topdown(env_instance, trajectory, goal_positions, topdown_path)

        print(
            f"Episode {ep_idx}: steps={len(actions_taken):3d}  "
            f"reward={sum(rewards):.2f}  "
            f"success={obs.get('success', 0):.0f}  "
            f"category={object_category}"
        )

    print(f"\n--- Summary ({args.episodes} episodes) ---")
    print(f"Success: {np.mean([r['success'] for r in results]) * 100:.1f}%")
    print(f"SPL: {np.mean([r['spl'] for r in results]):.3f}")
    print(f"Mean reward: {np.mean([r['reward'] for r in results]):.2f}")
    print(f"Mean steps: {np.mean([r['steps'] for r in results]):.0f}")

    meta = {"agent": "random" if args.random else eff_checkpoint}
    output = {"meta": meta, "results": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")

    env_instance.close()
    return output
