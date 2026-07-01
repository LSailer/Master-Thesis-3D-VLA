"""Per-step instrumented L1 VGGT R2Dreamer eval — dumps npz artifacts.

Forks src.r2dreamer.launch.evaluate.evaluate() but adds per-step
artifact dumping for downstream visualization. Captures RGB, raw VGGT
outputs (world_points + camera_pose), RSSM latents (stoch, deter, feat),
agent pose, action, reward, and is_first/is_episode_end flags.

Designed to be run with --episodes N to roll 0..N-1 sequentially with
the same seed=42 as evaluate.py, dumping npz only for episode indices
listed in --dump_episodes.

Usage:
    uv run python scripts/debug_viz/evaluate_debug.py \\
        --checkpoint <path/to/step_000300000.pkl> \\
        --curriculum_path data/curriculum/level1_1house_1goal.json \\
        --episodes 12 --dump_episodes 7,11 \\
        --output_dir output/runs/.../debug/viz-pair-a/
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

# Make src.* importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from src.configs.config import R2DreamerConfig
from src.environments.habitat import build_habitat_env
from src.r2dreamer.adapters import VGGT_FEATURE_DIM
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.launch.registries import encoder_registry
from src.r2dreamer.observation_preparation.vggt_readouts import (
    flatten_world_points_camera_pose,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--random", action="store_true", help="Use random agent instead of a checkpoint"
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=12,
        help="Roll episodes 0..N-1 sequentially with --seed",
    )
    p.add_argument(
        "--dump_episodes",
        type=str,
        default=None,
        help="CSV of episode indices to dump npz for; default = all",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--curriculum_path", type=str, default=None)
    p.add_argument("--render_resolution", type=int, default=518)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--semantic", action="store_true")
    return p


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _agent_state_arrays(env_instance):
    """Return (position[3], rotation[w,x,y,z]) as float32 numpy arrays."""
    state = env_instance._env.sim.get_agent_state()
    pos = np.asarray(state.position, dtype=np.float32)
    q = state.rotation
    rot = np.asarray([q.w, q.x, q.y, q.z], dtype=np.float32)
    return pos, rot


def main(argv: list[str] | None = None) -> dict:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # --- resolve curriculum path ---
    if args.curriculum_path is not None:
        curriculum_path = args.curriculum_path
    else:
        raise ValueError("--curriculum_path is required")

    # --- resolve checkpoint ---
    if not args.random and args.checkpoint is None:
        raise ValueError("--checkpoint is required (or use --random)")

    # --- resolve dump_episodes ---
    if args.dump_episodes is None or args.dump_episodes == "":
        dump_set = set(range(args.episodes))
    else:
        dump_set = {int(x) for x in args.dump_episodes.split(",") if x.strip() != ""}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_resolution = args.render_resolution

    # --- env ---
    env_instance = build_habitat_env(
        obs_shape=(3, render_resolution, render_resolution),
        split=args.split,
        semantic=args.semantic,
        curriculum_path=curriculum_path,
        curriculum_mode="eval",
    )

    # --- encoder + adapter (VGGT) ---
    encoder_cls = encoder_registry["vggt"]
    enc = encoder_cls(resolution=render_resolution)
    adapter = enc.make_adapter()
    extractor = adapter._extractor  # un-flattened access

    # --- agent ---
    config = R2DreamerConfig(
        encoder_type="vggt",
        encoder_module_cls=enc.spec().module_cls,
        obs_shape=(VGGT_FEATURE_DIM,),
        num_actions=4,
    )
    rng_key = jax.random.PRNGKey(args.seed)

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

    # --- eval loop ---
    ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    results = []
    start_ts = time.time()
    total_dump_bytes = 0
    total_dump_steps = 0

    for ep_idx in range(args.episodes):
        dump_this = ep_idx in dump_set
        ep_dir = output_dir / f"episode_{ep_idx:03d}"
        if dump_this:
            ep_dir.mkdir(parents=True, exist_ok=True)

        obs = env_instance.reset()
        if adapter.on_episode_reset:
            adapter.on_episode_reset()

        # First-frame VGGT extract — reuse the dict for both agent input and dump.
        rgb0 = obs["image"]
        vggt_out0 = extractor.extract(rgb0)
        feat_vec0 = flatten_world_points_camera_pose(vggt_out0)
        agent_obs = {"features": feat_vec0, "is_first": obs.get("is_first", True)}

        actions_taken: list[int] = []
        rewards: list[float] = []
        start_pos = env_instance._env.sim.get_agent_state().position.tolist()
        try:
            habitat_episode_id = env_instance._env.current_episode.episode_id
        except Exception:
            habitat_episode_id = None

        # Initial pose / rotation for META.
        ep_start_pos_arr, ep_start_rot_arr = _agent_state_arrays(env_instance)
        start_rotation = ep_start_rot_arr.tolist()

        goal_positions: list[list[float]] = []
        for goal in env_instance._env.current_episode.goals:
            if goal.view_points:
                for vp in goal.view_points:
                    pos = vp.agent_state.position
                    goal_positions.append(
                        pos.tolist() if hasattr(pos, "tolist") else list(pos)
                    )
                    break
            else:
                pos = goal.position
                goal_positions.append(
                    pos.tolist() if hasattr(pos, "tolist") else list(pos)
                )
        scene_id = env_instance._env.current_episode.scene_id
        object_category = env_instance._env.current_episode.object_category

        # Per-step buffer for cur-step values BEFORE we know reward/done.
        cur_rgb = rgb0
        cur_vggt = vggt_out0
        cur_is_first = True

        for step in range(500):
            # --- act ---
            if agent is not None:
                rng_key, act_key = jax.random.split(rng_key)
                action = agent.act(
                    agent_obs, agent_obs["is_first"], act_key, training=False
                )
                # Capture RSSM latents AFTER act() — they're updated in-place.
                act_state = agent.snapshot_act_state()
                stoch = np.asarray(act_state.stoch[0]).astype(np.float32)
                deter = np.asarray(act_state.deter[0]).astype(np.float32)
            else:
                action = int(np.random.randint(0, config.num_actions))
                stoch = np.zeros(
                    (config.stoch_classes, config.stoch_discrete), dtype=np.float32
                )
                deter = np.zeros((config.deter_size,), dtype=np.float32)
            feat = np.concatenate([stoch.reshape(-1), deter]).astype(np.float32)

            # --- agent pose for THIS step (pre-step state matches the obs) ---
            cur_pos, cur_rot = _agent_state_arrays(env_instance)

            # --- env step ---
            next_obs = env_instance.step(action)
            actions_taken.append(int(action))
            rewards.append(float(next_obs["reward"]))
            is_episode_end = bool(next_obs["is_episode_end"])

            # --- dump per-step npz ---
            if dump_this:
                npz_path = ep_dir / f"step_{step:03d}.npz"
                np.savez_compressed(
                    npz_path,
                    rgb=cur_rgb.astype(np.uint8),
                    world_points=cur_vggt["world_points"].astype(np.float32),
                    camera_pose=cur_vggt["camera_pose"].astype(np.float32),
                    stoch=stoch,
                    deter=deter,
                    feat=feat,
                    agent_position=cur_pos,
                    agent_rotation=cur_rot,
                    action=np.int32(action),
                    reward=np.float32(next_obs["reward"]),
                    is_first=np.bool_(cur_is_first),
                    is_episode_end=np.bool_(is_episode_end),
                )
                total_dump_bytes += npz_path.stat().st_size
                total_dump_steps += 1

            if is_episode_end:
                obs = next_obs
                break

            # --- prep next step ---
            cur_rgb = next_obs["image"]
            cur_vggt = extractor.extract(cur_rgb)
            feat_vec_next = flatten_world_points_camera_pose(cur_vggt)
            agent_obs = {
                "features": feat_vec_next,
                "is_first": next_obs.get("is_first", False),
            }
            cur_is_first = False
            obs = next_obs

        steps = len(actions_taken)
        ep_result = {
            "episode": ep_idx,
            "scene_id": scene_id,
            "object_category": object_category,
            "steps": steps,
            "reward": sum(rewards),
            "success": float(obs.get("success", 0.0)),
            "spl": float(obs.get("spl", 0.0)),
            "actions": actions_taken,
            "action_counts": {
                name: actions_taken.count(idx) for idx, name in ACTIONS.items()
            },
            "start_position": start_pos,
            "goal_positions": goal_positions,
        }
        results.append(ep_result)

        # --- per-episode META.json ---
        if dump_this:
            meta = {
                "episode_idx": ep_idx,
                "scene_id": scene_id,
                "object_category": object_category,
                "habitat_episode_id": habitat_episode_id,
                "start_position": start_pos,
                "start_rotation": start_rotation,
                "goal_positions": goal_positions,
                "success": int(ep_result["success"]),
                "spl": float(ep_result["spl"]),
                "steps": steps,
                "reward": float(ep_result["reward"]),
                "actions": actions_taken,
                "rewards": [float(r) for r in rewards],
                "checkpoint": args.checkpoint if not args.random else "random",
                "git_sha": _git_sha(),
                "seed": args.seed,
            }
            with open(ep_dir / "META.json", "w") as f:
                json.dump(meta, f, indent=2)

        print(
            f"Episode {ep_idx}: steps={steps:3d}  "
            f"reward={sum(rewards):.2f}  "
            f"success={obs.get('success', 0):.0f}  "
            f"category={object_category}  "
            f"dumped={dump_this}"
        )

    end_ts = time.time()

    # --- run-level MANIFEST ---
    bytes_per_step_avg = (
        total_dump_bytes / total_dump_steps if total_dump_steps > 0 else 0
    )
    manifest = {
        "purpose": "Per-step instrumented L1 VGGT R2Dreamer re-roll for debug viz",
        "checkpoint": args.checkpoint if not args.random else "random",
        "git_sha": _git_sha(),
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "start_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_ts)),
        "end_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_ts)),
        "wall_seconds": int(end_ts - start_ts),
        "curriculum_path": curriculum_path,
        "num_episodes_requested": args.episodes,
        "num_episodes_completed": len(results),
        "encoder": "vggt",
        "render_resolution": render_resolution,
        "split": "val (overridden by curriculum_mode='eval' inside HabitatObjectNavEnv)",
        "dumped_episodes": sorted(dump_set),
        "bytes_per_step_avg": int(bytes_per_step_avg),
        "total_dump_bytes": int(total_dump_bytes),
        "total_dump_steps": total_dump_steps,
        "summary_metrics": {
            "success_rate": float(np.mean([r["success"] for r in results])),
            "spl_mean": float(np.mean([r["spl"] for r in results])),
            "reward_mean": float(np.mean([r["reward"] for r in results])),
            "steps_mean": float(np.mean([r["steps"] for r in results])),
        },
    }
    with open(output_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Side-by-side eval_results.json (keeps shape parity with evaluate.py).
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(
            {
                "meta": {"agent": args.checkpoint if not args.random else "random"},
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n--- Summary ({args.episodes} episodes) ---")
    print(f"Success: {manifest['summary_metrics']['success_rate'] * 100:.1f}%")
    print(f"SPL: {manifest['summary_metrics']['spl_mean']:.3f}")
    print(f"Mean reward: {manifest['summary_metrics']['reward_mean']:.2f}")
    print(f"Mean steps: {manifest['summary_metrics']['steps_mean']:.0f}")
    print(f"Dumped {total_dump_steps} steps over {len(dump_set)} episode(s)")
    print(
        f"Total dump bytes: {total_dump_bytes / 1e6:.1f} MB "
        f"({bytes_per_step_avg / 1024:.1f} KB/step avg)"
    )
    print(f"Manifest: {output_dir / 'MANIFEST.json'}")

    env_instance.close()
    return manifest


if __name__ == "__main__":
    main()
