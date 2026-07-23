"""Drive the live hybrid house-points pipeline with a trained Dreamer actor.

Successor of ``prototyp/live_vggt/run_house_visualization.py``: the same
PERSIST_SCENE VGGT stream accumulating a cumulative, voxel-deduped house
cloud — but the walker is the R2Dreamer actor restored from a training
checkpoint instead of ``RandomAgent`` (``--random`` restores the old
behaviour for plumbing smoke tests).

Unlike the live_vggt script there is no host-side point accumulation here:
the ``vggt_hybrid_house_points_pose`` encoder's adapter
(:class:`VGGTHybridHousePointsPoseObsAdapter`) already feeds every frame's
confident points into a per-scene :class:`HouseContextPoseBuffer` that
voxel-dedups on device, and the same fixed-size snapshot it hands to the
actor is the "cumulated downsampled points" input. At the end the buffers
are exported as binary PLY via ``buffer.save()``.

A MANIFEST.json records ok/failed status — habitat GL teardown poisons
SLURM exit codes on this cluster, so the manifest is the ground truth for
run success.

Run on a GPU node:
    .venv/bin/python prototyp/hybrid_actor_run/run_hybrid_actor.py \
        --checkpoint outputs/<run>/checkpoints/<step> --episodes 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

# Host-only I/O (video frames, JSON results); everything numeric on the
# device stays inside the adapter/agent JAX graphs.
import numpy as np
from PIL import Image as PILImage

from src.baselines.random_agent import RandomAgent
from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.launch.evaluate import _load_arch_overrides_from_manifest
from src.r2dreamer.launch.registries import encoder_registry
from src.shared.video_utils import write_frames_mp4

_ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
_HOUSE_ENCODERS = ("vggt_hybrid_house_points_pose", "vggt_house_points_pose")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the hybrid actor-driven collection run.

    Returns:
        The parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "R2Dreamer checkpoint directory (as written by training). "
            "Architecture overrides are read from the MANIFEST.json next to "
            "it. Required unless --random is set."
        ),
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="Use RandomAgent instead of the checkpoint actor (smoke test).",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default="vggt_hybrid_house_points_pose",
        choices=_HOUSE_ENCODERS,
        help=(
            "House-points encoder driving the adapter pipeline; must match "
            "the checkpoint's encoder_type."
        ),
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=5,
        help=(
            "Eval episodes to run. Under the single-house L1 curriculum all "
            "episodes share one scene, so the house buffer and the VGGT "
            "world frame (PERSIST_SCENE) keep accumulating across episodes."
        ),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Step cap per episode (also the env's max_episode_steps).",
    )
    p.add_argument("--seed", type=int, default=42, help="Env + actor RNG seed.")
    p.add_argument("--curriculum", type=str, default="L1")
    p.add_argument(
        "--pointcloud-dump-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated env steps at which to snapshot the house buffer "
            "as PLY under <run>/pointcloud_dumps/ (final export always runs)."
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/prototype/hybrid_actor_run",
        help="Output root; a per-run subdirectory is created inside.",
    )
    p.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Write observations.mp4 of the walked frames (0 disables).",
    )
    return p.parse_args()



def run_episode(
    *,
    ep_idx: int,
    args: argparse.Namespace,
    env,
    adapter,
    agent,
    rng_key,
    video_frames: list[np.ndarray] | None,
    out_dir: Path,
) -> tuple[dict, jax.Array]:
    """Walk one episode with the actor while the adapter grows the cloud.

    Args:
      ep_idx: Zero-based episode index (used for logging/results).
      args: Parsed CLI namespace.
      env: Habitat env instance.
      adapter: House-points obs adapter (feeds the buffer inside
        ``prepare_env_step``).
      agent: RandomAgent or R2DreamerAgent.
      rng_key: JAX PRNG key threaded through actor sampling.
      video_frames: Host-side frame sink, or None when video is disabled.
      out_dir: Per-run output directory (first frame dump on episode 0).

    Returns:
      A ``(episode_result, rng_key)`` tuple with summary metrics and the
      advanced PRNG key.
    """
    obs = env.reset()
    if adapter.on_episode_reset:
        # Scene-aware reset: saves/restores the per-scene VGGT cache
        # (PERSIST_SCENE) — never wipes the house buffer. Same contract as
        # the trainer and evaluate.py.
        adapter.on_episode_reset(getattr(obs, "scene_id", None) or "scene")
    prepared = adapter.prepare_env_step(obs)
    encoder_obs, is_first = prepared.encoder_obs, prepared.is_first
    if ep_idx == 0:
        PILImage.fromarray(np.asarray(obs.image)).save(out_dir / "first_frame.png")

    is_random = isinstance(agent, RandomAgent)
    act_state = None if is_random else agent.initial_act_state()
    actions_taken: list[int] = []
    rewards: list[float] = []
    t0 = time.perf_counter()

    for _step in range(args.max_steps):
        if video_frames is not None:
            video_frames.append(np.asarray(obs.image))
        if is_random:
            next_obs = agent.act()
            action = next_obs.previous_action
            if action is None:
                raise RuntimeError(
                    "random-agent step observation is missing previous_action"
                )
        else:
            rng_key, act_key = jax.random.split(rng_key)
            action, act_state = agent.act_with_state(
                encoder_obs, is_first, act_state, act_key, training=False
            )
            next_obs = env.step(action)
        prepared = adapter.prepare_env_step(next_obs)
        actions_taken.append(int(action))
        rewards.append(float(next_obs.reward))
        obs = next_obs
        encoder_obs, is_first = prepared.encoder_obs, prepared.is_first
        if next_obs.done:
            break

    steps = len(actions_taken)
    elapsed = time.perf_counter() - t0
    result = {
        "episode": ep_idx,
        "scene_id": obs.scene_id,
        "steps": steps,
        "reward": sum(rewards),
        "success": float(obs.success),
        "spl": float(obs.spl),
        "action_counts": {
            name: actions_taken.count(idx) for idx, name in _ACTIONS.items()
        },
        "seconds": elapsed,
    }
    print(
        f"[hybrid-actor] episode {ep_idx + 1}/{args.episodes}: "
        f"steps={steps:3d} reward={sum(rewards):.2f} "
        f"success={float(obs.success):.0f} spl={float(obs.spl):.3f} "
        f"({elapsed / max(steps, 1):.2f}s/step)",
        flush=True,
    )
    return result, rng_key


def export_outputs(
    adapter, results: list[dict], video_frames: list[np.ndarray] | None,
    args: argparse.Namespace, out_dir: Path,
) -> None:
    """Write the final house cloud(s), run results, and optional video.

    Args:
      adapter: House-points obs adapter holding the per-scene buffers.
      results: Per-episode summary dicts.
      video_frames: Collected observation frames, or None.
      args: Parsed CLI namespace.
      out_dir: Per-run output directory.
    """
    # Final cumulative cloud: the buffer content IS already voxel-deduped
    # (default 0.01 VGGT units), so save() is the downsampled export.
    # _buffers is adapter-private; acceptable in a throwaway prototype.
    cloud_root = out_dir / "house_cloud"
    for buffer in adapter._buffers.values():
        if buffer.point_count > 0:
            scene_dir = buffer.save(cloud_root)
            print(
                f"[hybrid-actor] house cloud: {buffer.point_count} points "
                f"-> {scene_dir}",
                flush=True,
            )
    summary = {
        "results": results,
        "house_buffer": adapter.diagnostics(),
        "growth_history": adapter.growth_history,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    if results:
        print(
            f"[hybrid-actor] summary: "
            f"success={np.mean([r['success'] for r in results]) * 100:.1f}% "
            f"spl={np.mean([r['spl'] for r in results]):.3f} "
            f"mean_steps={np.mean([r['steps'] for r in results]):.0f}",
            flush=True,
        )
    if video_frames:
        video_path = write_frames_mp4(
            video_frames, out_dir / "observations.mp4", fps=args.video_fps
        )
        print(f"[hybrid-actor] observation video: {video_path}", flush=True)


def collect(args: argparse.Namespace, out_dir: Path) -> None:
    """Build env/encoder/agent, run all episodes, export cloud + results.

    Args:
      args: Parsed CLI namespace from :func:`parse_args`.
      out_dir: Per-run output directory (already created).
    """
    env = None

    t_env = time.perf_counter()
    env = HabitatObjectNavEnv(
        HabitatEnvConfig(
            obs_shape=(518, 518, 3),
            max_episode_steps=args.max_steps,
            reward_type="geodesic_delta",
            curriculum=args.curriculum,
            mode="eval",
        ),
        seed=args.seed,
    )
    print(
        f"[hybrid-actor] env ready in {time.perf_counter() - t_env:.1f}s",
        flush=True,
    )


    print(
        "[hybrid-actor] building encoder + VGGT extractor "
        "(loads weights; first extract is slow)...",
    flush=True,
    )
    encoder = encoder_registry[args.encoder](
        resolution=518,
        pointcloud_dump_steps=dump_steps,
        pointcloud_dump_dir=(
            str(out_dir / "pointcloud_dumps") if dump_steps else None
        ),
    )
    adapter = encoder.make_adapter()
    agent = build_agent(args, env, encoder.spec())

    rng_key = jax.random.PRNGKey(args.seed)
    if not isinstance(agent, RandomAgent):
        # Match evaluate.py's split so act_key chains stay comparable.
        rng_key, _ = jax.random.split(rng_key)

    video_frames: list[np.ndarray] | None = (
        [] if args.video_fps > 0 else None
    )
    results: list[dict] = []
    for ep_idx in range(args.episodes):
        ep_result, rng_key = run_episode(
            ep_idx=ep_idx,
            args=args,
            env=env,
            adapter=adapter,
            agent=agent,
            rng_key=rng_key,
            video_frames=video_frames,
            out_dir=out_dir,
        )
        results.append(ep_result)

    export_outputs(adapter, results, video_frames, args, out_dir)


def main() -> None:
    """Run the collection and record ok/failed status in MANIFEST.json."""
    args = parse_args()
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    agent_tag = "random" if args.random else "actor"
    run_tag = (
        time.strftime("%Y%m%d_%H%M%S")
        + f"_{agent_tag}_seed{args.seed}_ep{args.episodes}"
    )
    out_dir = Path(args.output) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hybrid-actor] output dir: {out_dir}", flush=True)
    collect(args, out_dir)


if __name__ == "__main__":
    main()
