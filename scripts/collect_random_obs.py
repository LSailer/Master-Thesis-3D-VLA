"""Collect observations from random-action rollouts in Habitat ObjectNav."""

import argparse
import random
from pathlib import Path

import numpy as np

try:
    import habitat_sim
except ImportError:
    raise ImportError("habitat-sim required — install via pixi")

import wandb


NUM_ACTIONS = 6  # STOP=0, MOVE_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3, LOOK_UP=4, LOOK_DOWN=5


def make_sim(scene_path: str) -> habitat_sim.Simulator:
    """Create Habitat simulator for a single scene."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    # RGB + depth sensors
    rgb_cfg = habitat_sim.CameraSensorSpec()
    rgb_cfg.uuid = "rgb"
    rgb_cfg.sensor_type = habitat_sim.SensorType.COLOR
    rgb_cfg.resolution = [256, 256]
    rgb_cfg.position = [0.0, 1.5, 0.0]

    depth_cfg = habitat_sim.CameraSensorSpec()
    depth_cfg.uuid = "depth"
    depth_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_cfg.resolution = [256, 256]
    depth_cfg.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=[rgb_cfg, depth_cfg])
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def get_scene_paths(scene_dataset: str) -> list[str]:
    """Glob for .glb scene files."""
    paths = sorted(Path(scene_dataset).rglob("*.glb"))
    if not paths:
        raise FileNotFoundError(f"No .glb scenes in {scene_dataset}")
    return [str(p) for p in paths]


def run_episode(
    sim: habitat_sim.Simulator,
    max_steps: int,
    episode_id: str,
    output_dir: Path | None,
    log_image_every: int,
    global_step: int,
) -> tuple[int, int]:
    """Run one random-action episode. Returns (steps_taken, new_global_step)."""
    action_names = ["stop", "move_forward", "turn_left", "turn_right", "look_up", "look_down"]

    agent = sim.get_agent(0)
    agent.state = habitat_sim.AgentState()
    sim.reset()

    obs_list = []
    step = 0
    for step in range(max_steps):
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]  # drop alpha
        depth = obs["depth"]

        # Wandb logging
        metrics = {"step/global_step": global_step, "step/episode": episode_id}
        if global_step % log_image_every == 0:
            metrics["step/rgb"] = wandb.Image(rgb)
            depth_vis = (depth / (depth.max() + 1e-6) * 255).astype(np.uint8)
            metrics["step/depth"] = wandb.Image(depth_vis)
        wandb.log(metrics, step=global_step)

        # Save obs
        if output_dir is not None:
            obs_list.append({"rgb": rgb, "depth": depth})

        # Random action (exclude STOP=0 to keep episode going)
        action_idx = random.randint(1, NUM_ACTIONS - 1)
        sim.step(action_names[action_idx])
        global_step += 1

    # Save episode observations
    if output_dir is not None:
        ep_dir = output_dir / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ep_dir / "observations.npz",
            rgb=np.stack([o["rgb"] for o in obs_list]),
            depth=np.stack([o["depth"] for o in obs_list]),
        )

    return step + 1, global_step


def main():
    parser = argparse.ArgumentParser(description="Collect random-action observations")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--wandb-project", default="3d-vla-objectnav")
    parser.add_argument("--scene-dataset", default="data/scene_datasets/hm3d/minival")
    parser.add_argument("--output-dir", default="output/observations")
    parser.add_argument("--log-image-every", type=int, default=50)
    parser.add_argument("--no-save", action="store_true", help="Skip saving .npz files")
    args = parser.parse_args()

    output_dir = None if args.no_save else Path(args.output_dir)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    scenes = get_scene_paths(args.scene_dataset)
    print(f"Found {len(scenes)} scenes in {args.scene_dataset}")

    run = wandb.init(
        project=args.wandb_project,
        config=vars(args),
        tags=["random-rollout", "observation-collection"],
    )

    # Wandb table for per-episode summary
    ep_table = wandb.Table(columns=["episode_id", "scene", "steps"])

    global_step = 0
    for ep in range(args.num_episodes):
        scene_path = scenes[ep % len(scenes)]
        episode_id = f"ep_{ep:04d}"
        print(f"[{ep+1}/{args.num_episodes}] {episode_id} — {Path(scene_path).stem}")

        sim = make_sim(scene_path)
        steps, global_step = run_episode(
            sim, args.max_steps, episode_id, output_dir, args.log_image_every, global_step
        )
        sim.close()

        ep_table.add_data(episode_id, Path(scene_path).stem, steps)
        wandb.log({"episode/length": steps, "episode/index": ep}, step=global_step)
        print(f"  steps={steps}")

    wandb.log({"episodes_summary": ep_table})
    run.finish()
    print(f"Done. {args.num_episodes} episodes, {global_step} total steps.")


if __name__ == "__main__":
    main()
