"""Run R2-Dreamer (JAX) on Habitat ObjectNav, output metrics to CSV + WandB."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import jax

from modules.r2dreamer.agent import R2DreamerAgent
from modules.r2dreamer.config import R2DreamerConfig
from modules.r2dreamer.trainer import Trainer, TrainerConfig, habitat_defaults
from modules.shared.configs import DreamerConfig
from modules.envs.habitat import HabitatObjectNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--prefill", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--checkpoint_every", type=int, default=50_000)
    parser.add_argument("--wandb_project", type=str, default="3d-vla-objectnav")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to pre-collected val .npz for val loss")
    parser.add_argument("--val_loss_every", type=int, default=10_000,
                        help="Compute val loss every N steps")
    parser.add_argument("--step_counts_path", type=str, default=None,
                        help="Path to episode_step_counts.json for filtering")
    parser.add_argument("--curriculum_path", type=str, default=None,
                        help="Path to curriculum JSON config")
    parser.add_argument("--curriculum_mode", type=str, default="train",
                        help="Curriculum split: train or eval")
    parser.add_argument("--wandb_tags", type=str, default=None,
                        help="Comma-separated WandB tags (appended to defaults)")
    args = parser.parse_args()

    # --- Model config ---
    config = R2DreamerConfig(
        obs_shape=(3, 64, 64),
        num_actions=4,  # STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
        total_steps=args.steps,
        prefill_steps=args.prefill,
        buffer_capacity=1_000_000,
        act_entropy=3e-2,
        seed=args.seed,
        log_every=args.log_every,
        logdir=args.output_dir,
    )

    # --- Environment ---
    hab_config = DreamerConfig(
        obs_shape=(3, 64, 64),
        max_episode_steps=500,
        split="train",
        reward_type="geodesic_delta",
    )
    env = HabitatObjectNavEnv(
        hab_config,
        step_counts_path=args.step_counts_path,
        curriculum_path=args.curriculum_path,
        curriculum_mode=args.curriculum_mode,
    )

    # --- Agent ---
    rng_key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    agent = R2DreamerAgent(config, init_key)

    # --- Trainer ---
    tags = ["r2dreamer", "habitat", "baseline"]
    if args.wandb_tags:
        tags.extend(t.strip() for t in args.wandb_tags.split(","))

    trainer = Trainer(
        agent=agent,
        env=env,
        agent_config=config,
        trainer_config=TrainerConfig(
            output_dir=args.output_dir,
            total_steps=args.steps,
            prefill_steps=args.prefill,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            wandb_tags=tags,
            val_data=args.val_data,
            val_loss_every=args.val_loss_every,
        ),
        **habitat_defaults(env),
    )
    trainer.run()


if __name__ == "__main__":
    main()
