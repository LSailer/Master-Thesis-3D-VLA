"""Run R2-Dreamer (JAX) on Crafter, output metrics to CSV."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import jax

from modules.r2dreamer.agent import R2DreamerAgent
from modules.r2dreamer.config import R2DreamerConfig
from modules.r2dreamer.trainer import Trainer, TrainerConfig
from modules.envs.crafter import CrafterEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--prefill", type=int, default=5000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=250)
    args = parser.parse_args()

    config = R2DreamerConfig(
        obs_shape=(3, 64, 64),
        num_actions=17,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        seed=args.seed,
        log_every=args.log_every,
    )

    env = CrafterEnv(size=(64, 64), seed=args.seed)
    rng_key, init_key = jax.random.split(jax.random.PRNGKey(config.seed))
    agent = R2DreamerAgent(config, init_key)

    trainer = Trainer(
        agent=agent,
        env=env,
        agent_config=config,
        trainer_config=TrainerConfig(
            output_dir=os.path.dirname(args.output) or ".",
            total_steps=args.steps,
            prefill_steps=args.prefill,
            log_every=args.log_every,
            seed=args.seed,
            wandb_project=None,  # no WandB for Crafter sanity checks
        ),
    )
    trainer.run()


if __name__ == "__main__":
    main()
