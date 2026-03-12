"""DreamerV3 training loop for HM3D ObjectNav."""

import argparse
import dataclasses
import os

import jax
import numpy as np

from .configs import DreamerConfig
from .agent import DreamerAgent
from .replay_buffer import ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    for f in dataclasses.fields(DreamerConfig):
        if f.type in (int, float, str):
            parser.add_argument(f"--{f.name}", type=f.type, default=f.default)
    args = parser.parse_args()

    config = DreamerConfig(**{f.name: getattr(args, f.name)
                              for f in dataclasses.fields(DreamerConfig)
                              if f.type in (int, float, str)})

    rng_key = jax.random.PRNGKey(config.seed)

    # Lazy import — Habitat may not be available during shape tests
    from .env_habitat import HabitatObjectNavEnv
    env = HabitatObjectNavEnv(config)

    rng_key, init_key = jax.random.split(rng_key)
    agent = DreamerAgent(config, init_key)
    buffer = ReplayBuffer(config)

    # Optional wandb
    try:
        import wandb
        wandb.init(project="dreamerv3-objectnav", config=dataclasses.asdict(config),
                   dir=config.logdir)
        use_wandb = True
    except Exception:
        use_wandb = False

    os.makedirs(config.logdir, exist_ok=True)

    # Prefill with random actions
    print(f"Prefilling buffer with {config.prefill_steps} random steps...")
    obs = env.reset()
    for _ in range(config.prefill_steps):
        action = np.random.randint(0, config.num_actions)
        next_obs = env.step(action)
        buffer.add(obs["image"], action, next_obs["reward"], next_obs["done"])
        obs = next_obs if not next_obs["done"] else env.reset()

    # Training loop
    print(f"Starting training for {config.total_steps} steps...")
    obs = env.reset()
    episode_reward = 0.0
    episode_count = 0

    for step in range(config.total_steps):
        rng_key, act_key, train_key = jax.random.split(rng_key, 3)

        # Act
        action = agent.act(obs, act_key)
        next_obs = env.step(action)
        buffer.add(obs["image"], action, next_obs["reward"], next_obs["done"])
        episode_reward += next_obs["reward"]

        if next_obs["done"]:
            episode_count += 1
            log_data = {
                "episode_reward": episode_reward,
                "episode_count": episode_count,
                "success": next_obs.get("success", 0.0),
                "spl": next_obs.get("spl", 0.0),
            }
            if use_wandb:
                wandb.log(log_data, step=step)
            if step % config.log_every == 0:
                print(f"[step {step}] ep={episode_count} reward={episode_reward:.2f} "
                      f"success={next_obs.get('success', 0):.1f}")
            episode_reward = 0.0
            obs = env.reset()
        else:
            obs = next_obs

        # Train
        if buffer.size >= config.batch_size * config.seq_len:
            batch = buffer.sample(config.batch_size, config.seq_len)
            metrics = agent.train_step(batch, train_key)

            if step % config.log_every == 0:
                if use_wandb:
                    wandb.log(metrics, step=step)
                print(f"[step {step}] wm={metrics.get('wm_loss', 0):.3f} "
                      f"actor={metrics.get('actor_loss', 0):.3f} "
                      f"critic={metrics.get('critic_loss', 0):.3f}")

        # Save
        if step > 0 and step % config.save_every == 0:
            agent.save(config.logdir)
            print(f"[step {step}] Checkpoint saved to {config.logdir}")

    agent.save(config.logdir)
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
