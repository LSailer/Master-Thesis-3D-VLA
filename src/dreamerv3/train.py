"""DreamerV3 training loop for HM3D ObjectNav."""

import argparse
import dataclasses
import os

import jax
import numpy as np

from .configs import DreamerConfig
from .agent import DreamerAgent
from .replay_buffer import ReplayBuffer


def greedy_eval(agent, env, rng_key, num_episodes=10, max_steps=500):
    """Run greedy evaluation episodes, return mean metrics."""
    rewards, successes, spls, lengths = [], [], [], []
    for _ in range(num_episodes):
        obs = env.reset()
        ep_reward, steps = 0.0, 0
        rng_key, act_key = jax.random.split(rng_key)
        while not obs.get("done", False) and steps < max_steps:
            action = agent.act(obs, act_key, training=False)
            obs = env.step(action)
            ep_reward += obs["reward"]
            steps += 1
        rewards.append(ep_reward)
        successes.append(obs.get("success", 0.0))
        spls.append(obs.get("spl", 0.0))
        lengths.append(steps)
    return {
        "eval/reward": np.mean(rewards),
        "eval/success": np.mean(successes),
        "eval/spl": np.mean(spls),
        "eval/episode_length": np.mean(lengths),
    }


def main():
    parser = argparse.ArgumentParser()
    for f in dataclasses.fields(DreamerConfig):
        if f.type in (int, float, str):
            parser.add_argument(f"--{f.name}", type=f.type, default=f.default)
    parser.add_argument("--obs_size", type=int, default=None,
                        help="Override obs resolution (sets obs_shape=(3,N,N))")
    parser.add_argument("--max_geodesic", type=float, default=None,
                        help="Filter episodes to geodesic distance < this value")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="WandB run name")
    parser.add_argument("--wandb_tags", type=str, default=None,
                        help="WandB tags (comma-separated)")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="WandB group name")
    parser.add_argument("--eval_every", type=int, default=50_000,
                        help="Run greedy eval every N steps (0 to disable)")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of greedy eval episodes")
    args = parser.parse_args()

    config = DreamerConfig(**{f.name: getattr(args, f.name)
                              for f in dataclasses.fields(DreamerConfig)
                              if f.type in (int, float, str)})
    if args.obs_size is not None:
        config = dataclasses.replace(config, obs_shape=(3, args.obs_size, args.obs_size))

    rng_key = jax.random.PRNGKey(config.seed)

    # Lazy import — Habitat may not be available during shape tests
    from .env_habitat import HabitatObjectNavEnv
    env = HabitatObjectNavEnv(config, max_geodesic=args.max_geodesic)

    rng_key, init_key = jax.random.split(rng_key)
    agent = DreamerAgent(config, init_key)
    buffer = ReplayBuffer(config)

    # Resume from checkpoint
    start_step = 0
    if args.checkpoint is not None:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Optional wandb
    try:
        import wandb
        wandb_kwargs = {
            "project": "dreamerv3-objectnav",
            "config": dataclasses.asdict(config),
            "dir": config.logdir,
        }
        if args.wandb_name:
            wandb_kwargs["name"] = args.wandb_name
        if args.wandb_tags:
            wandb_kwargs["tags"] = args.wandb_tags.split(",")
        if args.wandb_group:
            wandb_kwargs["group"] = args.wandb_group
        wandb.init(**wandb_kwargs)
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

    for step in range(start_step, config.total_steps):
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
                      f"actor={metrics.get('loss/policy', 0):.3f} "
                      f"critic={metrics.get('loss/value', 0):.3f}")

        # Save
        if step > 0 and step % config.save_every == 0:
            agent.save(config.logdir)
            print(f"[step {step}] Checkpoint saved to {config.logdir}")

        # Periodic eval
        if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            rng_key, eval_key = jax.random.split(rng_key)
            eval_metrics = greedy_eval(
                agent, env, eval_key,
                num_episodes=args.eval_episodes,
                max_steps=config.max_episode_steps,
            )
            if use_wandb:
                wandb.log(eval_metrics, step=step)
            print(f"[step {step}] EVAL: reward={eval_metrics['eval/reward']:.2f} "
                  f"SR={eval_metrics['eval/success']:.2f} "
                  f"SPL={eval_metrics['eval/spl']:.2f}")
            # greedy_eval mutates the shared env; reset before resuming training
            obs = env.reset()
            episode_reward = 0.0

    agent.save(config.logdir)
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
