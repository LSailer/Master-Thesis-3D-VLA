"""DreamerV3 training entry point for HM3D ObjectNav.

Usage:
    python -m src.dreamer.train --config configs/dreamer_objectnav.yaml
    python -m src.dreamer.train --total-steps 100 --prefill-steps 10  # smoke test
"""

import argparse
import logging
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .agent import train_actor_critic, train_world_model
from .config import DreamerConfig
from .envs import DummyEnv, HabitatObjectNavEnv
from .networks import Critic, DiscreteActor, WorldModel
from .replay_buffer import ReplayBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_optimizer(lr: float, max_grad_norm: float) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )


def act(
    wm: WorldModel,
    actor: DiscreteActor,
    h: jnp.ndarray,
    z: jnp.ndarray,
    obs: jnp.ndarray,
    prev_action: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Single environment step: encode obs, update RSSM, sample action.

    Returns: (h', z', action_onehot, action_int)
    """
    k1, k2, k3 = jax.random.split(key, 3)
    embed = wm.encoder(obs)
    h = wm.recurrent(h, z, prev_action)
    posterior = wm.posterior(h, embed, key=k1)
    z = posterior.sample
    state = jnp.concatenate([h, z])
    action_onehot = actor.sample(state, key=k2)
    action_int = jnp.argmax(action_onehot)
    return h, z, action_onehot, int(action_int)


def train(config: DreamerConfig):
    logger.info("Initializing DreamerV3 training")
    logger.info(f"Config: action_size={config.action_size}, obs_shape={config.obs_shape}")

    key = jax.random.PRNGKey(config.seed)

    # --- Initialize models ---
    key, k_wm, k_actor, k_critic = jax.random.split(key, 4)
    wm = WorldModel(config, key=k_wm)
    actor = DiscreteActor(
        config.state_size, config.action_size,
        config.hidden_size, config.num_layers, key=k_actor,
    )
    critic = Critic(
        config.state_size, config.hidden_size,
        config.num_layers, key=k_critic,
    )

    # --- Optimizers ---
    wm_opt = make_optimizer(config.learning_rate, config.max_grad_norm)
    actor_opt = make_optimizer(config.learning_rate, config.max_grad_norm)
    critic_opt = make_optimizer(config.learning_rate, config.max_grad_norm)

    wm_opt_state = wm_opt.init(eqx.filter(wm, eqx.is_array))
    actor_opt_state = actor_opt.init(eqx.filter(actor, eqx.is_array))
    critic_opt_state = critic_opt.init(eqx.filter(critic, eqx.is_array))

    # --- Environment ---
    try:
        env = HabitatObjectNavEnv()
        logger.info("Using Habitat ObjectNav environment")
    except Exception:
        env = DummyEnv(obs_shape=config.obs_shape, action_size=config.action_size)
        logger.info("Habitat not available, using DummyEnv")

    # --- Replay buffer ---
    buffer = ReplayBuffer(config.obs_shape, config.action_size, config.buffer_capacity)

    # --- W&B ---
    try:
        import wandb
        wandb.init(project=config.wandb_project, config=vars(config))
        use_wandb = True
    except Exception:
        use_wandb = False
        logger.info("W&B not available, logging to stdout only")

    # --- Training loop ---
    obs = env.reset()
    h, z = wm.initial_state(batch_size=1)
    prev_action = jnp.zeros(config.action_size)
    episode_reward = 0.0
    episode_count = 0

    for step in range(config.total_steps):
        key, k_act = jax.random.split(key)

        # Act
        if step < config.prefill_steps:
            action_int = np.random.randint(0, config.action_size)
            action_onehot = jnp.zeros(config.action_size).at[action_int].set(1.0)
        else:
            h, z, action_onehot, action_int = act(
                wm, actor, h, z, jnp.array(obs), prev_action, k_act,
            )

        # Step environment
        next_obs, reward, done, info = env.step(action_int)
        buffer.add(obs, np.array(action_onehot), reward, done)
        episode_reward += reward
        prev_action = action_onehot

        if done:
            episode_count += 1
            logger.info(f"Episode {episode_count} reward={episode_reward:.2f} step={step}")
            if use_wandb:
                wandb.log({
                    "episode/reward": episode_reward,
                    "episode/count": episode_count,
                    **{f"episode/{k}": v for k, v in info.items() if isinstance(v, (int, float))},
                }, step=step)
            episode_reward = 0.0
            obs = env.reset()
            h, z = wm.initial_state(batch_size=1)
            prev_action = jnp.zeros(config.action_size)
        else:
            obs = next_obs

        # Train
        if step >= config.prefill_steps and step % config.train_every == 0 and len(buffer) > config.batch_size * config.sequence_length:
            key, k_wm_train, k_ac_train = jax.random.split(key, 3)
            batch = buffer.sample(config.batch_size, config.sequence_length)

            # World model update
            wm, wm_opt_state, wm_metrics = train_world_model(
                wm, wm_opt_state, wm_opt, batch, config, k_wm_train,
            )

            # Get initial states for imagination from the batch
            # Use last timestep states as starting points
            h_batch, z_batch = wm.initial_state(batch_size=config.batch_size)

            # Actor-critic update
            actor, critic, actor_opt_state, critic_opt_state, ac_metrics = train_actor_critic(
                wm, actor, critic,
                actor_opt_state, critic_opt_state,
                actor_opt, critic_opt,
                h_batch, z_batch, config, k_ac_train,
            )

            if step % config.log_interval == 0:
                metrics = {**wm_metrics, **ac_metrics}
                logger.info(f"Step {step}: " + ", ".join(f"{k}={float(v):.4f}" for k, v in metrics.items()))
                if use_wandb:
                    wandb.log({k: float(v) for k, v in metrics.items()}, step=step)

    # --- Save checkpoint ---
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(ckpt_dir / "world_model.eqx", wm)
    eqx.tree_serialise_leaves(ckpt_dir / "actor.eqx", actor)
    eqx.tree_serialise_leaves(ckpt_dir / "critic.eqx", critic)
    logger.info(f"Saved checkpoint to {ckpt_dir}")

    env.close()
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="DreamerV3 for HM3D ObjectNav")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--prefill-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        config = DreamerConfig.from_yaml(args.config)
    else:
        config = DreamerConfig()

    # CLI overrides
    if args.total_steps is not None:
        config = DreamerConfig(**{**vars(config), "total_steps": args.total_steps})
    if args.prefill_steps is not None:
        config = DreamerConfig(**{**vars(config), "prefill_steps": args.prefill_steps})
    if args.seed is not None:
        config = DreamerConfig(**{**vars(config), "seed": args.seed})

    train(config)


if __name__ == "__main__":
    main()
