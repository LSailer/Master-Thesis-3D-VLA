"""DreamerV3 agent: world model training + actor-critic imagination training.

All training functions are JIT-compiled. Uses Equinox's functional style:
filter params/static, compute grads, apply updates via optax.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .config import DreamerConfig
from .networks import DiscreteActor, Critic, WorldModel


# ---------------------------------------------------------------------------
# World Model Training
# ---------------------------------------------------------------------------


def _kl_divergence(
    post_logits: jnp.ndarray,
    prior_logits: jnp.ndarray,
    free_nats: float,
    balance: float,
) -> jnp.ndarray:
    """KL divergence between two categorical distributions with free nats and balancing.

    post_logits, prior_logits: (latent_length, latent_classes)
    Returns scalar KL loss.
    """
    post_probs = jax.nn.softmax(post_logits, axis=-1)
    prior_probs = jax.nn.softmax(prior_logits, axis=-1)

    # KL(post || prior) per categorical variable
    kl_per_var = jnp.sum(
        post_probs * (jnp.log(post_probs + 1e-8) - jnp.log(prior_probs + 1e-8)),
        axis=-1,
    )  # (latent_length,)
    kl = jnp.sum(kl_per_var)  # scalar

    # Free nats: don't penalize KL below threshold
    kl = jnp.maximum(kl, free_nats)

    return kl


def _world_model_loss(
    wm: WorldModel,
    batch: dict[str, jnp.ndarray],
    config: DreamerConfig,
    key: jax.Array,
):
    """Compute world model losses on a single sequence.

    batch: dict with (seq_len, ...) arrays (no batch dim — use vmap outside).
    Returns scalar loss and metrics dict.
    """
    seq_len = batch["observations"].shape[0]
    h, z = wm.initial_state(batch_size=1)

    recon_loss = 0.0
    reward_loss = 0.0
    continue_loss = 0.0
    kl_loss = 0.0

    keys = jax.random.split(key, seq_len * 2)

    def step_fn(carry, t):
        h, z, losses = carry
        obs = batch["observations"][t]
        action = batch["actions"][t]
        reward = batch["rewards"][t]
        done = batch["dones"][t]

        # Encode observation
        embed = wm.encoder(obs)

        # Recurrent update
        h = wm.recurrent(h, z, action)

        # Prior and posterior
        prior = wm.prior(h, key=keys[t * 2])
        posterior = wm.posterior(h, embed, key=keys[t * 2 + 1])
        z = posterior.sample

        # State for predictions
        state = jnp.concatenate([h, z])

        # Reconstruction loss
        recon = wm.decoder(state)
        recon_l = jnp.mean((recon - obs) ** 2)

        # Reward loss (Gaussian NLL)
        r_mean, r_std = wm.reward(state)
        reward_l = 0.5 * ((reward - r_mean) / (r_std + 1e-8)) ** 2 + jnp.log(r_std + 1e-8)

        # Continue loss (binary cross-entropy)
        cont_logit = wm.continue_model(state)
        cont_target = 1.0 - done
        continue_l = -cont_target * jax.nn.log_sigmoid(cont_logit) - (1 - cont_target) * jax.nn.log_sigmoid(-cont_logit)

        # KL loss
        kl_l = _kl_divergence(posterior.logits, prior.logits, config.kl_free_nats, config.kl_balance)

        new_losses = (
            losses[0] + recon_l,
            losses[1] + reward_l,
            losses[2] + continue_l,
            losses[3] + kl_l,
        )
        return (h, z, new_losses), None

    init_losses = (0.0, 0.0, 0.0, 0.0)
    (h_final, z_final, (recon_loss, reward_loss, continue_loss, kl_loss)), _ = jax.lax.scan(
        step_fn, (h, z, init_losses), jnp.arange(seq_len),
    )

    # Average over sequence
    recon_loss = recon_loss / seq_len
    reward_loss = reward_loss / seq_len
    continue_loss = continue_loss / seq_len
    kl_loss = kl_loss / seq_len

    total = recon_loss + reward_loss + continue_loss + kl_loss

    metrics = {
        "wm/total": total,
        "wm/recon": recon_loss,
        "wm/reward": reward_loss,
        "wm/continue": continue_loss,
        "wm/kl": kl_loss,
    }
    return total, metrics


def train_world_model(
    wm: WorldModel,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: dict[str, jnp.ndarray],
    config: DreamerConfig,
    key: jax.Array,
) -> tuple[WorldModel, optax.OptState, dict]:
    """One gradient step on the world model.

    batch: dict with (batch_size, seq_len, ...) arrays.
    """
    # Average loss over batch using vmap
    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(wm):
        keys = jax.random.split(key, config.batch_size)

        def single_loss(i):
            single_batch = jax.tree.map(lambda x: x[i], batch)
            return _world_model_loss(wm, single_batch, config, keys[i])

        losses_and_metrics = jax.vmap(lambda i: single_loss(i))(jnp.arange(config.batch_size))
        mean_loss = jnp.mean(losses_and_metrics[0])
        mean_metrics = jax.tree.map(lambda x: jnp.mean(x), losses_and_metrics[1])
        return mean_loss, mean_metrics

    (loss, metrics), grads = loss_fn(wm)
    updates, opt_state = optimizer.update(grads, opt_state, wm)
    wm = eqx.apply_updates(wm, updates)
    return wm, opt_state, metrics


# ---------------------------------------------------------------------------
# Actor-Critic Imagination Training
# ---------------------------------------------------------------------------


def _imagine_trajectory(
    wm: WorldModel,
    actor: DiscreteActor,
    h0: jnp.ndarray,
    z0: jnp.ndarray,
    horizon: int,
    key: jax.Array,
) -> dict[str, jnp.ndarray]:
    """Imagine a trajectory using the world model and actor policy.

    Returns dict with (horizon, ...) arrays for states, rewards, continues, actions.
    """
    def step_fn(carry, t):
        h, z, key = carry
        key, k_act, k_prior = jax.random.split(key, 3)

        state = jnp.concatenate([h, z])
        action = actor.sample(state, key=k_act)

        # World model step (prior only — no observations in imagination)
        h = wm.recurrent(h, z, action)
        prior = wm.prior(h, key=k_prior)
        z = prior.sample

        new_state = jnp.concatenate([h, z])
        r_mean, _ = wm.reward(new_state)
        cont_logit = wm.continue_model(new_state)
        cont = jax.nn.sigmoid(cont_logit)

        return (h, z, key), {
            "states": new_state,
            "rewards": r_mean,
            "continues": cont,
            "actions": action,
        }

    _, trajectory = jax.lax.scan(step_fn, (h0, z0, key), jnp.arange(horizon))
    return trajectory


def _lambda_returns(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    continues: jnp.ndarray,
    gamma: float,
    lambda_: float,
) -> jnp.ndarray:
    """Compute lambda-returns. All inputs: (horizon,). Returns: (horizon,)."""
    horizon = rewards.shape[0]

    def scan_fn(next_return, t):
        # Reverse scan: from last step backwards
        idx = horizon - 1 - t
        r = rewards[idx]
        v = values[idx]
        c = continues[idx]
        ret = r + gamma * c * ((1 - lambda_) * v + lambda_ * next_return)
        return ret, ret

    _, returns = jax.lax.scan(scan_fn, values[-1], jnp.arange(horizon))
    # Reverse to get correct order
    return returns[::-1]


def train_actor_critic(
    wm: WorldModel,
    actor: DiscreteActor,
    critic: Critic,
    actor_opt_state: optax.OptState,
    critic_opt_state: optax.OptState,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    h0: jnp.ndarray,
    z0: jnp.ndarray,
    config: DreamerConfig,
    key: jax.Array,
) -> tuple[DiscreteActor, Critic, optax.OptState, optax.OptState, dict]:
    """Train actor and critic on imagined trajectories.

    h0, z0: initial states from the world model (batch_size, dim).
    """
    # --- Actor loss ---
    @eqx.filter_value_and_grad(has_aux=True)
    def actor_loss_fn(actor):
        k1, k2 = jax.random.split(key)

        def single_imagine(i):
            ki = jax.random.fold_in(k1, i)
            traj = _imagine_trajectory(wm, actor, h0[i], z0[i], config.imagination_horizon, ki)
            # Compute values with current critic (detached)
            values = jax.vmap(lambda s: critic(s)[0])(traj["states"])
            returns = _lambda_returns(
                traj["rewards"], values, traj["continues"],
                config.gamma, config.lambda_,
            )
            # Actor loss: maximize returns + entropy bonus
            logits = jax.vmap(actor)(traj["states"])
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
            actor_loss = -jnp.mean(returns) - config.entropy_scale * jnp.mean(entropy)
            return actor_loss, {"ac/entropy": jnp.mean(entropy), "ac/returns": jnp.mean(returns)}

        losses_and_metrics = jax.vmap(single_imagine)(jnp.arange(h0.shape[0]))
        mean_loss = jnp.mean(losses_and_metrics[0])
        mean_metrics = jax.tree.map(jnp.mean, losses_and_metrics[1])
        return mean_loss, mean_metrics

    (actor_loss, actor_metrics), actor_grads = actor_loss_fn(actor)
    actor_updates, actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state, actor)
    actor = eqx.apply_updates(actor, actor_updates)

    # --- Critic loss ---
    @eqx.filter_value_and_grad(has_aux=True)
    def critic_loss_fn(critic):
        k1, _ = jax.random.split(key)

        def single_critic(i):
            ki = jax.random.fold_in(k1, i)
            # Re-imagine with updated actor (or use same trajectory — simpler)
            traj = _imagine_trajectory(wm, actor, h0[i], z0[i], config.imagination_horizon, ki)
            values_mean, values_std = jax.vmap(critic)(traj["states"])
            target_values = jax.vmap(lambda s: critic(s)[0])(traj["states"])
            returns = _lambda_returns(
                traj["rewards"], target_values, traj["continues"],
                config.gamma, config.lambda_,
            )
            # Gaussian NLL
            nll = 0.5 * ((returns - values_mean) / (values_std + 1e-8)) ** 2 + jnp.log(values_std + 1e-8)
            return jnp.mean(nll), {"ac/value": jnp.mean(values_mean)}

        losses_and_metrics = jax.vmap(single_critic)(jnp.arange(h0.shape[0]))
        mean_loss = jnp.mean(losses_and_metrics[0])
        mean_metrics = jax.tree.map(jnp.mean, losses_and_metrics[1])
        return mean_loss, mean_metrics

    (critic_loss, critic_metrics), critic_grads = critic_loss_fn(critic)
    critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state, critic)
    critic = eqx.apply_updates(critic, critic_updates)

    metrics = {
        "ac/actor_loss": actor_loss,
        "ac/critic_loss": critic_loss,
        **actor_metrics,
        **critic_metrics,
    }
    return actor, critic, actor_opt_state, critic_opt_state, metrics
