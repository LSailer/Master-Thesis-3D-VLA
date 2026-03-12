"""DreamerV3 agent — world model + behavior training in JAX."""

import functools

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .configs import DreamerConfig
from .networks import Encoder, Decoder, RSSM, MLP, Actor, Critic, symlog


def _feature(h, z):
    """Concatenate deterministic and stochastic state."""
    return jnp.concatenate([h, z], axis=-1)


class DreamerAgent:
    def __init__(self, config: DreamerConfig, rng_key: jax.Array):
        self.cfg = config
        feat_size = config.hidden_size + config.stoch_size

        # Init networks
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng_key, 7)
        dummy_obs = jnp.zeros((1, *config.obs_shape))
        dummy_feat = jnp.zeros((1, feat_size))
        dummy_h = jnp.zeros((1, config.hidden_size))
        dummy_z = jnp.zeros((1, config.stoch_size))
        dummy_a = jnp.zeros((1,), dtype=jnp.int32)

        self.encoder = Encoder(config.encoder_depth, config.hidden_size)
        self.decoder = Decoder(config.encoder_depth, config.hidden_size, config.obs_shape)
        self.rssm = RSSM(config.hidden_size, config.latent_classes, config.latent_dims, config.num_actions)
        self.reward_pred = MLP(config.mlp_hidden, config.mlp_layers, 1, name="reward")
        self.continue_pred = MLP(config.mlp_hidden, config.mlp_layers, 1, name="continue")
        self.actor = Actor(config.mlp_hidden, config.mlp_layers, config.num_actions)
        self.critic = Critic(config.mlp_hidden, config.mlp_layers)

        # Init params
        enc_p = self.encoder.init(k1, dummy_obs)
        dec_p = self.decoder.init(k2, dummy_feat)
        rssm_p = self.rssm.init(k3, dummy_h, dummy_z, dummy_a, jnp.zeros((1, config.hidden_size)))
        rew_p = self.reward_pred.init(k4, dummy_feat)
        cont_p = self.continue_pred.init(k5, dummy_feat)
        actor_p = self.actor.init(k6, dummy_feat)
        critic_p = self.critic.init(k7, dummy_feat)

        # World model params (bundled)
        wm_params = {"encoder": enc_p, "decoder": dec_p, "rssm": rssm_p,
                      "reward": rew_p, "continue": cont_p}

        def make_ts(params, lr):
            return TrainState.create(
                apply_fn=None, params=params,
                tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                               optax.adam(lr)),
            )

        self.wm_state = make_ts(wm_params, config.lr_world)
        self.actor_state = make_ts(actor_p, config.lr_actor)
        self.critic_state = make_ts(critic_p, config.lr_critic)
        self.critic_target_params = critic_p

        # Persistent RSSM state for acting
        self._act_h = jnp.zeros((1, config.hidden_size))
        self._act_z = jnp.zeros((1, config.stoch_size))
        self._act_a = jnp.zeros((1,), dtype=jnp.int32)

        # JIT compile
        self._train_step_jit = jax.jit(self._train_step)
        self._act_jit = jax.jit(self._act_forward)

    # ------ Acting ------

    def act(self, obs_dict: dict, rng_key: jax.Array, training: bool = True) -> int:
        obs = jnp.array(obs_dict["image"][None], dtype=jnp.float32) / 255.0

        if obs_dict.get("is_first", False):
            self._act_h = jnp.zeros_like(self._act_h)
            self._act_z = jnp.zeros_like(self._act_z)
            self._act_a = jnp.zeros_like(self._act_a)

        action, self._act_h, self._act_z = self._act_jit(
            self.wm_state.params, self.actor_state.params,
            obs, self._act_h, self._act_z, self._act_a, rng_key,
        )
        a = int(action[0])
        self._act_a = action
        return a

    def _act_forward(self, wm_params, actor_params, obs, h, z, prev_a, rng_key):
        embed = self.encoder.apply(wm_params["encoder"], obs)
        h_new, _, _, z_new = self.rssm.apply(wm_params["rssm"], h, z, prev_a, embed)
        feat = _feature(h_new, z_new)
        logits = self.actor.apply(actor_params, feat)
        action = jax.random.categorical(rng_key, logits)
        return action, h_new, z_new

    # ------ Training ------

    def train_step(self, batch: dict, rng_key: jax.Array) -> dict:
        (self.wm_state, self.actor_state, self.critic_state,
         self.critic_target_params, metrics) = self._train_step_jit(
            self.wm_state, self.actor_state, self.critic_state,
            self.critic_target_params, batch, rng_key,
        )
        return {k: float(v) for k, v in metrics.items()}

    def _train_step(self, wm_state, actor_state, critic_state,
                    critic_target, batch, rng_key):
        k1, k2 = jax.random.split(rng_key)

        # --- World model ---
        wm_state, wm_metrics, states = self._train_world_model(wm_state, batch, k1)

        # --- Behavior (imagination) ---
        actor_state, critic_state, critic_target, beh_metrics = (
            self._train_behavior(wm_state.params, actor_state, critic_state,
                                 critic_target, states, k2)
        )

        metrics = {**wm_metrics, **beh_metrics}
        return wm_state, actor_state, critic_state, critic_target, metrics

    def _train_world_model(self, wm_state, batch, rng_key):
        def wm_loss_fn(params):
            obs = batch["obs"]  # (B, T, C, H, W)
            actions = batch["actions"]  # (B, T)
            rewards = batch["rewards"]  # (B, T)
            dones = batch["dones"]  # (B, T)
            is_first = batch["is_first"]  # (B, T)
            B, T = obs.shape[0], obs.shape[1]

            # Encode all timesteps
            flat_obs = obs.reshape(B * T, *obs.shape[2:])
            flat_embed = self.encoder.apply(params["encoder"], flat_obs)
            embeds = flat_embed.reshape(B, T, -1)

            # RSSM rollout
            h, z = self.rssm.apply(
                params["rssm"], method=self.rssm.initial_state,
                batch_size=B,
            )
            hs, zs, prior_logits_all, post_logits_all = [], [], [], []

            for t in range(T):
                # Reset state on episode boundaries
                mask = 1.0 - is_first[:, t:t+1]
                h = h * mask
                z = z * jnp.broadcast_to(mask, z.shape)

                a = actions[:, max(0, t - 1)] if t > 0 else jnp.zeros(B, dtype=jnp.int32)
                h, prior_logits, post_logits, z = self.rssm.apply(
                    params["rssm"], h, z, a, embeds[:, t],
                )
                hs.append(h)
                zs.append(z)
                prior_logits_all.append(prior_logits)
                post_logits_all.append(post_logits)

            hs = jnp.stack(hs, axis=1)  # (B, T, hidden)
            zs = jnp.stack(zs, axis=1)  # (B, T, stoch)
            features = jnp.concatenate([hs, zs], axis=-1)  # (B, T, feat)

            # Decode
            flat_feat = features.reshape(B * T, -1)
            recon = self.decoder.apply(params["decoder"], flat_feat)
            recon = recon.reshape(B, T, *obs.shape[2:])

            # Reconstruction loss (MSE on symlog space)
            recon_loss = jnp.mean((recon - symlog(obs)) ** 2)

            # Reward prediction
            reward_pred = self.reward_pred.apply(params["reward"], flat_feat)
            reward_pred = reward_pred.reshape(B, T)
            reward_loss = jnp.mean((reward_pred - symlog(rewards)) ** 2)

            # Continue prediction
            cont_pred = self.continue_pred.apply(params["continue"], flat_feat)
            cont_pred = cont_pred.reshape(B, T)
            cont_target = 1.0 - dones
            cont_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(cont_pred, cont_target))

            # KL loss
            prior_all = jnp.stack(prior_logits_all, axis=1)  # (B, T, classes, dims)
            post_all = jnp.stack(post_logits_all, axis=1)
            kl = _kl_categorical(post_all, prior_all)
            kl = jnp.maximum(kl, self.cfg.free_nats)
            kl_loss = jnp.mean(kl)

            loss = recon_loss + reward_loss + cont_loss + self.cfg.kl_weight * kl_loss

            metrics = {"recon_loss": recon_loss, "reward_loss": reward_loss,
                       "cont_loss": cont_loss, "kl_loss": kl_loss, "wm_loss": loss}
            return loss, (metrics, (hs, zs))

        grads, (metrics, states) = jax.grad(wm_loss_fn, has_aux=True)(wm_state.params)
        wm_state = wm_state.apply_gradients(grads=grads)
        return wm_state, metrics, states

    def _train_behavior(self, wm_params, actor_state, critic_state,
                        critic_target, states, rng_key):
        hs, zs = states  # (B, T, ...)
        B, T = hs.shape[0], hs.shape[1]
        cfg = self.cfg

        # Pick random starting states from the rollout
        k1, k2 = jax.random.split(rng_key)
        t_idx = jax.random.randint(k1, (B,), 0, T)
        h0 = hs[jnp.arange(B), t_idx]
        z0 = zs[jnp.arange(B), t_idx]

        def imagine(actor_params):
            """Imagination rollout returning features, rewards, continues."""
            h, z = h0, z0
            feats, rewards, conts = [], [], []
            for _ in range(cfg.imagination_horizon):
                feat = _feature(h, z)
                feats.append(feat)
                logits = self.actor.apply(actor_params, feat)
                action = jax.random.categorical(k2, logits)
                h, prior_logits, _, z = self.rssm.apply(wm_params["rssm"], h, z, action)
                feat_next = _feature(h, z)
                r = self.reward_pred.apply(wm_params["reward"], feat_next).squeeze(-1)
                c = jax.nn.sigmoid(
                    self.continue_pred.apply(wm_params["continue"], feat_next).squeeze(-1)
                )
                rewards.append(r)
                conts.append(c)
            feats.append(_feature(h, z))  # terminal state
            return jnp.stack(feats, 1), jnp.stack(rewards, 1), jnp.stack(conts, 1)

        # Actor loss
        def actor_loss_fn(actor_params):
            feats, rewards, conts = imagine(actor_params)
            # Lambda returns
            values = self.critic.apply(critic_target, feats).squeeze(-1)  # (B, H+1)
            returns = _lambda_return(rewards, values[:, 1:], conts, cfg.discount, cfg.lambda_)

            # Actor: maximize returns + entropy
            logits_seq = jax.vmap(
                lambda f: self.actor.apply(actor_params, f)
            )(feats[:, :-1].reshape(-1, feats.shape[-1]))
            entropy = _categorical_entropy(logits_seq).mean()

            actor_loss = -returns.mean() - cfg.entropy_scale * entropy
            return actor_loss, {"actor_loss": actor_loss, "entropy": entropy,
                                "imag_reward": rewards.mean(),
                                "imag_return": returns.mean()}

        actor_grads, actor_metrics = jax.grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        # Critic loss (on stop-gradiented targets)
        def critic_loss_fn(critic_params):
            feats, rewards, conts = imagine(jax.lax.stop_gradient(actor_state.params))
            values = self.critic.apply(critic_target, feats).squeeze(-1)
            returns = _lambda_return(rewards, values[:, 1:], conts, cfg.discount, cfg.lambda_)

            v_pred = self.critic.apply(critic_params, feats[:, :-1]).squeeze(-1)
            critic_loss = jnp.mean((v_pred - jax.lax.stop_gradient(returns)) ** 2)
            return critic_loss, {"critic_loss": critic_loss}

        critic_grads, critic_metrics = jax.grad(critic_loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        # EMA update critic target
        tau = 0.02
        critic_target = jax.tree.map(
            lambda t, s: (1 - tau) * t + tau * s,
            critic_target, critic_state.params,
        )

        metrics = {**actor_metrics, **critic_metrics}
        return actor_state, critic_state, critic_target, metrics

    def save(self, path: str):
        import os, pickle
        os.makedirs(path, exist_ok=True)
        data = {
            "wm_params": self.wm_state.params,
            "actor_params": self.actor_state.params,
            "critic_params": self.critic_state.params,
            "critic_target": self.critic_target_params,
        }
        with open(os.path.join(path, "checkpoint.pkl"), "wb") as f:
            pickle.dump(jax.tree.map(lambda x: x.tolist(), data), f)

    def load(self, path: str):
        import os, pickle
        with open(os.path.join(path, "checkpoint.pkl"), "rb") as f:
            data = pickle.load(f)
        data = jax.tree.map(jnp.array, data)
        self.wm_state = self.wm_state.replace(params=data["wm_params"])
        self.actor_state = self.actor_state.replace(params=data["actor_params"])
        self.critic_state = self.critic_state.replace(params=data["critic_params"])
        self.critic_target_params = data["critic_target"]


# ------ Utilities ------

def _kl_categorical(post_logits, prior_logits):
    """KL divergence between two categorical distributions (per-class, summed)."""
    post = jax.nn.softmax(post_logits, axis=-1)
    prior = jax.nn.softmax(prior_logits, axis=-1)
    kl = post * (jnp.log(post + 1e-8) - jnp.log(prior + 1e-8))
    return kl.sum(axis=-1).sum(axis=-1)  # sum over dims and classes → (B, T)


def _lambda_return(rewards, next_values, continues, discount, lambda_):
    """Compute GAE-style lambda returns."""
    # rewards: (B, H), next_values: (B, H), continues: (B, H)
    H = rewards.shape[1]
    returns = jnp.zeros_like(rewards[:, -1])  # (B,)
    returns = next_values[:, -1]

    result = [returns]
    for t in reversed(range(H - 1)):
        returns = rewards[:, t] + discount * continues[:, t] * (
            (1 - lambda_) * next_values[:, t] + lambda_ * returns
        )
        result.append(returns)
    result = jnp.stack(list(reversed(result)), axis=1)  # (B, H)
    return result


def _categorical_entropy(logits):
    """Entropy of categorical distribution."""
    probs = jax.nn.softmax(logits, axis=-1)
    return -(probs * jnp.log(probs + 1e-8)).sum(axis=-1)
