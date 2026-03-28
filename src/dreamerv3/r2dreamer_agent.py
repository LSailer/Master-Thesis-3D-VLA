"""R2DreamerAgent — JAX/Flax port of the R2-Dreamer agent.

Single-optimizer agent with Barlow Twins representation loss, LaProp optimizer,
AGC gradient clipping, imagination-based actor-critic, and replay-based value
learning (repval) with gradients through the world model.
"""

import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from src.dreamerv3.r2dreamer_config import R2DreamerConfig
from src.dreamerv3.r2dreamer_networks import (
    R2Encoder,
    R2RSSM,
    Projector,
    ReturnEMA,
)
from src.dreamerv3.networks import MLP, TwoHotDist
from src.dreamerv3.optim import laprop, agc


# ---------------------------------------------------------------------------
# Helper: Flax module wrappers for apply calls
# ---------------------------------------------------------------------------


def _make_rssm(cfg: R2DreamerConfig) -> R2RSSM:
    return R2RSSM(
        deter_size=cfg.deter_size,
        stoch_classes=cfg.stoch_classes,
        stoch_discrete=cfg.stoch_discrete,
        num_actions=cfg.num_actions,
        hidden=cfg.hidden_size,
        blocks=cfg.blocks,
        dyn_layers=cfg.dyn_layers,
        obs_layers=cfg.obs_layers,
        img_layers=cfg.img_layers,
        unimix_ratio=cfg.unimix_ratio,
    )


def _make_encoder(cfg: R2DreamerConfig) -> R2Encoder:
    return R2Encoder(
        depth=cfg.encoder_depth,
        kernel_size=cfg.encoder_kernel,
        mults=cfg.encoder_mults,
    )


# ---------------------------------------------------------------------------
# R2DreamerAgent
# ---------------------------------------------------------------------------


class R2DreamerAgent:
    """R2-Dreamer agent with a single LaProp optimizer over all parameters.

    All Flax modules are *stateless* — parameters live in a flat pytree dict
    ``self.params``.  Training is done via ``jax.grad`` of a single loss
    function that encompasses world-model, actor-critic, and repval losses.
    """

    def __init__(self, config: R2DreamerConfig, rng_key: jnp.ndarray):
        self.cfg = config
        self.twohot = TwoHotDist(num_bins=config.twohot_bins)

        # ---- Instantiate Flax modules (for .apply) ----
        self.encoder_mod = _make_encoder(config)
        self.rssm_mod = _make_rssm(config)

        # Dummy forward to discover embed_size
        rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
        dummy_obs = jnp.zeros((1, *config.obs_shape))
        enc_params = self.encoder_mod.init(k1, dummy_obs)
        embed = self.encoder_mod.apply(enc_params, dummy_obs)
        self.embed_size = embed.shape[-1]

        # RSSM
        stoch0 = jnp.zeros((1, config.stoch_classes, config.stoch_discrete))
        deter0 = jnp.zeros((1, config.deter_size))
        action0 = jnp.zeros((1, config.num_actions))
        embed0 = jnp.zeros((1, self.embed_size))
        rssm_params = self.rssm_mod.init(k2, stoch0, deter0, action0, embed0)

        # Projector: feat_size -> embed_size
        self.proj_mod = Projector(out_dim=self.embed_size)
        feat0 = jnp.zeros((1, config.feat_size))
        proj_params = self.proj_mod.init(k3, feat0)

        # MLP heads
        rng_key, k_rew, k_con, k_act, k_cri = jax.random.split(rng_key, 5)
        self.reward_mod = MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_reward,
            out_dim=config.twohot_bins,
        )
        rew_params = self.reward_mod.init(k_rew, feat0)

        self.cont_mod = MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_cont,
            out_dim=1,
        )
        con_params = self.cont_mod.init(k_con, feat0)

        self.actor_mod = MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_actor,
            out_dim=config.num_actions,
        )
        act_params = self.actor_mod.init(k_act, feat0)

        self.critic_mod = MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_critic,
            out_dim=config.twohot_bins,
        )
        cri_params = self.critic_mod.init(k_cri, feat0)

        # ---- Bundle all params ----
        self.params = {
            "encoder": enc_params,
            "rssm": rssm_params,
            "projector": proj_params,
            "reward": rew_params,
            "cont": con_params,
            "actor": act_params,
            "critic": cri_params,
        }

        # ---- Optimizer: LaProp (constant lr, no warmup for v1) ----
        self.tx = laprop(
            lr=config.lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        )
        self.opt_state = self.tx.init(self.params)

        # ---- Slow target critic (EMA) ----
        self.slow_critic_params = jax.tree.map(jnp.copy, self.params["critic"])

        # ---- Return EMA ----
        self.return_ema = ReturnEMA()
        self.ema_state = self.return_ema.init_state()

        # ---- Acting state (for single-env stepping) ----
        self._act_stoch = np.zeros(
            (1, config.stoch_classes, config.stoch_discrete), dtype=np.float32
        )
        self._act_deter = np.zeros((1, config.deter_size), dtype=np.float32)
        self._act_prev_action = np.zeros((1, config.num_actions), dtype=np.float32)

        # ---- JIT-compiled functions ----
        self._jit_train_step = jax.jit(self._train_step)
        self._jit_act = jax.jit(self._act_jit)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, obs_dict: Dict[str, Any], rng_key: jnp.ndarray, training: bool = True) -> int:
        """Select an action for a single environment step.

        Args:
            obs_dict: {"image": uint8 array (C,H,W), "is_first": bool}
            rng_key: PRNG key
            training: if False, use argmax (greedy)

        Returns:
            Integer action in [0, num_actions).
        """
        # Preprocess observation
        image = obs_dict["image"].astype(np.float32) / 255.0
        obs = jnp.array(image[None])  # (1, C, H, W)

        is_first = bool(obs_dict["is_first"])
        if is_first:
            self._act_stoch = np.zeros_like(self._act_stoch)
            self._act_deter = np.zeros_like(self._act_deter)
            self._act_prev_action = np.zeros_like(self._act_prev_action)

        stoch = jnp.array(self._act_stoch)
        deter = jnp.array(self._act_deter)
        prev_action = jnp.array(self._act_prev_action)

        action_int, new_stoch, new_deter = self._jit_act(
            self.params, obs, stoch, deter, prev_action, rng_key, training
        )
        action_int = int(action_int)

        # Update acting state
        self._act_stoch = np.array(new_stoch)
        self._act_deter = np.array(new_deter)
        self._act_prev_action = np.zeros(
            (1, self.cfg.num_actions), dtype=np.float32
        )
        self._act_prev_action[0, action_int] = 1.0

        return action_int

    def _act_jit(self, params, obs, stoch, deter, prev_action, rng_key, training):
        """JIT-able acting logic.  Returns (action_int, new_stoch, new_deter)."""
        embed = self.encoder_mod.apply(params["encoder"], obs)
        new_stoch, new_deter, _ = self.rssm_mod.apply(
            params["rssm"], stoch, deter, prev_action, embed
        )
        feat = self.rssm_mod.apply(
            params["rssm"], new_stoch, new_deter, method=self.rssm_mod.get_feat
        )
        logits = self.actor_mod.apply(params["actor"], feat)  # (1, num_actions)

        def _sample(logits, rng_key):
            return jax.random.categorical(rng_key, logits, axis=-1)[0]

        def _greedy(logits, _rng_key):
            return jnp.argmax(logits, axis=-1)[0]

        action_int = jax.lax.cond(training, _sample, _greedy, logits, rng_key)
        return action_int, new_stoch, new_deter

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, jnp.ndarray], rng_key: jnp.ndarray) -> Dict[str, float]:
        """Perform one gradient step on the given batch.

        Args:
            batch: dict with keys obs (B,T,C,H,W), actions (B,T,A),
                   rewards (B,T), is_first (B,T), is_last (B,T),
                   is_terminal (B,T).
            rng_key: PRNG key.

        Returns:
            Metrics dict (Python floats).
        """
        (
            self.params,
            self.opt_state,
            self.slow_critic_params,
            self.ema_state,
            metrics,
        ) = self._jit_train_step(
            self.params,
            self.opt_state,
            self.slow_critic_params,
            self.ema_state,
            batch,
            rng_key,
        )
        return {k: float(v) for k, v in metrics.items()}

    def _train_step(self, params, opt_state, slow_critic_params, ema_state, batch, rng_key):
        """Pure-functional training step (JIT-able)."""

        loss_fn = functools.partial(
            self._loss_fn,
            slow_critic_params=slow_critic_params,
            ema_state=ema_state,
            batch=batch,
            rng_key=rng_key,
        )

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Adaptive gradient clipping
        grads = agc(grads, params, clip=self.cfg.agc_clip, pmin=self.cfg.agc_pmin)

        # Optimizer step
        updates, new_opt_state = self.tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Slow critic EMA update
        tau = self.cfg.slow_target_fraction
        new_slow = jax.tree.map(
            lambda s, p: tau * p + (1 - tau) * s,
            slow_critic_params,
            new_params["critic"],
        )

        # Return EMA update
        metrics = aux["metrics"]
        new_ema_state = ema_state  # default: don't update if no returns
        # Update EMA from imagination returns stored in aux
        imag_returns = aux["imag_returns"]
        new_ema_state = self.return_ema.update(ema_state, imag_returns)

        metrics["total_loss"] = total_loss
        return new_params, new_opt_state, new_slow, new_ema_state, metrics

    def _loss_fn(self, params, *, slow_critic_params, ema_state, batch, rng_key):
        """Compute all losses in one function for jax.grad.

        Returns:
            (total_loss, aux_dict)
        """
        cfg = self.cfg
        B, T = batch["obs"].shape[0], batch["obs"].shape[1]
        metrics = {}
        losses = {}

        # ----------------------------------------------------------
        # 1. World model: encode observations, posterior rollout, KL
        # ----------------------------------------------------------
        embed = self.encoder_mod.apply(params["encoder"], batch["obs"].reshape(B * T, *cfg.obs_shape))
        embed = embed.reshape(B, T, -1)  # (B, T, embed_size)

        stoch0, deter0 = self.rssm_mod.apply(
            params["rssm"], B, method=self.rssm_mod.initial_state
        )
        post_stochs, post_deters, post_logits = self.rssm_mod.apply(
            params["rssm"], embed, batch["actions"], (stoch0, deter0),
            batch["is_first"], method=self.rssm_mod.observe,
        )
        # (B, T, stoch_classes, stoch_discrete)

        # Prior logits from posterior deters
        def _prior_fn(deter_flat):
            _, logit = self.rssm_mod.apply(
                params["rssm"], deter_flat, method=self.rssm_mod.prior
            )
            return logit

        prior_logits = _prior_fn(post_deters.reshape(B * T, -1))
        prior_logits = prior_logits.reshape(B, T, cfg.stoch_classes, cfg.stoch_discrete)

        # KL losses (free-nats clipping)
        post_logits_flat = post_logits.reshape(B * T, cfg.stoch_classes, cfg.stoch_discrete)
        prior_logits_flat = prior_logits.reshape(B * T, cfg.stoch_classes, cfg.stoch_discrete)

        dyn_loss, rep_loss = _kl_loss(
            post_logits_flat, prior_logits_flat,
            cfg.stoch_classes, cfg.stoch_discrete, cfg.kl_free
        )
        losses["dyn"] = jnp.mean(dyn_loss)
        losses["rep"] = jnp.mean(rep_loss)

        # Feature vector
        feat = self.rssm_mod.apply(
            params["rssm"], post_stochs, post_deters,
            method=self.rssm_mod.get_feat
        )  # (B, T, feat_size)

        # ----------------------------------------------------------
        # 2. Barlow Twins representation loss
        # ----------------------------------------------------------
        feat_flat = feat.reshape(B * T, -1)
        x1 = self.proj_mod.apply(params["projector"], feat_flat)  # (BT, embed_size)
        x2 = jax.lax.stop_gradient(embed.reshape(B * T, -1))  # stop grad on encoder side

        x1_norm = (x1 - jnp.mean(x1, axis=0)) / (jnp.std(x1, axis=0) + 1e-8)
        x2_norm = (x2 - jnp.mean(x2, axis=0)) / (jnp.std(x2, axis=0) + 1e-8)

        c = (x1_norm.T @ x2_norm) / (B * T)  # (embed_size, embed_size)
        invariance_loss = jnp.sum((jnp.diag(c) - 1.0) ** 2)
        # Off-diagonal: mask the diagonal
        off_diag_mask = 1.0 - jnp.eye(c.shape[0])
        redundancy_loss = jnp.sum((c * off_diag_mask) ** 2)
        losses["barlow"] = invariance_loss + cfg.barlow_lambda * redundancy_loss

        # ----------------------------------------------------------
        # 3. Reward and continue losses
        # ----------------------------------------------------------
        rew_logits = self.reward_mod.apply(params["reward"], feat_flat)
        rew_logits = rew_logits.reshape(B, T, -1)
        losses["rew"] = jnp.mean(self.twohot.loss(rew_logits, batch["rewards"]))

        cont_logits = self.cont_mod.apply(params["cont"], feat_flat)
        cont_logits = cont_logits.reshape(B, T, 1)
        cont_target = 1.0 - batch["is_terminal"]  # (B, T)
        losses["con"] = jnp.mean(
            optax.sigmoid_binary_cross_entropy(
                cont_logits[..., 0], cont_target
            )
        )

        # ----------------------------------------------------------
        # 4. Imagination rollout (actor-critic)
        # ----------------------------------------------------------
        rng_key, imag_key = jax.random.split(rng_key)

        # Start from ALL posterior states, detached from world model
        start_stoch = jax.lax.stop_gradient(
            post_stochs.reshape(B * T, cfg.stoch_classes, cfg.stoch_discrete)
        )
        start_deter = jax.lax.stop_gradient(
            post_deters.reshape(B * T, cfg.deter_size)
        )

        horizon = cfg.imagination_horizon + 1
        imag_feats, imag_actions, imag_rng_keys = _imagine(
            params["rssm"], params["actor"],
            self.rssm_mod, self.actor_mod,
            start_stoch, start_deter,
            horizon, imag_key,
        )
        # imag_feats: (BT, H, feat_size), imag_actions: (BT, H, num_actions)

        # Reward and cont predictions on imagined features (frozen = stop_gradient)
        imag_feat_flat = imag_feats.reshape(B * T * horizon, -1)
        imag_rew_logits = self.reward_mod.apply(
            jax.lax.stop_gradient(params["reward"]), imag_feat_flat
        )
        imag_reward = self.twohot.pred(imag_rew_logits).reshape(B * T, horizon, 1)

        imag_cont_logits = self.cont_mod.apply(
            jax.lax.stop_gradient(params["cont"]), imag_feat_flat
        )
        imag_cont = jax.nn.sigmoid(imag_cont_logits).reshape(B * T, horizon, 1)

        imag_val_logits = self.critic_mod.apply(
            jax.lax.stop_gradient(params["critic"]), imag_feat_flat
        )
        imag_value = self.twohot.pred(imag_val_logits).reshape(B * T, horizon, 1)

        imag_slow_logits = self.critic_mod.apply(
            slow_critic_params, imag_feat_flat
        )
        imag_slow_value = self.twohot.pred(imag_slow_logits).reshape(B * T, horizon, 1)

        disc = 1.0 - 1.0 / cfg.horizon
        weight = jnp.cumprod(imag_cont * disc, axis=1)

        last = jnp.zeros_like(imag_cont)
        term = 1.0 - imag_cont
        ret = _lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, cfg.lamb
        )  # (BT, H-1, 1)

        # Advantage
        ret_offset, ret_scale = self.return_ema.get_stats(ema_state)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        # Actor loss: REINFORCE + entropy bonus
        # Compute actor logits on imagined features (with gradient through actor)
        actor_logits = self.actor_mod.apply(
            params["actor"], imag_feats.reshape(B * T * horizon, -1)
        ).reshape(B * T, horizon, cfg.num_actions)

        log_probs = jax.nn.log_softmax(actor_logits, axis=-1)  # (BT, H, A)
        # log_prob of taken action: sum(one_hot * log_softmax)
        logpi = jnp.sum(log_probs[:, :-1] * imag_actions[:, :-1], axis=-1, keepdims=True)

        # Entropy: -sum(p * log p)
        probs = jax.nn.softmax(actor_logits[:, :-1], axis=-1)
        entropy = -jnp.sum(probs * log_probs[:, :-1], axis=-1, keepdims=True)

        losses["policy"] = jnp.mean(
            jax.lax.stop_gradient(weight[:, :-1])
            * -(logpi * jax.lax.stop_gradient(adv) + cfg.act_entropy * entropy)
        )

        # Critic loss on imagined features
        cri_logits_imag = self.critic_mod.apply(
            params["critic"], imag_feats.reshape(B * T * horizon, -1)
        ).reshape(B * T, horizon, cfg.twohot_bins)

        tar_padded = jnp.concatenate(
            [ret, jnp.zeros_like(ret[:, -1:])], axis=1
        )  # (BT, H, 1)

        critic_loss_tar = self.twohot.loss(cri_logits_imag[:, :-1], tar_padded[:, :-1, 0])
        critic_loss_slow = self.twohot.loss(
            cri_logits_imag[:, :-1],
            jax.lax.stop_gradient(imag_slow_value[:, :-1, 0]),
        )
        losses["value"] = jnp.mean(
            jax.lax.stop_gradient(weight[:, :-1, 0])
            * (critic_loss_tar + critic_loss_slow)
        )

        # ----------------------------------------------------------
        # 5. Replay-based value learning (repval)
        # ----------------------------------------------------------
        # Use feat WITH gradients through world model
        replay_last = batch["is_last"]   # (B, T)
        replay_term = batch["is_terminal"]  # (B, T)
        replay_reward = batch["rewards"]  # (B, T)

        # Bootstrap from imagination returns at first step
        boot = ret[:, 0].reshape(B, T, 1)

        # Value predictions (frozen) for lambda return computation
        replay_val_logits = self.critic_mod.apply(
            jax.lax.stop_gradient(params["critic"]),
            feat.reshape(B * T, -1),
        ).reshape(B, T, cfg.twohot_bins)
        replay_value = self.twohot.pred(replay_val_logits)[..., None]  # (B, T, 1)

        replay_slow_logits = self.critic_mod.apply(
            slow_critic_params,
            feat.reshape(B * T, -1),
        ).reshape(B, T, cfg.twohot_bins)
        replay_slow_value = self.twohot.pred(replay_slow_logits)[..., None]

        replay_ret = _lambda_return(
            replay_last[..., None], replay_term[..., None],
            replay_reward[..., None], replay_value, boot, disc, cfg.lamb,
        )  # (B, T-1, 1)
        ret_padded_replay = jnp.concatenate(
            [replay_ret, jnp.zeros_like(replay_ret[:, -1:])], axis=1
        )

        # Critic on replay features (WITH world model gradients)
        repval_logits = self.critic_mod.apply(
            params["critic"], feat.reshape(B * T, -1)
        ).reshape(B, T, cfg.twohot_bins)

        repval_weight = 1.0 - replay_last  # (B, T)
        repval_loss_tar = self.twohot.loss(
            repval_logits[:, :-1],
            jax.lax.stop_gradient(ret_padded_replay[:, :-1, 0]),
        )
        repval_loss_slow = self.twohot.loss(
            repval_logits[:, :-1],
            jax.lax.stop_gradient(replay_slow_value[:, :-1, 0]),
        )
        losses["repval"] = jnp.mean(
            repval_weight[:, :-1] * (repval_loss_tar + repval_loss_slow)
        )

        # ----------------------------------------------------------
        # 6. Total loss with scales
        # ----------------------------------------------------------
        total_loss = (
            cfg.scale_dyn * losses["dyn"]
            + cfg.scale_rep * losses["rep"]
            + cfg.scale_barlow * losses["barlow"]
            + cfg.scale_rew * losses["rew"]
            + cfg.scale_con * losses["con"]
            + cfg.scale_policy * losses["policy"]
            + cfg.scale_value * losses["value"]
            + cfg.scale_repval * losses["repval"]
        )

        # Metrics dict
        for k, v in losses.items():
            metrics[f"loss/{k}"] = v

        aux = {
            "metrics": metrics,
            "imag_returns": ret.reshape(-1),  # flat for percentile computation
        }
        return total_loss, aux


# ---------------------------------------------------------------------------
# Standalone pure functions (used inside JIT)
# ---------------------------------------------------------------------------


def _kl_loss(post_logits, prior_logits, stoch_classes, stoch_discrete, kl_free):
    """Compute DreamerV3-style KL losses with free nats.

    Args:
        post_logits: (N, C, K) posterior logits
        prior_logits: (N, C, K) prior logits
        kl_free: free bits threshold

    Returns:
        dyn_loss: (N,) — KL(sg(post) || prior), clipped
        rep_loss: (N,) — KL(post || sg(prior)), clipped
    """
    post_probs = jax.nn.softmax(post_logits, axis=-1)
    prior_probs = jax.nn.softmax(prior_logits, axis=-1)

    post_log = jnp.log(post_probs + 1e-8)
    prior_log = jnp.log(prior_probs + 1e-8)

    # KL(post || prior) = sum(post * (log post - log prior))
    def _kl(p, logp, logq):
        return jnp.sum(p * (logp - logq), axis=-1)  # (N, C)

    kl_post_prior = _kl(post_probs, post_log, prior_log)  # (N, C)
    kl_val = jnp.sum(kl_post_prior, axis=-1)  # (N,) sum over classes

    # dyn_loss: train prior toward frozen posterior
    sg_post_probs = jax.lax.stop_gradient(post_probs)
    sg_post_log = jax.lax.stop_gradient(post_log)
    kl_dyn = jnp.sum(_kl(sg_post_probs, sg_post_log, prior_log), axis=-1)
    dyn_loss = jnp.maximum(kl_dyn, kl_free)

    # rep_loss: train posterior toward frozen prior
    sg_prior_log = jax.lax.stop_gradient(prior_log)
    kl_rep = jnp.sum(_kl(post_probs, post_log, sg_prior_log), axis=-1)
    rep_loss = jnp.maximum(kl_rep, kl_free)

    return dyn_loss, rep_loss


def _imagine(rssm_params, actor_params, rssm_mod, actor_mod,
             start_stoch, start_deter, horizon, rng_key):
    """Imagination rollout in latent space.

    All operations use stop_gradient on world model params — only the actor
    gets gradients for the policy loss.

    Args:
        rssm_params: RSSM parameters (will be stop_gradient'd for img_step)
        actor_params: Actor parameters (gradients flow through)
        rssm_mod: R2RSSM module
        actor_mod: MLP module for actor
        start_stoch: (N, C, K) starting stochastic state
        start_deter: (N, D) starting deterministic state
        horizon: number of steps to imagine
        rng_key: PRNG key

    Returns:
        feats: (N, horizon, feat_size)
        actions: (N, horizon, num_actions)
    """
    # Use frozen RSSM for imagination
    frozen_rssm_params = jax.lax.stop_gradient(rssm_params)

    stoch = start_stoch
    deter = start_deter
    feats = []
    actions = []

    for step in range(horizon):
        feat = rssm_mod.apply(
            frozen_rssm_params, stoch, deter, method=rssm_mod.get_feat
        )

        # Actor with gradient (for policy loss)
        logits = actor_mod.apply(actor_params, feat)
        rng_key, k = jax.random.split(rng_key)
        action_idx = jax.random.categorical(k, logits, axis=-1)
        action = jax.nn.one_hot(action_idx, logits.shape[-1])
        # Straight-through for gradient: action = one_hot + softmax - sg(softmax)
        soft = jax.nn.softmax(logits, axis=-1)
        action = action + soft - jax.lax.stop_gradient(soft)

        feats.append(feat)
        actions.append(action)

        stoch, deter = rssm_mod.apply(
            frozen_rssm_params, stoch, deter, action, method=rssm_mod.img_step
        )

    return jnp.stack(feats, axis=1), jnp.stack(actions, axis=1), None


def _lambda_return(last, term, reward, value, boot, disc, lamb):
    """Compute lambda-returns (generalized advantage estimation target).

    All inputs: (..., T, 1).
    Returns: (..., T-1, 1).
    """
    live = (1.0 - term)[..., 1:, :] * disc
    cont = (1.0 - last)[..., 1:, :] * lamb
    interm = reward[..., 1:, :] + (1.0 - cont) * live * boot[..., 1:, :]
    T_minus_1 = live.shape[-2]

    # Backward scan
    def _scan_fn(carry, i):
        # i counts from 0 to T_minus_1 - 1, but we want reversed order
        idx = T_minus_1 - 1 - i
        val = interm[..., idx, :] + live[..., idx, :] * cont[..., idx, :] * carry
        return val, val

    init = boot[..., -1, :]
    _, outs = jax.lax.scan(
        _scan_fn, init, jnp.arange(T_minus_1)
    )
    # outs: (T_minus_1, ..., 1) — need to reverse and transpose
    # jax.lax.scan stacks along axis 0, we need to reverse since we computed backwards
    outs = jnp.flip(outs, axis=0)
    # Move time axis: (T-1, ..., 1) -> (..., T-1, 1)
    # For (..., T, 1) input where ... can be (B,) or (B*T,), the scan output is (T-1, ..., 1)
    # We need to move axis 0 to the second-to-last position
    ndim = outs.ndim
    axes = list(range(1, ndim - 1)) + [0, ndim - 1]
    outs = jnp.transpose(outs, axes)
    return outs
