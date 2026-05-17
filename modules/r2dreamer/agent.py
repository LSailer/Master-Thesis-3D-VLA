"""R2DreamerAgent — composition root.

The agent is a thin orchestrator: it owns parameters, the LaProp optimizer,
the slow-target EMA, the acting state, and the JIT'd train/eval entry points.
The actual loss math lives in three subpackages, each with its own loss file:

    world_model/loss.py     — KL (dyn + rep) + reward + continue heads
    behavior/loss.py        — imagination rollout, actor + critic losses
    representation/loss.py  — Barlow Twins + replay-based value learning

A single shared forward pass (`_world_model_forward`) computes `embed`, the
RSSM posterior states, prior logits, and `feat`. Those tensors thread into
all three sub-loss functions so the encoder/RSSM receive the correct combined
gradient signal under one `jax.grad`.
"""

import functools
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import R2DreamerConfig
from .world_model.rssm import R2RSSM
from .world_model.encoders import ConvEncoder
from .world_model.heads import R2MLP, R2TwoHotDist
from .world_model.loss import world_model_loss, kl_loss as _kl_loss
from .behavior.return_ema import ReturnEMA
from .behavior.imagination import _imagine, _lambda_return
from .behavior.loss import behavior_loss
from .representation.barlow import Projector
from .representation.loss import representation_loss
from modules.shared.optim import laprop, agc

# Re-export internal helpers so test_cross_framework.py keeps working.
__all__ = ["R2DreamerAgent", "_kl_loss", "_lambda_return", "_imagine"]


# ---------------------------------------------------------------------------
# Module factories
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


def _make_encoder(cfg: R2DreamerConfig):
    cls = cfg.encoder_module_cls
    if cls is ConvEncoder:
        return cls(
            depth=cfg.encoder_depth,
            kernel_size=cfg.encoder_kernel,
            mults=cfg.encoder_mults,
        )
    return cls(embed_dim=cfg.vggt_embed_dim)


# ---------------------------------------------------------------------------
# R2DreamerAgent
# ---------------------------------------------------------------------------


class R2DreamerAgent:
    """R2-Dreamer agent with a single LaProp optimizer over all parameters.

    All Flax modules are *stateless* — parameters live in a flat pytree dict
    ``self.params``.  Training is done via ``jax.grad`` of a single loss
    function (``_loss_fn``) that composes the world-model, behavior, and
    representation sub-losses.
    """

    def __init__(self, config: R2DreamerConfig, rng_key: jnp.ndarray):
        self.cfg = config
        self.twohot = R2TwoHotDist(num_bins=config.twohot_bins)

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
        rng_key, k_sample = jax.random.split(rng_key)
        rssm_params = self.rssm_mod.init(
            {"params": k2, "sample": k_sample}, stoch0, deter0, action0, embed0)

        # Projector: feat_size -> embed_size
        self.proj_mod = Projector(out_dim=self.embed_size)
        feat0 = jnp.zeros((1, config.feat_size))
        proj_params = self.proj_mod.init(k3, feat0)

        # MLP heads (outscale matches PyTorch: 0.0 for reward/critic, 0.01 for actor)
        rng_key, k_rew, k_con, k_act, k_cri = jax.random.split(rng_key, 5)
        self.reward_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_reward,
            out_dim=config.twohot_bins,
            outscale=0.0,
        )
        rew_params = self.reward_mod.init(k_rew, feat0)

        self.cont_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_cont,
            out_dim=1,
        )
        con_params = self.cont_mod.init(k_con, feat0)

        self.actor_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_actor,
            out_dim=config.num_actions,
            outscale=0.01,
        )
        act_params = self.actor_mod.init(k_act, feat0)

        self.critic_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_critic,
            out_dim=config.twohot_bins,
            outscale=0.0,
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

        # Module bundle passed to sub-loss functions
        self._modules = {
            "encoder": self.encoder_mod,
            "rssm": self.rssm_mod,
            "projector": self.proj_mod,
            "reward": self.reward_mod,
            "cont": self.cont_mod,
            "actor": self.actor_mod,
            "critic": self.critic_mod,
        }

        # ---- Optimizer: LaProp with linear warmup ----
        self.tx = laprop(
            lr=config.lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            warmup=config.warmup_steps,
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
        self._jit_eval_loss = jax.jit(self._eval_loss_fn)
        self._jit_act = jax.jit(self._act_jit)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, obs_dict: Dict[str, Any], rng_key: jnp.ndarray, training: bool = True) -> int:
        """Select an action for a single environment step.

        Args:
            obs_dict: {"image": uint8 (C,H,W), "is_first": bool} for CNN, or
                      {"features": float32 (D,), "is_first": bool} for VGGT.
            rng_key: PRNG key.
            training: if False, use argmax (greedy).

        Returns:
            Integer action in [0, num_actions).
        """
        if self.cfg.encoder_type in ("vggt", "vggt_aggregator_mlp"):
            obs = jnp.asarray(obs_dict["features"])[None]
        else:
            image = obs_dict["image"].astype(np.float32) / 255.0
            obs = jnp.array(image[None])

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

        self._act_stoch = np.array(new_stoch)
        self._act_deter = np.array(new_deter)
        self._act_prev_action = np.zeros(
            (1, self.cfg.num_actions), dtype=np.float32
        )
        self._act_prev_action[0, action_int] = 1.0

        return action_int

    def _act_jit(self, params, obs, stoch, deter, prev_action, rng_key, training):
        """JIT-able acting logic. Returns (action_int, new_stoch, new_deter)."""
        embed = self.encoder_mod.apply(params["encoder"], obs)
        rng_key, k_sample = jax.random.split(rng_key)
        new_stoch, new_deter, _ = self.rssm_mod.apply(
            params["rssm"], stoch, deter, prev_action, embed,
            rngs={"sample": k_sample},
        )
        feat = self.rssm_mod.apply(
            params["rssm"], new_stoch, new_deter, method=self.rssm_mod.get_feat
        )
        logits = self.actor_mod.apply(params["actor"], feat)

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
        """One LaProp step on `batch`. Returns Python-float metrics."""
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

        # Slow critic EMA: update BEFORE loss (matches PyTorch _update_slow_target)
        tau = self.cfg.slow_target_fraction
        updated_slow = jax.tree.map(
            lambda s, p: tau * p + (1 - tau) * s,
            slow_critic_params,
            params["critic"],
        )

        loss_fn = functools.partial(
            self._loss_fn,
            slow_critic_params=updated_slow,
            ema_state=ema_state,
            batch=batch,
            rng_key=rng_key,
        )

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # NaN guard: skip update if loss is non-finite (mirrors PyTorch GradScaler)
        is_finite = jnp.isfinite(total_loss)

        grads = agc(grads, params, clip=self.cfg.agc_clip, pmin=self.cfg.agc_pmin)
        updates, new_opt_state = self.tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_ema_state = self.return_ema.update(ema_state, aux["imag_returns"])

        # Roll back to pre-update state on NaN/inf
        new_params = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), new_params, params)
        new_opt_state = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), new_opt_state, opt_state)
        new_slow = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), updated_slow, slow_critic_params)
        new_ema_state = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), new_ema_state, ema_state)

        metrics = aux["metrics"]
        metrics["total_loss"] = total_loss
        metrics["nan_skipped"] = 1.0 - is_finite.astype(jnp.float32)
        return new_params, new_opt_state, new_slow, new_ema_state, metrics

    def eval_loss(self, batch: Dict[str, jnp.ndarray], rng_key: jnp.ndarray) -> Dict[str, float]:
        """Forward-only loss for validation. Same metrics as `train_step`."""
        metrics = self._jit_eval_loss(
            self.params, self.slow_critic_params, self.ema_state, batch, rng_key,
        )
        return {k: float(v) for k, v in metrics.items()}

    def _eval_loss_fn(self, params, slow_critic_params, ema_state, batch, rng_key):
        total_loss, aux = self._loss_fn(
            params,
            slow_critic_params=slow_critic_params,
            ema_state=ema_state,
            batch=batch,
            rng_key=rng_key,
        )
        metrics = aux["metrics"]
        metrics["total_loss"] = total_loss
        return metrics

    # ------------------------------------------------------------------
    # Composition root: shared forward + 3 sub-losses
    # ------------------------------------------------------------------

    def _world_model_forward(self, params, batch, rng_key):
        """Encoder + posterior rollout + prior + features. Shared across sub-losses.

        Computing this once is essential: if each sub-loss recomputed `embed`,
        the encoder would receive doubled gradient signal and the
        `barlow_stop_grad` toggle would no longer mean what it claims.
        """
        cfg = self.cfg
        B, T = batch["obs"].shape[0], batch["obs"].shape[1]

        obs_flat = batch["obs"].reshape(B * T, *cfg.obs_shape)
        embed = self.encoder_mod.apply(params["encoder"], obs_flat).reshape(B, T, -1)

        stoch0, deter0 = self.rssm_mod.apply(
            params["rssm"], B, method=self.rssm_mod.initial_state)

        rng_key, k_obs = jax.random.split(rng_key)
        post_stochs, post_deters, post_logits = self.rssm_mod.apply(
            params["rssm"], embed, batch["actions"], (stoch0, deter0),
            batch["is_first"], method=self.rssm_mod.observe,
            rngs={"sample": k_obs},
        )

        rng_key, k_prior = jax.random.split(rng_key)
        _, prior_logits_flat = self.rssm_mod.apply(
            params["rssm"], post_deters.reshape(B * T, -1),
            method=self.rssm_mod.prior,
            rngs={"sample": k_prior},
        )
        prior_logits = prior_logits_flat.reshape(
            B, T, cfg.stoch_classes, cfg.stoch_discrete)

        feat = self.rssm_mod.apply(
            params["rssm"], post_stochs, post_deters, method=self.rssm_mod.get_feat,
        )

        return {
            "embed": embed,
            "post_stochs": post_stochs,
            "post_deters": post_deters,
            "post_logits": post_logits,
            "prior_logits": prior_logits,
            "feat": feat,
        }

    def _loss_fn(self, params, *, slow_critic_params, ema_state, batch, rng_key):
        """Compose the world-model, behavior, and representation losses.

        Returns:
            (total_loss, aux) — `aux` carries metrics and the imagination
            returns used for the post-step `ReturnEMA` update.
        """
        cfg = self.cfg
        B, T = batch["obs"].shape[0], batch["obs"].shape[1]

        rng_key, k_fwd = jax.random.split(rng_key)
        forward = self._world_model_forward(params, batch, k_fwd)

        wm_losses, wm_metrics = world_model_loss(
            forward=forward, params=params, batch=batch,
            modules=self._modules, cfg=cfg, twohot=self.twohot,
        )

        rng_key, k_behavior = jax.random.split(rng_key)
        bh_losses, bh_metrics, imag_ret = behavior_loss(
            forward=forward, params=params, modules=self._modules,
            cfg=cfg, twohot=self.twohot,
            slow_critic_params=slow_critic_params, ema_state=ema_state,
            return_ema=self.return_ema, rng_key=k_behavior,
            B=B, T=T,
        )

        rep_losses, rep_metrics = representation_loss(
            forward=forward, batch=batch, params=params, modules=self._modules,
            cfg=cfg, twohot=self.twohot,
            slow_critic_params=slow_critic_params, imag_ret=imag_ret,
            B=B, T=T,
        )

        losses = {**wm_losses, **bh_losses, **rep_losses}
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

        # ---- Metrics ----
        metrics = {**wm_metrics, **bh_metrics, **rep_metrics}
        for k, v in losses.items():
            metrics[f"loss/{k}"] = v

        # Encoder L2 — Protocol D diagnostic for whether Barlow's gradient
        # toggle is actually moving the encoder weights.
        enc_sq = jax.tree_util.tree_reduce(
            lambda acc, x: acc + jnp.sum(jnp.square(x)),
            params["encoder"], 0.0,
        )
        metrics["params/encoder_l2"] = jnp.sqrt(enc_sq)

        aux = {"metrics": metrics, "imag_returns": imag_ret.reshape(-1)}
        return total_loss, aux
