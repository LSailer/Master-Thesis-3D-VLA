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
import pickle
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import R2DreamerConfig
from .world_model.rssm import R2RSSM
from .world_model.encoders import (
    ConvEncoder,
    WPConvEncoder,
    ConvDecoder,
    HybridEncoder as WMHybridEncoder,
    VGGTEncoder as WMVGGTEncoder,
    VGGTAggTokenTransformerEncoder as WMVGGTAggTokenTransformerEncoder,
    VGGTAggregatorMLPEncoder as WMVGGTAggregatorMLPEncoder,
    HYBRID_RGB_DIM,
)
from .world_model.heads import R2MLP, R2TwoHotDist
from .world_model.loss import world_model_loss, kl_loss as _kl_loss
from .behavior.return_ema import ReturnEMA
from .behavior.imagination import _imagine, _lambda_return
from .behavior.loss import behavior_loss
from .representation.barlow import Projector
from .representation.loss import representation_loss
from .obs_batch import (
    decoder_rgb_target,
    encoder_obs_from_agent_obs,
    encoder_obs_from_batch,
    obs_leading_shape,
)
from src.shared.optim import laprop, agc

# Re-export internal helpers so test_cross_framework.py keeps working.
__all__ = [
    "R2DreamerAgent",
    "load_policy_checkpoint",
    "_kl_loss",
    "_lambda_return",
    "_imagine",
]


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


def _missing_pickle_class(module: str, name: str) -> type:
    class MissingPickleClass:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __setstate__(self, state):
            self.state = state

    MissingPickleClass.__name__ = name
    MissingPickleClass.__module__ = module
    return MissingPickleClass


class _CheckpointUnpickler(pickle.Unpickler):
    """Load old checkpoints even when unused optimizer-state classes moved."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return _missing_pickle_class(module, name)


def load_policy_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load an R2DreamerAgent checkpoint, tolerating moved optimizer classes."""
    path = Path(path)
    with path.open("rb") as f:
        ckpt = _CheckpointUnpickler(f).load()
    missing = {"params", "slow_critic_params"} - set(ckpt)
    if missing:
        raise KeyError(f"checkpoint {path} is missing required keys: {sorted(missing)}")
    return ckpt


def _resolve_encoder_cls(cfg: R2DreamerConfig):
    # Launcher-created configs pass EncoderSpec.module_cls explicitly. Unit tests
    # and direct R2DreamerConfig() construction rely on encoder_type, so map the
    # documented names to their Flax modules when no class is supplied.
    cls = cfg.encoder_module_cls
    if cls is None:
        cls = {
            "cnn": ConvEncoder,
            "vggt": WMVGGTEncoder,
            "vggt_wp_cp_64": WMVGGTEncoder,  # same MLP module, finer WP grid (obs 12297)
            "vggt_aggregator_mlp": WMVGGTAggregatorMLPEncoder,
            "vggt_agg_token_transformer": WMVGGTAggTokenTransformerEncoder,
            "vggt_wp_dense_cnn": WPConvEncoder,
            "hybrid": WMHybridEncoder,
            "vggt_house_context": WMHybridEncoder,
        }.get(cfg.encoder_type)
        if cls is None:
            raise ValueError(f"unknown encoder_type {cfg.encoder_type!r}")
    return cls


def _validate_encoder_config(cfg: R2DreamerConfig, cls) -> None:
    if cls in (ConvEncoder, WPConvEncoder) and cfg.vggt_mlp_layers != 1:
        # Fail loud instead of silently dropping the knob: conv encoders have no
        # MLP depth, so a non-default vggt_mlp_layers here is a misconfiguration.
        raise ValueError(
            f"vggt_mlp_layers={cfg.vggt_mlp_layers} has no effect on "
            f"{cls.__name__} (a conv encoder, no MLP blocks). Only the 'vggt' and "
            f"'vggt_aggregator_mlp' encoders consume vggt_mlp_layers; leave it at 1 "
            f"for cnn / vggt_wp_dense_cnn."
        )


def _make_conv_encoder(cfg: R2DreamerConfig):
    return ConvEncoder(
        depth=cfg.encoder_depth,
        kernel_size=cfg.encoder_kernel,
        mults=cfg.encoder_mults,
    )


def _make_wp_conv_encoder(cfg: R2DreamerConfig):
    # Full-res world-point map -> conv stack -> embed_dim (3D-53). Reuses the
    # RGB conv hyperparameters; symlog (not /255) handles the metric XYZ range.
    return WPConvEncoder(
        embed_dim=cfg.vggt_embed_dim,
        depth=cfg.encoder_depth,
        kernel_size=cfg.encoder_kernel,
        mults=cfg.encoder_mults,
    )


def _make_hybrid_encoder(cfg: R2DreamerConfig):
    # CNN(RGB) + gated MLP(WP/CP) fused into one embed (3D-50/51/52).
    # Guard the packed encoder-layout contract: replay may store modalities in
    # separate fields, but obs_batch packs them into this flat shape before the
    # Flax encoder and decoder see them.
    expected_shape = (HYBRID_RGB_DIM + cfg.vggt_feature_dim,)
    if not (
        cfg.obs_shape == expected_shape
        and cfg.obs_shape[0] - cfg.vggt_feature_dim == HYBRID_RGB_DIM
    ):
        raise ValueError(
            "hybrid obs_shape/split mismatch: expected "
            f"{expected_shape} with vggt_feature_dim={cfg.vggt_feature_dim}, "
            f"got obs_shape={cfg.obs_shape}, "
            f"vggt_feature_dim={cfg.vggt_feature_dim}"
        )
    return WMHybridEncoder(
        cnn_depth=cfg.encoder_depth,
        cnn_kernel=cfg.encoder_kernel,
        cnn_mults=cfg.encoder_mults,
        vggt_embed_dim=cfg.vggt_embed_dim,
        mlp_hidden=cfg.mlp_vggt_hidden,
        mlp_layers=cfg.mlp_vggt_layers,
        vggt_dim=cfg.vggt_feature_dim,
    )


def _make_mlp_encoder(cfg: R2DreamerConfig, cls):
    # wp_cp + aggregator MLP encoders: depth from cfg.vggt_mlp_layers (3D-52).
    return cls(
        embed_dim=cfg.vggt_embed_dim,
        hidden=cfg.vggt_embed_dim,
        num_layers=cfg.vggt_mlp_layers,
    )


def _make_token_transformer_encoder(cfg: R2DreamerConfig):
    return WMVGGTAggTokenTransformerEncoder(
        embed_dim=cfg.vggt_embed_dim,
        token_dim=cfg.vggt_token_dim,
        num_tokens=cfg.vggt_token_count,
        projection_dim=cfg.vggt_token_projection_dim,
        layers=cfg.vggt_token_transformer_layers,
        heads=cfg.vggt_token_transformer_heads,
        mlp_ratio=cfg.vggt_token_transformer_mlp_ratio,
        keep_register_tokens=cfg.vggt_keep_register_tokens,
    )


def _make_encoder(cfg: R2DreamerConfig):
    cls = _resolve_encoder_cls(cfg)
    _validate_encoder_config(cfg, cls)
    if cls is ConvEncoder:
        return _make_conv_encoder(cfg)
    if cls is WPConvEncoder:
        return _make_wp_conv_encoder(cfg)
    if cls is WMHybridEncoder:
        return _make_hybrid_encoder(cfg)
    if cls is WMVGGTAggTokenTransformerEncoder:
        return _make_token_transformer_encoder(cfg)
    return _make_mlp_encoder(cfg, cls)


def _weighted_total_loss(cfg: R2DreamerConfig, losses: dict[str, Any]):
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
    # Co-trained decoder term (only present when cfg.decoder; 3D-51).
    if cfg.decoder:
        total_loss = total_loss + cfg.scale_decoder * losses["decoder"]
    return total_loss


def _add_loss_metrics(metrics: dict[str, Any], losses: dict[str, Any]) -> None:
    for k, v in losses.items():
        metrics[f"loss/{k}"] = v


def _add_encoder_l2_metric(metrics: dict[str, Any], params: dict[str, Any]) -> None:
    # Encoder L2 — Protocol D diagnostic for whether Barlow's gradient
    # toggle is actually moving the encoder weights.
    enc_sq = jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(jnp.square(x)),
        params["encoder"], 0.0,
    )
    metrics["params/encoder_l2"] = jnp.sqrt(enc_sq)


def _add_hybrid_contribution_metrics(
    metrics: dict[str, Any],
    *,
    cfg: R2DreamerConfig,
    params: dict[str, Any],
    forward: dict[str, Any],
    B: int,
    T: int,
) -> None:
    # Reuse the already-computed fused embed instead of a second encoder
    # forward: embed == concat([cnn_e, gate * vggt_mlp(...)]), so the
    # leading cnn_dim columns are the CNN branch and the rest are the
    # gated VGGT branch. The raw gate scalar is read straight from params.
    embed_flat = forward["embed"].reshape(B * T, -1)
    cnn_dim = embed_flat.shape[-1] - cfg.vggt_embed_dim
    cnn_e = embed_flat[:, :cnn_dim]
    vggt_e = embed_flat[:, cnn_dim:]
    gate = params["encoder"]["params"]["gate"]
    cnn_l2 = jnp.sqrt(jnp.mean(jnp.sum(cnn_e ** 2, axis=-1)))
    vggt_l2 = jnp.sqrt(jnp.mean(jnp.sum(vggt_e ** 2, axis=-1)))
    denom = cnn_l2 + vggt_l2 + 1e-8
    metrics["hybrid/gate"] = gate
    metrics["hybrid/cnn_l2"] = cnn_l2
    metrics["hybrid/vggt_l2"] = vggt_l2
    metrics["hybrid/cnn_std"] = jnp.std(cnn_e)
    metrics["hybrid/vggt_std"] = jnp.std(vggt_e)
    metrics["hybrid/cnn_frac"] = cnn_l2 / denom
    metrics["hybrid/vggt_frac"] = vggt_l2 / denom


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

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        obs_shape: tuple[int, ...],
        num_actions: int,
        seed: int,
        **config_kwargs: Any,
    ) -> "R2DreamerAgent":
        """Build an agent and load ``params`` + ``slow_critic_params`` from disk.

        Extra ``config_kwargs`` flow into :class:`R2DreamerConfig` so callers
        that need ``encoder_type`` / ``encoder_module_cls`` (e.g. evaluate)
        can pass them through. The loaded checkpoint's ``step`` is stashed on
        the returned agent as ``checkpoint_step`` (``-1`` if absent).
        """
        config = R2DreamerConfig(
            obs_shape=obs_shape, num_actions=num_actions, **config_kwargs,
        )
        rng_key = jax.random.PRNGKey(seed)
        rng_key, init_key = jax.random.split(rng_key)
        agent = cls(config, init_key)
        ckpt = load_policy_checkpoint(path)
        agent.params = jax.tree.map(jnp.asarray, ckpt["params"])
        agent.slow_critic_params = jax.tree.map(jnp.asarray, ckpt["slow_critic_params"])
        agent.checkpoint_step = int(ckpt.get("step", -1))
        return agent

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

        # ---- Co-trained decoder (3D-51): built ONLY when cfg.decoder ----
        # Reconstructs the RGB image from `feat` for visual verification. Left
        # unbuilt by default so the params pytree (and thus checkpoints) of
        # CNN/VGGT runs is unchanged.
        self.decoder_mod = None
        dec_params = None
        if config.decoder:
            if config.encoder_type not in ("cnn", "hybrid", "vggt_house_context"):
                raise ValueError(
                    "decoder=True requires encoder_type in {'cnn', 'hybrid', "
                    "'vggt_house_context'} — the "
                    "ConvDecoder reconstructs an RGB image, but "
                    f"{config.encoder_type!r} carries no RGB modality to reconstruct."
                )
            rng_key, k_dec = jax.random.split(rng_key)
            self.decoder_mod = ConvDecoder(
                depth=config.encoder_depth,
                kernel_size=config.encoder_kernel,
                mults=config.encoder_mults,
            )
            dec_params = self.decoder_mod.init(k_dec, feat0)

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

        if config.decoder:
            self.params["decoder"] = dec_params
            self._modules["decoder"] = self.decoder_mod

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
        self._jit_act = jax.jit(self._act_jit)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, obs_dict: Dict[str, Any], rng_key: jnp.ndarray, training: bool = True) -> int:
        """Select an action for a single environment step.

        Args:
            obs_dict: {"image": uint8 (C,H,W), "is_first": bool} for CNN,
                      {"features": float32 (D,), "is_first": bool} for VGGT,
                      or {"image": uint8 (3,64,64), "wp_cp": float32 (4116,)}
                      for hybrid.
            rng_key: PRNG key.
            training: if False, use argmax (greedy).

        Returns:
            Integer action in [0, num_actions).
        """
        obs = encoder_obs_from_agent_obs(obs_dict, self.cfg)

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
    # Decoder reconstruction (visual verification; only when cfg.decoder)
    # ------------------------------------------------------------------

    def reconstruct(self, batch: Dict[str, jnp.ndarray]):
        """Decode RGB reconstructions for a batch (encoder -> RSSM -> decoder).

        Returns ``(target, recon)`` as numpy arrays ``(B*T, 3, 64, 64)`` in
        [0, 1], or ``None`` when no decoder is configured. Non-JIT, deterministic
        (fixed sample key) — called by the trainer at log cadence for W&B image
        logging, so it is intentionally cheap-and-occasional rather than fast.
        """
        if not self.cfg.decoder or self.decoder_mod is None:
            return None
        params = self.params
        B, T = obs_leading_shape(batch["obs"])
        obs_flat = encoder_obs_from_batch(batch, self.cfg)
        embed = self.encoder_mod.apply(params["encoder"], obs_flat).reshape(B, T, -1)
        stoch0, deter0 = self.rssm_mod.apply(
            params["rssm"], B, method=self.rssm_mod.initial_state)
        post_stochs, post_deters, _ = self.rssm_mod.apply(
            params["rssm"], embed, batch["actions"], (stoch0, deter0),
            batch["is_first"], method=self.rssm_mod.observe,
            rngs={"sample": jax.random.PRNGKey(0)},
        )
        feat = self.rssm_mod.apply(
            params["rssm"], post_stochs, post_deters, method=self.rssm_mod.get_feat)
        recon = self.decoder_mod.apply(params["decoder"], feat.reshape(B * T, -1))
        target = decoder_rgb_target(batch, self.cfg)
        return np.asarray(target), np.asarray(recon)

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
        B, T = obs_leading_shape(batch["obs"])

        obs_flat = encoder_obs_from_batch(batch, cfg)
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
        B, T = obs_leading_shape(batch["obs"])

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
        total_loss = _weighted_total_loss(cfg, losses)

        # ---- Metrics ----
        metrics = {**wm_metrics, **bh_metrics, **rep_metrics}
        _add_loss_metrics(metrics, losses)
        _add_encoder_l2_metric(metrics, params)

        # ---- Hybrid contribution diagnostics (3D-50) ----
        # Re-split the fused embed into its CNN and gated-VGGT branches via the
        # encoder's `branches` method (shares params with the forward pass) and
        # log how much each modality drives the latent. `gate` starts at 0 and
        # opens over training; `*_frac` is each branch's share of the embed norm.
        if cfg.encoder_type in ("hybrid", "vggt_house_context"):
            _add_hybrid_contribution_metrics(
                metrics, cfg=cfg, params=params, forward=forward, B=B, T=T,
            )

        aux = {"metrics": metrics, "imag_returns": imag_ret.reshape(-1)}
        return total_loss, aux
