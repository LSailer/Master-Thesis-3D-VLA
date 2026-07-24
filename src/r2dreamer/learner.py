"""R2DLearner — the Dreamer learner with the encoder injected (design card 3a).

The learner owns parameters, the three LaProp optimizers (world-model incl.
encoder / actor / critic), the slow-target EMA, the acting state, and the
JIT'd train/eval entry points. It does NOT construct the encoder: the
composition root (``launch/train.py``, or the temporary ``R2DreamerAgent``
shim) builds the encoder module from the ``EncoderRecipe`` registry and
injects it, together with the batch-1 observation used for ``enc.init`` —
per IDEA.md decision 1, the launcher owns encoder construction and wiring.

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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax

from src.buffer import ReplayBatch
from src.configs.config import R2DreamerConfig
from src.r2dreamer.decoder_targets import decoder_rgb_target, replay_batch_shape
from src.r2dreamer.encoder_types import RGB_BEARING_ENCODER_TYPES
from src.r2dreamer.encoders.shape_utils import batch_live_observation
from src.shared.optim import LaPropState, laprop, agc

from .behavior.loss import behavior_loss
from .behavior.return_ema import ReturnEMA
from .checkpointing import load_checkpoint
from .encoders.decoder import ConvDecoder
from .learning_types import AgentLossAux, WorldModelForward
from .representation.barlow import Projector
from .representation.loss import representation_loss
from .world_model.factory import compute_dtype_kwargs, make_rssm
from .world_model.heads import R2MLP, R2TwoHotDist
from .world_model.loss import world_model_loss


class ActState(NamedTuple):
    """Functional single-env acting state."""

    stoch: jax.Array
    deter: jax.Array
    prev_action: jax.Array


# ---------------------------------------------------------------------------
# Three-optimizer parameter partition (world-model incl. encoder / actor / critic)
# ---------------------------------------------------------------------------
#
# The WM optimizer owns every param subtree except the actor and the critic.
# The split is bit-identical to the historical single optimizer because LaProp
# and AGC are both per-leaf (jax.tree.map, no cross-leaf coupling): three
# LaProp instances with identical hyperparameters, each stepped once per
# train_step, evolve the shared scalar state (step / lr bias-correction terms)
# in lockstep, so every leaf's update matches the single-optimizer update
# exactly. See IDEA.md decision 2 and design card 2e (1b-tri).

_ACTOR_SUBTREE = "actor"
_CRITIC_SUBTREE = "critic"


def _partition_by_group(tree: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    """Split a params-shaped pytree into (world-model, actor, critic) subtrees."""
    actor = {_ACTOR_SUBTREE: tree[_ACTOR_SUBTREE]}
    critic = {_CRITIC_SUBTREE: tree[_CRITIC_SUBTREE]}
    wm = {k: v for k, v in tree.items() if k not in (_ACTOR_SUBTREE, _CRITIC_SUBTREE)}
    return wm, actor, critic


def _merge_groups(wm: Mapping, actor: Mapping, critic: Mapping) -> dict:
    """Recombine (world-model, actor, critic) subtrees into a params pytree."""
    return {**wm, **actor, **critic}


_ENCODER_SUBTREE = "encoder"


def _split_structural(tree: Mapping[str, Any]) -> tuple[Any, dict, dict, dict]:
    """Split a params-shaped pytree into the loss-fn argument groups.

    Decision 3 (structural): ``enc_params`` is an explicit argument of the
    loss, jointly differentiated with the world-model params. The groups are
    ``(encoder, wm-without-encoder, actor, critic)``; note the OPTIMIZER
    partition (:func:`_partition_by_group`) keeps the encoder inside the WM
    group — this split only shapes the loss-fn signature.
    """
    wm = {
        k: v
        for k, v in tree.items()
        if k not in (_ENCODER_SUBTREE, _ACTOR_SUBTREE, _CRITIC_SUBTREE)
    }
    return (
        tree[_ENCODER_SUBTREE],
        wm,
        {_ACTOR_SUBTREE: tree[_ACTOR_SUBTREE]},
        {_CRITIC_SUBTREE: tree[_CRITIC_SUBTREE]},
    )


def _split_single_opt_state(
    single: LaPropState, params: Mapping[str, Any]
) -> tuple[LaPropState, LaPropState, LaPropState]:
    """Migrate a legacy single LaProp state into the three-optimizer layout.

    Old checkpoints stored one LaProp state over the whole params pytree. The
    per-leaf moments partition cleanly by group, and the scalar step / lr
    bias-correction fields were shared across all leaves, so copying them into
    each of the three states is exact — resume stays bit-identical.

    Args:
      single: The legacy ``LaPropState`` over the full params pytree.
      params: Current params pytree (only its key structure is used).

    Returns:
      ``(wm_state, actor_state, critic_state)`` LaProp states.
    """
    del params  # partition is driven by the moment pytrees' own keys
    ea_wm, ea_a, ea_c = _partition_by_group(single.exp_avg)
    eas_wm, eas_a, eas_c = _partition_by_group(single.exp_avg_sq)

    def build(exp_avg, exp_avg_sq) -> LaPropState:
        return LaPropState(
            step=single.step,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            exp_avg_lr1=single.exp_avg_lr1,
            exp_avg_lr2=single.exp_avg_lr2,
        )

    return build(ea_wm, eas_wm), build(ea_a, eas_a), build(ea_c, eas_c)


class R2DTrainState(NamedTuple):
    """Mutable-on-Python-side training state bundled for the JIT step.

    The public learner keeps exposing ``params``/``opt_state``/
    ``slow_critic_params``/``ema_state`` properties for checkpoint and test
    compatibility, but the compiled training kernel receives and returns this
    single pytree so its interface stays compact.

    The optimizer state is split three ways — world-model (encoder + RSSM +
    heads + projector + optional decoder), actor, and critic — one LaProp state
    each. ``opt_state`` (the public property) presents them as a
    ``{"wm", "actor", "critic"}`` dict for checkpointing.
    """

    params: Dict[str, Any]
    wm_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    slow_critic_params: Dict[str, Any]
    ema_state: Any


def load_policy_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load an R2Dreamer checkpoint, tolerating moved optimizer classes."""
    path = Path(path)
    ckpt = load_checkpoint(str(path))
    missing = {"params", "slow_critic_params"} - set(ckpt)
    if missing:
        raise KeyError(f"checkpoint {path} is missing required keys: {sorted(missing)}")
    return ckpt


def _weighted_total_loss(cfg: R2DreamerConfig, losses: dict[str, Any]):
    """Agent objective, excluding the optional debug decoder probe."""
    return (
        cfg.scale_dyn * losses["dyn"]
        + cfg.scale_rep * losses["rep"]
        + cfg.scale_barlow * losses["barlow"]
        + cfg.scale_rew * losses["rew"]
        + cfg.scale_con * losses["con"]
        + cfg.scale_policy * losses["policy"]
        + cfg.scale_value * losses["value"]
        + cfg.scale_repval * losses["repval"]
    )


def _add_loss_metrics(metrics: dict[str, Any], losses: dict[str, Any]) -> None:
    for k, v in losses.items():
        metrics[f"loss/{k}"] = v


def _add_encoder_l2_metric(metrics: dict[str, Any], params: dict[str, Any]) -> None:
    # Encoder L2 — Protocol D diagnostic for whether Barlow's gradient
    # toggle is actually moving the encoder weights.
    enc_sq = jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(jnp.square(x)),
        params["encoder"],
        0.0,
    )
    metrics["params/encoder_l2"] = jnp.sqrt(enc_sq)


def _add_hybrid_contribution_metrics(
    metrics: dict[str, Any],
    *,
    cfg: R2DreamerConfig,
    params: dict[str, Any],
    forward: WorldModelForward,
    B: int,
    T: int,
) -> None:
    # Reuse the already-computed fused embed instead of a second encoder
    # forward: embed == concat([cnn_e, gate * vggt_mlp(...)]), so the
    # leading cnn_dim columns are the CNN branch and the rest are the
    # gated VGGT branch. The raw gate scalar is read straight from params.
    embed_flat = forward.embed.reshape(B * T, -1)
    cnn_dim = embed_flat.shape[-1] - cfg.vggt_embed_dim
    cnn_e = embed_flat[:, :cnn_dim]
    vggt_e = embed_flat[:, cnn_dim:]
    gate = params["encoder"]["params"]["gate"]
    cnn_l2 = jnp.sqrt(jnp.mean(jnp.sum(cnn_e**2, axis=-1)))
    vggt_l2 = jnp.sqrt(jnp.mean(jnp.sum(vggt_e**2, axis=-1)))
    denom = cnn_l2 + vggt_l2 + 1e-8
    metrics["hybrid/gate"] = gate
    metrics["hybrid/cnn_l2"] = cnn_l2
    metrics["hybrid/vggt_l2"] = vggt_l2
    metrics["hybrid/cnn_std"] = jnp.std(cnn_e)
    metrics["hybrid/vggt_std"] = jnp.std(vggt_e)
    metrics["hybrid/cnn_frac"] = cnn_l2 / denom
    metrics["hybrid/vggt_frac"] = vggt_l2 / denom


# ---------------------------------------------------------------------------
# R2DLearner
# ---------------------------------------------------------------------------


class R2DLearner:
    """Dreamer learner over an injected encoder module.

    All Flax modules are *stateless* — parameters live in a flat pytree dict
    exposed as ``self.params``.  Training state is bundled in
    ``self.train_state`` and threaded through one JIT-compiled pure step.

    The RNG key discipline in ``__init__`` (split order, one key per module
    init) is frozen: it must match the historical ``R2DreamerAgent``
    constructor exactly so that a given seed keeps producing bit-identical
    parameters across the encoder-split migration (golden-run gate).
    """

    @property
    def train_state(self) -> R2DTrainState:
        """The bundled training state threaded through the JIT step."""
        return self._train_state

    @train_state.setter
    def train_state(self, state: R2DTrainState) -> None:
        self._train_state = state

    @property
    def params(self):
        """The full parameter pytree (encoder + RSSM + heads + actor/critic)."""
        return self._train_state.params

    @params.setter
    def params(self, params):
        self._train_state = self._train_state._replace(params=params)

    @property
    def opt_state(self):
        """The three LaProp states as a stable ``{wm, actor, critic}`` dict."""
        # Present the three LaProp states as a stable dict for checkpointing.
        return {
            "wm": self._train_state.wm_opt_state,
            "actor": self._train_state.actor_opt_state,
            "critic": self._train_state.critic_opt_state,
        }

    @opt_state.setter
    def opt_state(self, opt_state):
        # Accept the three-optimizer dict (current format) or migrate a legacy
        # single LaProp state loaded from an old checkpoint.
        if isinstance(opt_state, Mapping) and {"wm", "actor", "critic"} <= set(
            opt_state
        ):
            wm, actor, critic = (
                opt_state["wm"],
                opt_state["actor"],
                opt_state["critic"],
            )
        else:
            wm, actor, critic = _split_single_opt_state(opt_state, self.params)
        self._train_state = self._train_state._replace(
            wm_opt_state=wm,
            actor_opt_state=actor,
            critic_opt_state=critic,
        )

    @property
    def slow_critic_params(self):
        """EMA slow-target critic parameters."""
        return self._train_state.slow_critic_params

    @slow_critic_params.setter
    def slow_critic_params(self, slow_critic_params):
        self._train_state = self._train_state._replace(
            slow_critic_params=slow_critic_params
        )

    @property
    def ema_state(self):
        """Return-normalization EMA state."""
        return self._train_state.ema_state

    @ema_state.setter
    def ema_state(self, ema_state):
        self._train_state = self._train_state._replace(ema_state=ema_state)

    def __init__(
        self,
        config: R2DreamerConfig,
        rng_key: jnp.ndarray,
        *,
        encoder: Any,
        encoder_init_obs: Any,
    ):
        """Build the learner around an externally-constructed encoder.

        Args:
            config: Effective agent config (RSSM/head/optimizer knobs).
            rng_key: Master init PRNG key; split internally per module.
            encoder: The trainable Flax encoder module (built by the
                composition root from the ``EncoderRecipe`` registry).
            encoder_init_obs: Batch-1 observation used for ``encoder.init`` —
                the first prepared frame at the launcher, or the recipe's
                init dummy for direct construction (tests, checkpoints).
        """
        self.cfg = config
        self.checkpoint_step = -1
        self.twohot = R2TwoHotDist(num_bins=config.twohot_bins)

        # ---- Injected encoder + RSSM module (for .apply) ----
        self.encoder_mod = encoder
        self.rssm_mod = make_rssm(config)

        # Dummy forward to discover embed_size
        rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
        dummy_obs = encoder_init_obs
        enc_params = self.encoder_mod.init(k1, dummy_obs)
        embed = cast(jnp.ndarray, self.encoder_mod.apply(enc_params, dummy_obs))
        self.embed_size = embed.shape[-1]

        # RSSM
        stoch0 = jnp.zeros((1, config.stoch_classes, config.stoch_discrete))
        deter0 = jnp.zeros((1, config.deter_size))
        action0 = jnp.zeros((1, config.num_actions))
        embed0 = jnp.zeros((1, self.embed_size))
        rng_key, k_sample = jax.random.split(rng_key)
        rssm_params = self.rssm_mod.init(
            {"params": k2, "sample": k_sample}, stoch0, deter0, action0, embed0
        )

        # Projector: feat_size -> embed_size
        self.proj_mod = Projector(out_dim=self.embed_size)
        feat0 = jnp.zeros((1, config.feat_size))
        proj_params = self.proj_mod.init(k3, feat0)

        # MLP heads (outscale matches PyTorch: 0.0 for reward/critic, 0.01 for actor)
        rng_key, k_rew, k_con, k_act, k_cri = jax.random.split(rng_key, 5)
        head_dtype_kwargs = compute_dtype_kwargs(config)
        self.reward_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_reward,
            out_dim=config.twohot_bins,
            outscale=0.0,
            **head_dtype_kwargs,
        )
        rew_params = self.reward_mod.init(k_rew, feat0)

        self.cont_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_cont,
            out_dim=1,
            **head_dtype_kwargs,
        )
        con_params = self.cont_mod.init(k_con, feat0)

        self.actor_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_actor,
            out_dim=config.num_actions,
            outscale=0.01,
            **head_dtype_kwargs,
        )
        act_params = self.actor_mod.init(k_act, feat0)

        self.critic_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_critic,
            out_dim=config.twohot_bins,
            outscale=0.0,
            **head_dtype_kwargs,
        )
        cri_params = self.critic_mod.init(k_cri, feat0)

        # ---- Debug decoder probe (3D-51): built ONLY when cfg.decoder ----
        # Reconstructs RGB from stop-gradient `feat` for visual verification.
        # Left unbuilt by default so the params pytree (and thus checkpoints) of
        # CNN/VGGT runs is unchanged.
        self.decoder_mod = None
        dec_params = None
        if config.decoder:
            if config.encoder_type not in RGB_BEARING_ENCODER_TYPES:
                raise ValueError(
                    "decoder=True requires an RGB-bearing encoder_type — the "
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
        params = {
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
            params["decoder"] = dec_params
            self._modules["decoder"] = self.decoder_mod

        # ---- Optimizers: three LaProp instances (WM incl. encoder / actor /
        # critic), each with identical hyperparameters. Because LaProp + AGC are
        # per-leaf, this is update-identical to the historical single optimizer
        # (IDEA.md decision 2). Per-module learning rates are a deliberate later
        # experiment, not this refactor.
        def _make_tx() -> optax.GradientTransformation:
            return laprop(
                lr=config.lr,
                b1=config.beta1,
                b2=config.beta2,
                eps=config.eps,
                warmup=config.warmup_steps,
            )

        self.wm_tx = _make_tx()
        self.actor_tx = _make_tx()
        self.critic_tx = _make_tx()
        wm_params, actor_params, critic_params = _partition_by_group(params)
        wm_opt_state = self.wm_tx.init(wm_params)
        actor_opt_state = self.actor_tx.init(actor_params)
        critic_opt_state = self.critic_tx.init(critic_params)

        # ---- Slow target critic (EMA) ----
        slow_critic_params = jax.tree.map(jnp.copy, params["critic"])

        # ---- Return EMA ----
        self.return_ema = ReturnEMA()
        ema_state = self.return_ema.init_state()

        self.train_state = R2DTrainState(
            params=params,
            wm_opt_state=wm_opt_state,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            slow_critic_params=slow_critic_params,
            ema_state=ema_state,
        )

        # ---- Acting state (for legacy single-env stepping wrapper) ----
        self._act_state = self.initial_act_state()

        # ---- JIT-compiled functions ----
        self._jitted_train_step = cast(Any, jax.jit(self._train_step_pure))
        self._jit_act_with_state = cast(Any, jax.jit(self.act_with_state_pure))

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def initial_act_state(self) -> ActState:
        """Return a zeroed functional single-env acting state."""
        return ActState(
            stoch=jnp.zeros(
                (1, self.cfg.stoch_classes, self.cfg.stoch_discrete), dtype=jnp.float32
            ),
            deter=jnp.zeros((1, self.cfg.deter_size), dtype=jnp.float32),
            prev_action=jnp.zeros((1, self.cfg.num_actions), dtype=jnp.float32),
        )

    def snapshot_act_state(self) -> ActState:
        """Copy the legacy mutable wrapper's acting state."""
        return jax.tree.map(jnp.copy, self._act_state)

    def restore_act_state(self, state: ActState) -> None:
        """Restore the legacy mutable wrapper's acting state."""
        self._act_state = state

    def act(
        self,
        encoder_obs: Any,
        is_first: bool,
        rng_key: jnp.ndarray,
        training: bool = True,
    ) -> int:
        """Select an action for a single prepared environment step.

        Args:
            encoder_obs: one live observation in the layout consumed by the encoder.
                The agent adds the single-env batch dimension internally.
            is_first: whether the step starts an episode and should reset RSSM state.
            rng_key: PRNG key.
            training: if False, use argmax (greedy).

        Returns:
            Integer action in [0, num_actions).
        """
        reset = jnp.asarray(is_first, dtype=jnp.bool_)
        batched_obs = batch_live_observation(encoder_obs)
        action_int, self._act_state = self._jit_act_with_state(
            self.params, batched_obs, self._act_state, reset, rng_key, training
        )

        # Honor the ``-> int`` contract: the jitted core returns a 0-d JAX array,
        # but callers (env.step, action_counts indexing) need a host Python int.
        # habitat's env.step only wraps int/np.integer into {"action": ...}; a
        # raw JAX array slips through to string indexing and raises.
        return int(action_int)

    def act_with_state(
        self,
        encoder_obs: Any,
        is_first: bool,
        state: ActState,
        rng_key: jnp.ndarray,
        training: bool = True,
    ) -> tuple[int, ActState]:
        """Functional acting wrapper for one raw live encoder observation."""
        reset = jnp.asarray(is_first, dtype=jnp.bool_)
        batched_obs = batch_live_observation(encoder_obs)
        action_int, new_state = self._jit_act_with_state(
            self.params, batched_obs, state, reset, rng_key, training
        )
        # As in ``act``: return a host int action (state pytree passes through
        # untouched so the next jitted call still sees stable shapes/dtypes).
        return int(action_int), new_state

    def act_with_state_pure(
        self, params, obs, state: ActState, is_first, rng_key, training
    ):
        """JIT-able acting logic. Returns (action_int, next ActState)."""
        state = jax.lax.cond(
            is_first,
            lambda _: self.initial_act_state(),
            lambda current: current,
            state,
        )
        embed = cast(jnp.ndarray, self.encoder_mod.apply(params["encoder"], obs))
        rng_key, k_sample = jax.random.split(rng_key)
        new_stoch, new_deter, _ = cast(
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(
                params["rssm"],
                state.stoch,
                state.deter,
                state.prev_action,
                embed,
                rngs={"sample": k_sample},
            ),
        )
        feat = cast(
            jnp.ndarray,
            self.rssm_mod.apply(
                params["rssm"], new_stoch, new_deter, method=self.rssm_mod.get_feat
            ),
        )
        logits = self.actor_mod.apply(params["actor"], feat)

        def _sample(logits, rng_key):
            return jax.random.categorical(rng_key, logits, axis=-1)[0]

        def _greedy(logits, _rng_key):
            return jnp.argmax(logits, axis=-1)[0]

        action_int = jax.lax.cond(training, _sample, _greedy, logits, rng_key)
        new_state = ActState(
            stoch=new_stoch,
            deter=new_deter,
            prev_action=jax.nn.one_hot(
                action_int, self.cfg.num_actions, dtype=jnp.float32
            )[None],
        )
        return action_int, new_state

    # ------------------------------------------------------------------
    # Decoder reconstruction probe (visual verification; only when cfg.decoder)
    # ------------------------------------------------------------------

    def reconstruct(self, batch: Any):
        """Decode RGB reconstructions for a batch (encoder -> RSSM -> decoder).

        Returns ``(target, recon)`` as JAX arrays ``(B*T, 64, 64, 3)`` in
        [0, 1], or ``None`` when no decoder is configured. Non-JIT, deterministic
        (fixed sample key) — called by the trainer at log cadence for W&B image
        logging, so it is intentionally cheap-and-occasional rather than fast.
        """
        if not self.cfg.decoder or self.decoder_mod is None:
            return None
        params = self.params
        B, T = replay_batch_shape(batch)
        embed = cast(
            jnp.ndarray, self.encoder_mod.apply(params["encoder"], batch.obs)
        )
        if embed.shape[:2] != (B, T):
            raise ValueError(
                f"encoder must preserve replay leading dims {(B, T)}, got {embed.shape}"
            )
        stoch0, deter0 = cast(
            tuple[jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(params["rssm"], B, method=self.rssm_mod.initial_state),
        )
        post_stochs, post_deters, _ = cast(
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(
                params["rssm"],
                embed,
                batch.actions,
                (stoch0, deter0),
                batch.is_first,
                method=self.rssm_mod.observe,
                rngs={"sample": jax.random.PRNGKey(0)},
            ),
        )
        feat = cast(
            jnp.ndarray,
            self.rssm_mod.apply(
                params["rssm"], post_stochs, post_deters, method=self.rssm_mod.get_feat
            ),
        )
        recon = self.decoder_mod.apply(params["decoder"], feat.reshape(B * T, -1))
        target = decoder_rgb_target(batch, self.cfg.encoder_type)
        return target, recon

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: ReplayBatch,
        rng_key: jnp.ndarray,
        *,
        materialize: bool = True,
    ) -> Dict[str, Any]:
        """One LaProp step on `batch`.

        Args:
          batch: The replay batch to train on.
          rng_key: PRNG key for the step.
          materialize: When ``True`` (default), block and return Python-float
            metrics. When ``False``, return the raw device-array metrics
            without forcing a device->host sync. The hot training loop passes
            ``False`` on non-logging steps so JAX async dispatch is not
            serialized ~every step for metrics that would be discarded.

        Returns:
          A dict of metric name to value. Python floats when ``materialize``,
          otherwise device ``jax.Array`` scalars.
        """
        self.train_state, metrics = self._jitted_train_step(
            self.train_state,
            batch,
            rng_key,
        )
        if materialize:
            return {k: float(v) for k, v in metrics.items()}
        return dict(metrics)

    def eval_loss(self, batch: Any, rng_key: jnp.ndarray) -> Dict[str, float]:
        """Evaluate the current objective on a batch without updating state."""
        enc_p, wm_p, actor_p, critic_p = _split_structural(self.params)
        _total_loss, aux = self._loss_fn(
            enc_p,
            wm_p,
            actor_p,
            critic_p,
            slow_critic_params=self.slow_critic_params,
            ema_state=self.ema_state,
            batch=batch,
            rng_key=rng_key,
        )
        metrics = dict(aux.metrics)
        metrics["total_loss"] = aux.agent_loss
        return {k: float(v) for k, v in metrics.items()}

    def _train_step_pure(self, state: R2DTrainState, batch, rng_key):
        """Pure-functional training step (JIT-able).

        Decision 3 (structural): the loss takes ``(enc_params, wm_params,
        actor_params, critic_params)`` as explicit arguments and ONE
        ``jax.value_and_grad(..., argnums=(0, 1, 2, 3))`` differentiates all
        four jointly in a single compiled graph — every param still receives
        ``d(total_loss)/d(param)``, so the gradient coupling between the
        representation losses and the encoder/RSSM is preserved, and the step
        stays bit-identical to the historical full-dict ``value_and_grad``.
        (Three SEPARATE grad calls are NOT bit-identical: XLA rounds the
        separate actor/critic backward graphs differently — measured 5.2e-12
        on actor, 6.7e-08 on critic — which would break the golden gate.)
        The optimizer application is split three ways (WM incl. encoder /
        actor / critic), update-identical because LaProp + AGC are per-leaf.
        """
        params = state.params
        slow_critic_params = state.slow_critic_params
        ema_state = state.ema_state

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

        enc_p, wm_only_p, actor_arg_p, critic_arg_p = _split_structural(params)
        (total_loss, aux), (enc_g, wm_g, actor_g, critic_g) = jax.value_and_grad(
            loss_fn, argnums=(0, 1, 2, 3), has_aux=True
        )(enc_p, wm_only_p, actor_arg_p, critic_arg_p)
        grads = {_ENCODER_SUBTREE: enc_g, **wm_g, **actor_g, **critic_g}

        # NaN guard: skip update if loss is non-finite (mirrors PyTorch GradScaler)
        is_finite = jnp.isfinite(total_loss)

        grads = agc(grads, params, clip=self.cfg.agc_clip, pmin=self.cfg.agc_pmin)

        # Split params + grads by group and step each optimizer once. The three
        # LaProp states advance their shared scalar (step / lr) fields in
        # lockstep, so per-leaf updates equal the single-optimizer updates.
        wm_p, actor_p, critic_p = _partition_by_group(params)
        wm_g, actor_g, critic_g = _partition_by_group(grads)
        wm_upd, new_wm_opt = self.wm_tx.update(wm_g, state.wm_opt_state, wm_p)
        actor_upd, new_actor_opt = self.actor_tx.update(
            actor_g, state.actor_opt_state, actor_p
        )
        critic_upd, new_critic_opt = self.critic_tx.update(
            critic_g, state.critic_opt_state, critic_p
        )
        new_params = _merge_groups(
            optax.apply_updates(wm_p, wm_upd),
            optax.apply_updates(actor_p, actor_upd),
            optax.apply_updates(critic_p, critic_upd),
        )

        new_ema_state = self.return_ema.update(ema_state, aux.imag_returns)

        # Roll back to pre-update state on NaN/inf
        def _rollback(new, old):
            return jax.tree.map(lambda n, o: jnp.where(is_finite, n, o), new, old)

        new_params = _rollback(new_params, params)
        new_wm_opt = _rollback(new_wm_opt, state.wm_opt_state)
        new_actor_opt = _rollback(new_actor_opt, state.actor_opt_state)
        new_critic_opt = _rollback(new_critic_opt, state.critic_opt_state)
        new_slow = _rollback(updated_slow, slow_critic_params)
        new_ema_state = _rollback(new_ema_state, ema_state)

        metrics = aux.metrics
        metrics["opt_loss"] = total_loss
        metrics["total_loss"] = aux.agent_loss
        metrics["nan_skipped"] = 1.0 - is_finite.astype(jnp.float32)
        new_state = R2DTrainState(
            params=new_params,
            wm_opt_state=new_wm_opt,
            actor_opt_state=new_actor_opt,
            critic_opt_state=new_critic_opt,
            slow_critic_params=new_slow,
            ema_state=new_ema_state,
        )
        return new_state, metrics

    # ------------------------------------------------------------------
    # Loss composition: shared forward + 3 sub-losses
    # ------------------------------------------------------------------

    def _world_model_forward(self, params, batch, rng_key) -> WorldModelForward:
        """Encoder + posterior rollout + prior + features. Shared across sub-losses.

        Computing this once is essential: if each sub-loss recomputed `embed`,
        the encoder would receive doubled gradient signal and the
        `barlow_stop_grad` toggle would no longer mean what it claims.
        """
        cfg = self.cfg
        B, T = replay_batch_shape(batch)

        embed = cast(
            jnp.ndarray, self.encoder_mod.apply(params["encoder"], batch.obs)
        )
        if embed.shape[:2] != (B, T):
            raise ValueError(
                f"encoder must preserve replay leading dims {(B, T)}, got {embed.shape}"
            )

        stoch0, deter0 = cast(
            tuple[jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(params["rssm"], B, method=self.rssm_mod.initial_state),
        )

        rng_key, k_obs = jax.random.split(rng_key)
        post_stochs, post_deters, post_logits = cast(
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(
                params["rssm"],
                embed,
                batch.actions,
                (stoch0, deter0),
                batch.is_first,
                method=self.rssm_mod.observe,
                rngs={"sample": k_obs},
            ),
        )

        rng_key, k_prior = jax.random.split(rng_key)
        _, prior_logits_flat = cast(
            tuple[jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(
                params["rssm"],
                post_deters.reshape(B * T, -1),
                method=self.rssm_mod.prior,
                rngs={"sample": k_prior},
            ),
        )
        prior_logits = prior_logits_flat.reshape(
            B, T, cfg.stoch_classes, cfg.stoch_discrete
        )

        feat = cast(
            jnp.ndarray,
            self.rssm_mod.apply(
                params["rssm"],
                post_stochs,
                post_deters,
                method=self.rssm_mod.get_feat,
            ),
        )

        return WorldModelForward(
            embed=embed,
            post_stochs=post_stochs,
            post_deters=post_deters,
            post_logits=post_logits,
            prior_logits=prior_logits,
            feat=feat,
        )

    def _loss_fn(
        self,
        enc_params,
        wm_params,
        actor_params,
        critic_params,
        *,
        slow_critic_params,
        ema_state,
        batch,
        rng_key,
    ):
        """Compose the world-model, behavior, and representation losses.

        Decision 3: ``enc_params`` is an explicit argument, jointly
        differentiated with ``wm_params`` (RSSM + heads + projector +
        optional decoder). The behavior losses run on the stop-gradiented
        imagination rollout (``behavior/loss.py`` cuts `feat`), so actor and
        critic gradients never reach the encoder; the deliberate exception is
        the replay-value representation loss, which couples the critic to the
        unfrozen replay features by design.

        Returns:
            (total_loss, aux) — `aux` carries metrics and the imagination
            returns used for the post-step `ReturnEMA` update.
        """
        params = {
            _ENCODER_SUBTREE: enc_params,
            **wm_params,
            **actor_params,
            **critic_params,
        }
        cfg = self.cfg
        B, T = replay_batch_shape(batch)

        rng_key, k_fwd = jax.random.split(rng_key)
        forward = self._world_model_forward(params, batch, k_fwd)

        wm_result = world_model_loss(
            forward=forward,
            params=params,
            batch=batch,
            modules=self._modules,
            cfg=cfg,
            twohot=self.twohot,
        )

        rng_key, k_behavior = jax.random.split(rng_key)
        behavior_result = behavior_loss(
            forward=forward,
            params=params,
            modules=self._modules,
            cfg=cfg,
            twohot=self.twohot,
            slow_critic_params=slow_critic_params,
            ema_state=ema_state,
            return_ema=self.return_ema,
            rng_key=k_behavior,
            B=B,
            T=T,
        )

        rep_result = representation_loss(
            forward=forward,
            batch=batch,
            params=params,
            modules=self._modules,
            cfg=cfg,
            twohot=self.twohot,
            slow_critic_params=slow_critic_params,
            imag_ret=behavior_result.imag_returns,
            B=B,
            T=T,
        )

        losses = {
            **wm_result.losses,
            **behavior_result.losses,
            **rep_result.losses,
        }
        agent_loss = _weighted_total_loss(cfg, losses)
        # The decoder is a stop-gradient visualisation probe. Add its detached
        # reconstruction loss only to the optimiser objective so the decoder
        # learns to read the current latent, while the agent/RSSM/encoder see
        # exactly the same objective as decoder-free runs.
        total_loss = agent_loss
        if cfg.decoder:
            total_loss = total_loss + cfg.scale_decoder * losses["decoder"]

        # ---- Metrics ----
        metrics = {
            **wm_result.metrics,
            **behavior_result.metrics,
            **rep_result.metrics,
        }
        _add_loss_metrics(metrics, losses)
        _add_encoder_l2_metric(metrics, params)

        # ---- Hybrid contribution diagnostics (3D-50) ----
        # Re-split the fused embed into its CNN and gated-VGGT branches via the
        # encoder's `branches` method (shares params with the forward pass) and
        # log how much each modality drives the latent. `gate` starts at 0 and
        # opens over training; `*_frac` is each branch's share of the embed norm.
        if cfg.encoder_type in ("hybrid", "vggt_house_context"):
            _add_hybrid_contribution_metrics(
                metrics,
                cfg=cfg,
                params=params,
                forward=forward,
                B=B,
                T=T,
            )

        aux = AgentLossAux(
            metrics=metrics,
            imag_returns=behavior_result.imag_returns.reshape(-1),
            agent_loss=agent_loss,
        )
        return total_loss, aux
