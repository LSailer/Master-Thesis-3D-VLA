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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax

from src.adapters.contract import (
    AdapterOutput,
    decoder_target_key,
    encoder_obs_from_fields,
)
from src.buffer import ReplayBatch
from src.configs.config import R2DreamerConfig
from src.r2dreamer.decoder_targets import decoder_rgb_target, replay_batch_shape
from src.shared.optim import agc, laprop

from .behavior.loss import behavior_loss
from .behavior.return_ema import ReturnEMA
from .checkpointing import load_checkpoint
from .encoders.decoder import ConvDecoder
from .encoders.routed_composite import routed_encoder_from_fields
from .learning_types import AgentLossAux, WorldModelForward
from .representation.barlow import Projector
from .representation.loss import representation_loss
from .world_model.heads import R2MLP, R2TwoHotDist
from .world_model.loss import world_model_loss
from .world_model.rssm_factory import compute_dtype_kwargs, rssm_from_config


class ActState(NamedTuple):
    """Functional single-env acting state."""

    stoch: jax.Array
    deter: jax.Array
    prev_action: jax.Array


class R2DTrainState(NamedTuple):
    """Mutable-on-Python-side training state bundled for the JIT step.

    The public ``R2DreamerAgent`` keeps exposing ``params``/``opt_state``/
    ``slow_critic_params``/``ema_state`` properties for checkpoint and test
    compatibility, but the compiled training kernel receives and returns this
    single pytree so its interface stays compact.
    """

    params: Dict[str, Any]
    opt_state: optax.OptState
    slow_critic_params: Dict[str, Any]
    ema_state: Any


def load_policy_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load an R2DreamerAgent checkpoint, tolerating moved optimizer classes."""
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


def _assert_params_match(fresh: Any, loaded: Any) -> None:
    """Check a freshly built param tree against one loaded from a checkpoint.

    Routed runs rebuild the encoder from a live adapter call instead of from a
    stored architecture description, so a config or adapter change since the
    checkpoint was written shows up here - as a named path, rather than as a
    shape error deep inside a jitted apply.

    Args:
        fresh: Params of the freshly initialized agent.
        loaded: Params read from the checkpoint.

    Raises:
        ValueError: On the first structural or shape difference.
    """

    def shapes(tree: Any) -> dict[str, tuple[int, ...]]:
        return {
            jax.tree_util.keystr(path): jnp.shape(leaf)
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
        }

    fresh_shapes, loaded_shapes = shapes(fresh), shapes(loaded)
    missing = sorted(set(fresh_shapes) - set(loaded_shapes))
    extra = sorted(set(loaded_shapes) - set(fresh_shapes))
    if missing or extra:
        raise ValueError(
            "checkpoint params do not match the rebuilt architecture: "
            f"missing in checkpoint {missing[:5]}, unknown in checkpoint "
            f"{extra[:5]}"
        )
    for path, shape in fresh_shapes.items():
        if loaded_shapes[path] != shape:
            raise ValueError(
                f"checkpoint param shape mismatch at {path}: rebuilt {shape}, "
                f"checkpoint {loaded_shapes[path]}"
            )


# ---------------------------------------------------------------------------
# R2DreamerAgent
# ---------------------------------------------------------------------------


class R2DreamerAgent:
    """R2-Dreamer agent with a single LaProp optimizer over all parameters.

    All Flax modules are *stateless* — parameters live in a flat pytree dict
    exposed as ``self.params``.  Training state is bundled in
    ``self.train_state`` and threaded through one JIT-compiled pure step.
    """

    @property
    def train_state(self) -> R2DTrainState:
        return self._train_state

    @train_state.setter
    def train_state(self, state: R2DTrainState) -> None:
        self._train_state = state

    @property
    def params(self):
        return self._train_state.params

    @params.setter
    def params(self, params):
        self._train_state = self._train_state._replace(params=params)

    @property
    def opt_state(self):
        return self._train_state.opt_state

    @opt_state.setter
    def opt_state(self, opt_state):
        self._train_state = self._train_state._replace(opt_state=opt_state)

    @property
    def slow_critic_params(self):
        return self._train_state.slow_critic_params

    @slow_critic_params.setter
    def slow_critic_params(self, slow_critic_params):
        self._train_state = self._train_state._replace(
            slow_critic_params=slow_critic_params
        )

    @property
    def ema_state(self):
        return self._train_state.ema_state

    @ema_state.setter
    def ema_state(self, ema_state):
        self._train_state = self._train_state._replace(ema_state=ema_state)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        num_actions: int,
        seed: int,
        fields: AdapterOutput,
        encoder_overrides: Dict[str, Any] | None = None,
        **config_kwargs: Any,
    ) -> "R2DreamerAgent":
        """Build an agent and load ``params`` + ``slow_critic_params`` from disk.

        The encoder is rebuilt from ``fields`` (one live adapter call) and only
        the parameters come from disk - the checkpoint carries no architecture
        description. ``_assert_params_match`` then guards against an
        architecture that drifted since the checkpoint was written.

        Args:
            path: Checkpoint path.
            num_actions: Action count of the env being evaluated.
            seed: PRNG seed for the throwaway init pass.
            fields: Adapter output for one representative frame; supplies the
                per-field routing the encoder is composed from.
            encoder_overrides: Branch overrides for the composite encoder.
            **config_kwargs: Extra :class:`R2DreamerConfig` fields (e.g. the
                architecture fields recovered from the run manifest).

        Returns:
            The agent, with the loaded checkpoint's ``step`` stashed as
            ``checkpoint_step`` (``-1`` if absent).
        """
        ckpt = load_policy_checkpoint(path)
        config = R2DreamerConfig(num_actions=num_actions, **config_kwargs)
        rng_key, init_key = jax.random.split(jax.random.PRNGKey(seed))
        del rng_key
        agent = cls(
            config, init_key, fields=fields, encoder_overrides=encoder_overrides
        )
        _assert_params_match(agent.params, ckpt["params"])
        agent.params = jax.tree.map(jnp.asarray, ckpt["params"])
        agent.slow_critic_params = jax.tree.map(jnp.asarray, ckpt["slow_critic_params"])
        agent.checkpoint_step = int(ckpt.get("step", -1))
        return agent

    def __init__(
        self,
        config: R2DreamerConfig,
        rng_key: jnp.ndarray,
        fields: AdapterOutput,
        encoder_overrides: Dict[str, Any] | None = None,
    ):
        self.cfg = config
        self.checkpoint_step = -1
        self.twohot = R2TwoHotDist(num_bins=config.twohot_bins)

        # ---- Instantiate Flax modules (for .apply) ----
        # ``fields`` (a sample adapter output) is the architecture description:
        # the routed composite encoder is built from the per-field Encoder
        # routing, and the same fields supply the observation the init forward
        # runs on.
        # Composition root: translate the config into branch overrides so the
        # run stays reproducible from the config alone; ``encoder_overrides``
        # (e.g. fusion_dim) win over the translation.
        self.encoder_mod = routed_encoder_from_fields(
            fields,
            conv_depth=config.encoder_depth,
            conv_kernel=config.encoder_kernel,
            conv_mults=config.encoder_mults,
            **(encoder_overrides or {}),
        )
        self.rssm_mod = rssm_from_config(config)

        # Forward the real first-frame observation to discover embed_size.
        # ``fields`` already holds live values in the acting layout, so
        # ``_batch_live_obs`` (which needs ``encoder_mod`` for ``global_keys``,
        # assigned above) is the single place that knows which fields take the
        # single-env batch dim - no zero-filled stand-in to keep in sync.
        rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
        init_obs = self._batch_live_obs(encoder_obs_from_fields(fields))
        enc_params = self.encoder_mod.init(k1, init_obs)
        embed = cast(jnp.ndarray, self.encoder_mod.apply(enc_params, init_obs))
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
        self._decoder_rgb_key: str | None = None

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

        # Built here, next to the bundles it joins, so the optional params never
        # travel as a possibly-None local.
        if config.decoder:
            # Which replay field the probe reconstructs: the adapter flags it
            # with ``decoder_target=True``.
            self._decoder_rgb_key = decoder_target_key(fields)
            rng_key, k_dec = jax.random.split(rng_key)
            self.decoder_mod = ConvDecoder(
                depth=config.encoder_depth,
                kernel_size=config.encoder_kernel,
                mults=config.encoder_mults,
            )
            params["decoder"] = self.decoder_mod.init(k_dec, feat0)
            self._modules["decoder"] = self.decoder_mod

        # ---- Optimizer: LaProp with linear warmup ----
        self.tx = laprop(
            lr=config.lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            warmup=config.warmup_steps,
        )
        opt_state = self.tx.init(params)

        # ---- Slow target critic (EMA) ----
        slow_critic_params = jax.tree.map(jnp.copy, params["critic"])

        # ---- Return EMA ----
        self.return_ema = ReturnEMA()
        ema_state = self.return_ema.init_state()

        self.train_state = R2DTrainState(
            params=params,
            opt_state=opt_state,
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

    def _batch_live_obs(self, encoder_obs: Any) -> Any:
        """Add the single-env batch dim to one live observation tree.

        Live (``buffer=False``) fields are one global event that the encoder
        broadcasts itself, so they must NOT gain a batch dim - the branch would
        see an extra axis and its shape check would fail.
        """
        if not isinstance(encoder_obs, Mapping):
            return jnp.asarray(encoder_obs)[None]
        global_keys = getattr(self.encoder_mod, "global_keys", ())
        return {
            key: jnp.asarray(value) if key in global_keys else jnp.asarray(value)[None]
            for key, value in encoder_obs.items()
            if key != "is_first"
        }

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
        batched_obs = self._batch_live_obs(encoder_obs)
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
        batched_obs = self._batch_live_obs(encoder_obs)
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
        # Bound to a local so the target key is narrowed for the whole body: all
        # three are set together when the probe is built, or none of them are.
        rgb_key = self._decoder_rgb_key
        if not self.cfg.decoder or self.decoder_mod is None or rgb_key is None:
            return None
        params = self.params
        B, T = replay_batch_shape(batch)
        embed = cast(
            jnp.ndarray,
            self.encoder_mod.apply(params["encoder"], self._encoder_obs(batch)),
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
        target = decoder_rgb_target(batch, rgb_key)
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
        _total_loss, aux = self._loss_fn(
            self.params,
            slow_critic_params=self.slow_critic_params,
            ema_state=self.ema_state,
            batch=batch,
            rng_key=rng_key,
        )
        metrics = dict(aux.metrics)
        metrics["total_loss"] = aux.agent_loss
        return {k: float(v) for k, v in metrics.items()}

    def _train_step_pure(self, state: R2DTrainState, batch, rng_key):
        """Pure-functional training step (JIT-able)."""
        params = state.params
        opt_state = state.opt_state
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

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # NaN guard: skip update if loss is non-finite (mirrors PyTorch GradScaler)
        is_finite = jnp.isfinite(total_loss)

        grads = agc(grads, params, clip=self.cfg.agc_clip, pmin=self.cfg.agc_pmin)
        updates, new_opt_state = self.tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_ema_state = self.return_ema.update(ema_state, aux.imag_returns)

        # Roll back to pre-update state on NaN/inf
        new_params = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), new_params, params
        )
        new_opt_state = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), new_opt_state, opt_state
        )
        new_slow = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old),
            updated_slow,
            slow_critic_params,
        )
        new_ema_state = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), new_ema_state, ema_state
        )

        metrics = aux.metrics
        metrics["opt_loss"] = total_loss
        metrics["total_loss"] = aux.agent_loss
        metrics["nan_skipped"] = 1.0 - is_finite.astype(jnp.float32)
        new_state = R2DTrainState(
            params=new_params,
            opt_state=new_opt_state,
            slow_critic_params=new_slow,
            ema_state=new_ema_state,
        )
        return new_state, metrics

    # ------------------------------------------------------------------
    # Composition root: shared forward + 3 sub-losses
    # ------------------------------------------------------------------

    def _encoder_obs(self, batch):
        """Assemble the encoder input from a replay batch.

        Global (``buffer=False``) fields are never stored per step; the live
        value rides on ``batch.global_feature`` and is merged back under its
        routed key so the encoder sees the same obs dict as at init time.

        Raises:
            ValueError: If the encoder routes a global key but the batch
                carries no global feature.
        """
        global_keys = getattr(self.encoder_mod, "global_keys", ())
        if not global_keys:
            return batch.obs
        if batch.global_feature is None:
            raise ValueError(
                f"encoder routes global keys {global_keys} but the sampled "
                "batch carries no global_feature — was the live field added "
                "to the replay buffer?"
            )
        return {**batch.obs, global_keys[0]: batch.global_feature}

    def _world_model_forward(self, params, batch, rng_key) -> WorldModelForward:
        """Encoder + posterior rollout + prior + features. Shared across sub-losses.

        Computing this once is essential: if each sub-loss recomputed `embed`,
        the encoder would receive doubled gradient signal and the
        `barlow_stop_grad` toggle would no longer mean what it claims.
        """
        cfg = self.cfg
        B, T = replay_batch_shape(batch)

        embed = cast(
            jnp.ndarray,
            self.encoder_mod.apply(params["encoder"], self._encoder_obs(batch)),
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

    def _loss_fn(self, params, *, slow_critic_params, ema_state, batch, rng_key):
        """Compose the world-model, behavior, and representation losses.

        Returns:
            (total_loss, aux) — `aux` carries metrics and the imagination
            returns used for the post-step `ReturnEMA` update.
        """
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
            decoder_rgb_key=self._decoder_rgb_key,
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

        aux = AgentLossAux(
            metrics=metrics,
            imag_returns=behavior_result.imag_returns.reshape(-1),
            agent_loss=agent_loss,
        )
        return total_loss, aux
