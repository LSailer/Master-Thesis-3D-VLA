"""R2DreamerAgent — composition root.

The agent is a thin orchestrator: it owns parameters, the LaProp optimizer,
the slow-target EMA, the acting state, and the JIT'd train/eval entry points.
Three collaborator modules do the heavy lifting so this file stays a facade:

    agent_modules.py — Flax module construction + parameter-bundle init
    agent_optim.py   — LaProp optimizer construction
    agent_loss.py    — total-objective composition (world-model + behavior +
                        representation losses, plus per-encoder diagnostics)

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
from pathlib import Path
from typing import Any, Dict, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax

from src.buffer import ReplayBatch
from src.configs.config import R2DreamerConfig
from src.r2dreamer.agent_loss import compose_agent_loss
from src.r2dreamer.agent_modules import build_agent_modules
from src.r2dreamer.agent_optim import make_optimizer
from src.r2dreamer.decoder_targets import decoder_rgb_target, replay_batch_shape
from src.r2dreamer.encoders.shape_utils import batch_live_observation
from src.shared.optim import agc

from .behavior.return_ema import ReturnEMA
from .checkpointing import load_checkpoint
from .learning_types import WorldModelForward
from .observation_preparation.contracts import recover_encoder_input_contract
from .world_model.heads import R2TwoHotDist


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
        obs_shape: tuple[int, ...] | dict[str, tuple[int, ...]] | None = None,
        num_actions: int,
        seed: int,
        **config_kwargs: Any,
    ) -> "R2DreamerAgent":
        """Build an agent and load ``params`` + ``slow_critic_params`` from disk.

        Extra ``config_kwargs`` flow into :class:`R2DreamerConfig` so callers
        that need ``encoder_type`` / ``encoder_module_cls`` (e.g. evaluate)
        can pass them through. When the checkpoint contains a durable Encoder
        Input Contract snapshot, missing encoder config is recovered from it.
        The loaded checkpoint's ``step`` is stashed on the returned agent as
        ``checkpoint_step`` (``-1`` if absent).
        """
        ckpt = load_policy_checkpoint(path)
        contract_snapshot = ckpt.get("encoder_input_contract")
        if contract_snapshot is not None:
            contract = recover_encoder_input_contract(contract_snapshot)
            if obs_shape is None:
                obs_shape = contract.encoder_input.buffer_shape()
            requested_type = config_kwargs.get("encoder_type")
            if requested_type is not None and requested_type != contract.encoder_type:
                raise ValueError(
                    "checkpoint encoder contract mismatch: requested "
                    f"{requested_type!r}, checkpoint has {contract.encoder_type!r}"
                )
            requested_shape = obs_shape
            contract_shape = contract.encoder_input.buffer_shape()
            if requested_shape != contract_shape:
                raise ValueError(
                    "checkpoint encoder shape mismatch: requested "
                    f"{requested_shape!r}, checkpoint has {contract_shape!r}"
                )
            config_kwargs["encoder_type"] = contract.encoder_type
            config_kwargs["encoder_module_cls"] = contract.encoder_module_cls
            config_kwargs["encoder_input_contract"] = contract_snapshot
        if obs_shape is None:
            raise ValueError(
                "obs_shape must be provided when checkpoint has no Encoder Input "
                "Contract snapshot"
            )
        config = R2DreamerConfig(
            obs_shape=obs_shape,
            num_actions=num_actions,
            **config_kwargs,
        )
        rng_key = jax.random.PRNGKey(seed)
        rng_key, init_key = jax.random.split(rng_key)
        agent = cls(config, init_key)
        agent.params = jax.tree.map(jnp.asarray, ckpt["params"])
        agent.slow_critic_params = jax.tree.map(jnp.asarray, ckpt["slow_critic_params"])
        agent.checkpoint_step = int(ckpt.get("step", -1))
        return agent

    def __init__(self, config: R2DreamerConfig, rng_key: jnp.ndarray):
        self.cfg = config
        self.checkpoint_step = -1
        self.twohot = R2TwoHotDist(num_bins=config.twohot_bins)

        # ---- Flax modules + initialized params (agent_modules.py) ----
        built = build_agent_modules(config, rng_key)
        self.encoder_mod = built.encoder_mod
        self.rssm_mod = built.rssm_mod
        self.proj_mod = built.proj_mod
        self.reward_mod = built.reward_mod
        self.cont_mod = built.cont_mod
        self.actor_mod = built.actor_mod
        self.critic_mod = built.critic_mod
        self.decoder_mod = built.decoder_mod
        self.embed_size = built.embed_size
        self._modules = built.modules
        params = built.params

        # ---- Optimizer: LaProp with linear warmup (agent_optim.py) ----
        self.tx = make_optimizer(config)
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
        action_int, self._act_state = self._jit_act_with_state.__call__(
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
        action_int, new_state = self._jit_act_with_state.__call__(
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

        Returns ``(target, recon)`` as JAX arrays ``(B*T, 3, 64, 64)`` in
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

    def train_step(self, batch: ReplayBatch , rng_key: jnp.ndarray) -> Dict[str, float]:
        """One LaProp step on `batch`. Returns Python-float metrics."""
        self.train_state, metrics = self._jitted_train_step.__call__(
            self.train_state,
            batch,
            rng_key,
        )
        return {k: float(v) for k, v in metrics.items()}

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
    # Composition root: shared forward + total loss (agent_loss.py)
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

    def _loss_fn(self, params, *, slow_critic_params, ema_state, batch, rng_key):
        """Compose the world-model, behavior, and representation losses.

        Returns:
            (total_loss, aux) — `aux` carries metrics and the imagination
            returns used for the post-step `ReturnEMA` update.
        """
        return compose_agent_loss(
            cfg=self.cfg,
            modules=self._modules,
            twohot=self.twohot,
            return_ema=self.return_ema,
            world_model_forward=self._world_model_forward,
            params=params,
            slow_critic_params=slow_critic_params,
            ema_state=ema_state,
            batch=batch,
            rng_key=rng_key,
        )
