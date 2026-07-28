"""Single public entry point: composition root and run orchestrator.

``main`` parses the command line once, composes every instance (env, adapter,
collector, agent, logger), and then owns the run loop itself: ``inference``
runs exactly one env step every iteration, and in train mode ``train_gate``
decides - visibly, here - when gradient steps fire. Evaluation is the same
loop without the gradient steps, with parameters loaded from a checkpoint and
an episode budget instead of a step budget.

Everything variant-specific is read off one row of ``src.adapters.ADAPTERS``:
the env's render resolution, whether a frozen VGGT extractor is needed, the
branch overrides for the composite encoder, and which CLI knobs the variant
consumes. The architecture itself comes from one live adapter call on the
first frame - the adapter's routed fields tell the agent which encoder branch
consumes which observation. There is no encoder-type string to dispatch on.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import jax

from src.adapters import ADAPTERS
from src.adapters.contract import AdapterFn, AdapterOutput
from src.launch.random_policy import RandomPolicy
from src.buffer.replay_buffer import ReplayBuffer
from src.configs.config import LATENT_PRESETS, R2DreamerConfig, TrainerConfig
from src.environments.crafter import CrafterEnv
from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv
from src.environments.habitat_metrics import HabitatEpisodeMetrics
from src.launch.eval_artifacts import EvalRecorder, arch_overrides_from_manifest
from src.launch.parser import build_parser
from src.launch.session import RunLogger, run_session
from src.r2dreamer.agent import ActState, R2DreamerAgent, materialize_metrics
from src.r2dreamer.checkpointing import apply_resume, save_checkpoint
from src.r2dreamer.experience import AgentStep, ExperienceCollector, StepResult
from src.shared.dtypes import compute_jnp_dtype

ENVS = ("habitat", "crafter")

# Typed fallback for the optional ``RUN_FLAGS`` constant. A bare ``()`` default
# would make ``getattr`` yield ``tuple[()]``, whose element type is ``Never``,
# and every dict built from those flags would infer as ``dict[Never, Any]``.
_NO_RUN_FLAGS: tuple[str, ...] = ()

# Scratch run directory for an ad-hoc launch that names no --output_dir.
_DEFAULT_OUTPUT_DIR = "output/dev"


def resolve_adapter(name: str) -> type:
    """Look up one adapter variant's class.

    Args:
        name: Variant name, a key of ``src.adapters.ADAPTERS``.

    Returns:
        The adapter class, which declares its own render resolution, feature
        need, extractor settings and encoder overrides.

    Raises:
        KeyError: If the variant is unknown.
    """
    if name not in ADAPTERS:
        raise KeyError(f"Unknown adapter {name!r}. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[name]


def _variant_flags() -> set[str]:
    """CLI flags that no config reads and only some adapter variant consumes.

    Every flag some adapter claims, minus the ones that also name a config
    field: those (``output_dir``) are run-level context an adapter may read but
    does not own, and they reach the run through the configs whether or not the
    current variant claims them. What is left is owned by adapters alone, so a
    set one and claimed by nobody would reach nothing at all.
    """
    config_fields = {
        field.name
        for config_cls in (R2DreamerConfig, TrainerConfig)
        for field in fields(config_cls)
    }
    flags = {
        flag
        for cls in ADAPTERS.values()
        for flag in getattr(cls, "RUN_FLAGS", _NO_RUN_FLAGS)
    }
    return flags - config_fields


def _adapter_kwargs(
    adapter_cls: type, args: Any, *, output_dir: str | None
) -> dict[str, Any]:
    """Constructor keywords for one adapter, read off the CLI by name.

    The variant declares which flags it consumes (``RUN_FLAGS``) and receives
    them as same-named keywords, exactly as :func:`_config_from_args` fills a
    config dataclass. An unset flag is left out so the adapter's own default
    stands, and the launcher never learns which variant owns which knob.

    ``output_dir`` is the exception: it names a config field rather than a
    variant knob, so a claiming adapter gets the run directory the run actually
    writes to, not the raw flag.

    Args:
        adapter_cls: The variant's adapter class.
        args: Parsed CLI namespace.
        output_dir: Resolved run directory, or ``None`` for a rollout that owns
            no artifacts.

    Returns:
        Keyword arguments for the adapter constructor.

    Raises:
        ValueError: If a variant-scoped flag is set that this adapter does not
            claim. These knobs drive diagnostics, and a diagnostic that quietly
            does not run costs a whole cluster job to discover.
    """
    claimed = tuple(getattr(adapter_cls, "RUN_FLAGS", _NO_RUN_FLAGS))
    unclaimed = sorted(
        flag
        for flag in _variant_flags().difference(claimed)
        if getattr(args, flag, None) is not None
    )
    if unclaimed:
        raise ValueError(
            f"{adapter_cls.__name__} does not consume {unclaimed}; those flags "
            "belong to another adapter variant"
        )
    kwargs = {
        flag: getattr(args, flag)
        for flag in claimed
        if getattr(args, flag, None) is not None
    }
    if "output_dir" in claimed:
        kwargs.pop("output_dir", None)
        if output_dir is not None:
            kwargs["output_dir"] = output_dir
    return kwargs


def make_adapter(
    adapter_cls: type, args: Any, *, output_dir: str | None = None
) -> AdapterFn:
    """Build one adapter, with its own extractor when the variant needs one.

    Called once per collector, so no two rollouts share adapter or extractor
    state. The extractor's ``reset_mode`` (from the adapter's
    ``EXTRACTOR_KWARGS``) decides whether its streaming cache is wiped per
    episode or only on a scene switch; the adapter drives it from inside.

    Args:
        adapter_cls: The variant's adapter class.
        args: Parsed CLI namespace, source of the per-run knobs the variant
            claims through ``RUN_FLAGS``.
        output_dir: Resolved run directory handed to a variant that claims
            ``output_dir``; ``None`` for a rollout that writes no artifacts.

    Returns:
        The per-step adapter the collector calls.
    """
    kwargs = _adapter_kwargs(adapter_cls, args, output_dir=output_dir)
    if adapter_cls.NEEDS_FEATURES:
        # Imported lazily: loading VGGT weights costs seconds and GPU memory,
        # and the RGB baseline must not pay for it.
        from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor

        return adapter_cls(
            JAXVGGTFeatureExtractor(**dict(adapter_cls.EXTRACTOR_KWARGS)), **kwargs
        )
    return adapter_cls(**kwargs)


def make_env(
    env: str,
    *,
    curriculum: str,
    render_resolution: int,
    mode: str = "train",
    seed: int = 42,
    max_episode_steps: int = 500,
    semantic: bool = False,
) -> HabitatObjectNavEnv | CrafterEnv:
    """Build one env instance at the resolution the adapter needs.

    Args:
        env: Environment name (``habitat`` or ``crafter``).
        curriculum: Habitat curriculum level (``L1``..``L4``); crafter ignores
            it, so a level is always safe to pass.
        render_resolution: Frame side length the adapter consumes.
        mode: Episode split (``train`` or ``eval``).
        seed: Env seed.
        max_episode_steps: Habitat per-episode step budget.
        semantic: Whether to render habitat's semantic sensor.

    Returns:
        The env instance.

    Raises:
        ValueError: If the env name is unknown.
    """
    if env == "habitat":
        return HabitatObjectNavEnv(
            config=HabitatEnvConfig(
                obs_shape=(render_resolution, render_resolution, 3),
                max_episode_steps=max_episode_steps,
                reward_type="geodesic_delta",
                curriculum=curriculum,
                mode=mode,
                semantic=semantic,
            ),
            seed=seed,
        )
    if env == "crafter":
        return CrafterEnv(size=(64, 64), seed=seed)
    raise ValueError(f"Unknown environment: {env!r}. Available: {list(ENVS)}")


def _fresh_env(
    *, adapter_cls: type, args: Any
) -> HabitatObjectNavEnv | CrafterEnv:
    """Build the run's env, sized by the adapter unless a flag overrides it."""
    return make_env(
        args.env,
        curriculum=args.curriculum,
        render_resolution=(
            args.render_resolution
            if args.render_resolution is not None
            else adapter_cls.RENDER_RESOLUTION
        ),
        mode=args.mode,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        semantic=args.semantic,
    )


def _renamed_agent_flags(args: Any) -> dict[str, Any]:
    """Agent-config values whose CLI flag is named differently or derived.

    Everything whose flag already matches its config field is picked up by
    :func:`_config_from_args`; only these need translating.
    """
    renamed: dict[str, Any] = {}
    for flag, field in (
        ("actor_loss_weight", "scale_policy"),
        ("value_loss_weight", "scale_value"),
        ("repval_loss_weight", "scale_repval"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            renamed[field] = value
    if getattr(args, "barlow_grad_to_encoder", False):
        renamed["barlow_stop_grad"] = False
    if getattr(args, "compute_dtype", None) is not None:
        renamed["compute_dtype"] = {"bf16": "bfloat16", "fp16": "float16"}.get(
            args.compute_dtype, args.compute_dtype
        )
    return renamed


def _agent_config(
    *,
    args: Any,
    num_actions: int,
    output_dir: str,
) -> R2DreamerConfig:
    """Agent config: size preset, then CLI fields, then renamed flags.

    The model-size ablation (3D-50) picks RSSM widths from ``LATENT_PRESETS``;
    an explicitly passed ``--deter_size`` and friends still win over the preset.
    """
    return _config_from_args(
        R2DreamerConfig,
        args,
        defaults=LATENT_PRESETS.get(getattr(args, "latent_preset", "12m"), {}),
        adapter=args.adapter,
        num_actions=num_actions,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        logdir=output_dir,
        **_renamed_agent_flags(args),
    )


def _config_from_args(
    config_cls: type,
    args: Any,
    *,
    defaults: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Build a config dataclass by matching its field names against CLI flags.

    The SLURM layer already owns the per-run YAML
    (``scripts/slurm/configs/*.yaml``, rendered into these flags), so the job
    here is only to stop hand-copying every field: a flag named like a config
    field fills it, an unset flag leaves the field's default, and a new config
    field with a matching flag needs no change here.

    Precedence, lowest first: ``defaults`` (e.g. a size preset), flags that
    match a field name, then ``overrides``.

    Args:
        config_cls: The config dataclass to build.
        args: Parsed CLI namespace.
        defaults: Values a matching CLI flag is allowed to override.
        **overrides: Values that win over everything read from ``args``.

    Returns:
        The constructed config instance.
    """
    from_flags = {
        field.name: getattr(args, field.name)
        for field in fields(config_cls)
        if getattr(args, field.name, None) is not None
    }
    return config_cls(**{**(defaults or {}), **from_flags, **overrides})


def _trainer_config(*, args: Any, output_dir: str) -> TrainerConfig:
    """Loop-control config: CLI fields plus the few that are renamed or derived."""
    wandb_tags = (
        [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
    )
    overrides: dict[str, Any] = {}
    if wandb_tags is not None:
        overrides["wandb_tags"] = wandb_tags
    return _config_from_args(
        TrainerConfig,
        args,
        output_dir=output_dir,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        # An empty --wandb_project "" disables W&B for the run.
        wandb_project=args.wandb_project or None,
        # Opt-in via the SLURM launcher: hard-exit a completed run before the
        # habitat_sim GL teardown can SIGABRT and poison the exit code.
        hard_exit_on_finish=os.environ.get("R2DREAMER_HARD_EXIT_ON_FINISH") == "1",
        **overrides,
    )


@dataclass(frozen=True)
class ComposedRun:
    """What a run needs before its loop starts, whatever the loop then does.

    Attributes:
        adapter_cls: Resolved adapter class.
        collector: Env + adapter, plus a replay buffer when the run records.
        fields: The first frame's routed fields - the architecture description
            the agent's encoder was composed from. ``None`` for the random
            baseline, which has no encoder to compose.
        agent: The agent: freshly initialised, restored from a checkpoint, or
            the random baseline policy.
        agent_config: The config the agent was built with. The random baseline
            carries the defaults, recorded for the manifest only.
    """

    adapter_cls: type
    collector: ExperienceCollector
    fields: AdapterOutput | None
    agent: R2DreamerAgent | RandomPolicy
    agent_config: R2DreamerConfig


def compose_run(args: Any, *, output_dir: str) -> ComposedRun:
    """Compose env, adapter, collector and agent for one run.

    Training and evaluation do not build different agents - they differ only
    in where the parameters come from. The architecture is discovered the same
    way in both: one adapter call on the first frame, whose routed fields
    compose the encoder. Only the parameter source branches, so any change to
    the routing plumbing reaches both workflows at once.

    Args:
        args: Parsed CLI namespace (the one parser's).
        output_dir: Run directory. Handed to a variant that claims it, and
            recorded as a fresh config's ``logdir``.

    Returns:
        The composed pieces.
    """
    training = args.mode == "train"
    checkpoint: str | None = None if training else args.checkpoint
    random_policy = bool(not training and args.random)
    adapter_cls = resolve_adapter(args.adapter)
    env_instance = _fresh_env(adapter_cls=adapter_cls, args=args)
    try:
        num_actions = env_instance.num_actions
        agent_config = None
        buffer = None
        if training:
            agent_config = _agent_config(
                args=args, num_actions=num_actions, output_dir=output_dir
            )
            buffer = ReplayBuffer(
                capacity=agent_config.buffer_capacity,
                num_actions=num_actions,
                float_dtype=compute_jnp_dtype(agent_config.compute_dtype),
            )
        observe = make_adapter(adapter_cls, args, output_dir=output_dir)
        collector = ExperienceCollector(
            env=env_instance,
            observe=observe,
            num_actions=num_actions,
            buffer=buffer,
            # End-of-run health metrics are the recording rollout's: a variant
            # that keeps accumulator state (e.g. the voxel buffer) reports it
            # through this hook, and a variant without one reports nothing.
            diagnostics_fn=(
                getattr(observe, "diagnostics", None) if buffer is not None else None
            ),
            # One metrics source for both workflows: eval episodes are scored
            # by the same HabitatEpisodeMetrics the training episodes log.
            episode_metrics_fn=(
                HabitatEpisodeMetrics(env_instance, track_collision_rate=not training)
                if args.env == "habitat"
                else None
            ),
            # Eval owns its episode boundaries: an auto-reset would advance the
            # pinned episode order behind the recorder's back.
            auto_reset=training,
        )

        agent: R2DreamerAgent | RandomPolicy
        first_fields: AdapterOutput | None = None
        if random_policy:
            agent = RandomPolicy(num_actions=num_actions, seed=args.seed)
            agent_config = R2DreamerConfig(
                adapter=args.adapter, num_actions=num_actions
            )
        else:
            # One adapter call on the first frame supplies the routing and
            # field shapes the agent needs at init. The loop resets again when
            # it starts; that second reset costs one episode of the iterator
            # and keeps the collector the only owner of the rollout.
            first_fields = collector.reset_fields()
            encoder_overrides = dict(adapter_cls.ENCODER_OVERRIDES)
            if checkpoint is not None:
                agent = R2DreamerAgent.from_checkpoint(
                    checkpoint,
                    num_actions=num_actions,
                    seed=args.seed,
                    fields=first_fields,
                    encoder_overrides=encoder_overrides,
                    adapter=args.adapter,
                    **arch_overrides_from_manifest(checkpoint),
                )
                agent_config = agent.cfg
            else:
                assert agent_config is not None  # set above for training runs
                _rng_key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
                agent = R2DreamerAgent(
                    agent_config,
                    init_key,
                    fields=first_fields,
                    encoder_overrides=encoder_overrides,
                )
                agent_config = agent.cfg
    except BaseException:
        # A half-composed run must not leak the env: an abandoned habitat sim
        # keeps its GL context, and the next env build then fails for a reason
        # that has nothing to do with the actual error.
        env_instance.close()
        raise

    assert agent_config is not None
    return ComposedRun(
        adapter_cls=adapter_cls,
        collector=collector,
        fields=first_fields,
        agent=agent,
        agent_config=agent_config,
    )


def inference(
    agent: R2DreamerAgent | RandomPolicy,
    agent_step: AgentStep,
    act_state: ActState,
    collector: ExperienceCollector,
    rng_key: jax.Array,
    *,
    training: bool,
) -> tuple[int, StepResult, ActState, jax.Array]:
    """Run exactly one env step: act, step the env, record the transition.

    The ``int()`` on the action is the one blocking device sync per step:
    habitat's ``env.step`` only wraps ``int``/``np.integer`` into its action
    dict, and in a single-env loop there is nothing to overlap the sync with.

    Args:
        agent: Policy providing the functional ``act``.
        agent_step: Current observation (encoder inputs + boundary flag).
        act_state: RSSM acting carry; reset via ``is_first`` inside ``act``.
        collector: Rollout owner; records to replay when it has a buffer.
        rng_key: PRNG key; split once per step.
        training: Sampled (train) vs greedy (eval) action selection.

    Returns:
        ``(action, step_result, new_act_state, advanced_rng_key)``.
    """
    rng_key, act_key = jax.random.split(rng_key)
    action_array, act_state = agent.act(
        agent.params,
        agent_step.encoder_obs,
        agent_step.is_first,
        act_state,
        act_key,
        training,
    )
    action = int(action_array)
    result = collector.step(action)
    return action, result, act_state, rng_key


def prefill(
    collector: ExperienceCollector,
    *,
    num_steps: int,
    num_actions: int,
    rng_key: jax.Array,
) -> jax.Array:
    """Fill the replay buffer with uniformly random actions.

    The collector's reset fires the scene-aware on_episode_reset callback
    (VGGT PERSIST_SCENE saves/restores per scene) even though prefill discards
    reset observations for replay purposes. ``summarize=False`` keeps the
    episode metrics fn (and its rolling trackers) untouched during random
    collection.

    Args:
        collector: Recording collector to fill.
        num_steps: Number of random env steps.
        num_actions: Discrete action-space size to sample from.
        rng_key: JAX PRNG key; split once per step.

    Returns:
        The advanced PRNG key.
    """
    print(f"Prefilling {num_steps} steps...")
    collector.reset()
    for _ in range(num_steps):
        rng_key, action_key = jax.random.split(rng_key)
        action = int(jax.random.randint(action_key, (), 0, num_actions))
        collector.step(action, summarize=False)
    return rng_key


def overfit(
    agent: R2DreamerAgent,
    collector: ExperienceCollector,
    tcfg: TrainerConfig,
    logger: RunLogger,
    rng_key: jax.Array,
) -> jax.Array:
    """Freeze one sampled batch and call train_step on it repeatedly.

    Proves the full stack (encoder -> RSSM -> heads) can memorise a real
    trajectory. If loss does not drop, the gradient path is broken - no amount
    of production wall-clock will save the run. This branch never enters the
    run loop: no env rollouts, no checkpointing.

    Args:
        agent: Agent under diagnosis.
        collector: Recording collector holding at least one prefilled batch.
        tcfg: Overfit knobs (overfit_steps, overfit_batch_size, ...).
        logger: Metric sinks.
        rng_key: JAX PRNG key.

    Returns:
        The advanced PRNG key.

    Raises:
        RuntimeError: If the buffer is too small or the loss-drop verification
            fails.
        ValueError: If ``overfit_steps`` is below one.
    """
    buffer_size = collector.buffer_size
    if buffer_size < tcfg.overfit_batch_size * tcfg.overfit_seq_len:
        raise RuntimeError(
            f"overfit_one_batch: buffer too small "
            f"({buffer_size} < {tcfg.overfit_batch_size}*{tcfg.overfit_seq_len}). "
            f"Increase --prefill."
        )
    if tcfg.overfit_steps < 1:
        raise ValueError(f"overfit_steps must be >= 1, got {tcfg.overfit_steps}")

    # Sample once, freeze, reuse.
    batch = collector.sample(tcfg.overfit_batch_size, tcfg.overfit_seq_len)
    print(
        f"Overfit mode: cached batch "
        f"B={tcfg.overfit_batch_size} T={tcfg.overfit_seq_len}; "
        f"running {tcfg.overfit_steps} train_step iterations."
    )

    logger.start_timing(0)
    first_loss = last_loss = 0.0
    for step in range(tcfg.overfit_steps):
        rng_key, train_key = jax.random.split(rng_key)
        agent.train_state, device_metrics = agent.train_step(
            agent.train_state, batch, train_key
        )
        metrics = materialize_metrics(device_metrics)
        last_loss = metrics["total_loss"]
        if step == 0:
            first_loss = last_loss

        if step % tcfg.log_every == 0 or step == tcfg.overfit_steps - 1:
            logger.log_train_metrics(metrics, step)

    loss_drop = (first_loss - last_loss) / max(abs(first_loss), 1e-12)
    logger.write_row(tcfg.overfit_steps - 1, "verify/overfit_loss_drop", loss_drop)
    logger.write_row(
        tcfg.overfit_steps - 1,
        "verify/overfit_pass",
        float(loss_drop >= tcfg.overfit_min_loss_drop),
    )
    print(
        f"Overfit verify: first_loss={first_loss:.6g} "
        f"last_loss={last_loss:.6g} drop={loss_drop:.1%} "
        f"required={tcfg.overfit_min_loss_drop:.1%}"
    )
    if loss_drop < tcfg.overfit_min_loss_drop:
        raise RuntimeError(
            "overfit_one_batch verification failed: total_loss did not drop "
            f"by at least {tcfg.overfit_min_loss_drop:.1%}. "
            "Do not launch a production run until this passes."
        )
    return rng_key


def _should_record_video(
    tcfg: TrainerConfig,
    logger: RunLogger,
    collector: ExperienceCollector,
    step: int,
    next_video_step: int,
) -> bool:
    return (
        logger.wandb_active
        and tcfg.video_log_every > 0
        and tcfg.video_log_episodes > 0
        and step >= next_video_step
        and collector.supports_video
    )


def run_loop(
    agent: R2DreamerAgent | RandomPolicy,
    collector: ExperienceCollector,
    acfg: R2DreamerConfig,
    tcfg: TrainerConfig,
    logger: RunLogger,
    rng_key: jax.Array,
    *,
    training: bool,
    episodes: int = 0,
    max_episode_steps: int = 500,
    recorder: EvalRecorder | None = None,
    start_step: int = 0,
) -> jax.Array:
    """The one run loop: an env step every iteration, gradients when due.

    Train mode runs ``start_step .. tcfg.total_steps`` env steps; the
    train-ratio gate accumulates fractional credit per env step and fires
    ``agent.train_step`` whenever a full gradient step is due. Eval mode runs
    the same loop without the gate and stops after ``episodes`` finished
    episodes; the recorder captures per-episode artifacts.

    Args:
        agent: Policy (trained agent or random baseline).
        collector: Rollout owner (auto-reset in train, manual reset in eval).
        acfg: Agent config (batch_size, seq_len, train_ratio, decoder flag).
        tcfg: Loop-control config (total_steps, cadences).
        logger: Metric/video sinks.
        rng_key: JAX PRNG key threaded through acting and training.
        training: Whether gradient steps fire and checkpoints save.
        episodes: Eval episode budget (eval mode only).
        max_episode_steps: Per-episode cap, bounding the eval loop.
        recorder: Eval artifact recorder (eval mode only).
        start_step: First loop step (non-zero when resuming a train run).

    Returns:
        The advanced PRNG key.
    """
    # The random baseline is eval-only; narrowing here keeps the train branch
    # typed against the real agent instead of the union.
    trained = agent if isinstance(agent, R2DreamerAgent) else None
    if training and trained is None:
        raise TypeError("train mode requires an R2DreamerAgent, not RandomPolicy")

    agent_step = collector.reset()
    act_state = agent.initial_act_state()
    logger.start_timing(start_step)
    batch_steps = acfg.batch_size * acfg.seq_len
    train_credit = 0.0
    log_pending = False
    video_next_step = start_step
    episodes_done = 0

    if training:
        total_steps = tcfg.total_steps
        print(f"Training from step {start_step} to {total_steps}...")
        if _should_record_video(tcfg, logger, collector, start_step, video_next_step):
            collector.start_video_capture()
    else:
        # The episode budget is the real bound; the step bound only guards
        # against an env that never reports done.
        total_steps = episodes * (max_episode_steps + 1)
        print(f"Evaluating {episodes} episodes...")
        if recorder is not None:
            recorder.start_episode()
            if recorder.record_video:
                collector.start_video_capture()

    for step in range(start_step, total_steps):
        action, result, act_state, rng_key = inference(
            agent, agent_step, act_state, collector, rng_key, training=training
        )
        agent_step = result.agent_step

        if training:
            if result.episode is not None:
                logger.log_episode(result.episode, step)
                if result.episode.video_frames is not None:
                    logger.log_video(
                        "train/episode_video", result.episode.video_frames, step
                    )
                    video_next_step = step + max(1, tcfg.video_log_every)
                if _should_record_video(
                    tcfg, logger, collector, step + 1, video_next_step
                ):
                    collector.start_video_capture()

            # --- Train-ratio gate: visible here, in the orchestrator ---
            assert trained is not None  # guaranteed by the entry check
            if collector.buffer_size >= batch_steps:
                train_credit += acfg.train_ratio / batch_steps
                if step % tcfg.log_every == 0:
                    log_pending = True
                # With fractional credit the update and the log cadence can
                # have opposite parity, so a due log waits for the next real
                # update.
                will_log = log_pending and train_credit >= 1.0
                batch = None
                metrics = None
                while train_credit >= 1.0:
                    rng_key, train_key = jax.random.split(rng_key)
                    batch = collector.sample(acfg.batch_size, acfg.seq_len)
                    trained.train_state, metrics = trained.train_step(
                        trained.train_state, batch, train_key, materialize=will_log
                    )
                    train_credit -= 1.0

                if will_log and metrics is not None:
                    logger.log_train_metrics(materialize_metrics(metrics), step)
                    log_pending = False
                    if acfg.decoder and batch is not None and logger.wandb_active:
                        pair = trained.reconstruct(batch)
                        if pair is not None:
                            target, recon = jax.device_get(pair)
                            logger.log_reconstructions(target, recon, step)

            if (step + 1) % tcfg.checkpoint_every == 0:
                save_checkpoint(trained, step + 1, tcfg.output_dir)
        else:
            if recorder is not None:
                recorder.record_step(action)
            if result.done:
                summary = collector.finish_episode()
                logger.log_episode(summary, step)
                if recorder is not None:
                    recorder.finish_episode(summary)
                episodes_done += 1
                if episodes_done >= episodes:
                    break
                agent_step = collector.reset()
                act_state = agent.initial_act_state()
                if recorder is not None:
                    recorder.start_episode()
                    if recorder.record_video:
                        collector.start_video_capture()

    return rng_key


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line once, compose the run, and own its loop."""
    args = build_parser().parse_args(
        list(argv) if argv is not None else sys.argv[1:]
    )
    training = args.mode == "train"
    if not training and not args.random and args.checkpoint is None:
        raise ValueError("eval mode requires --checkpoint (or --random)")
    output_dir = args.output_dir or _DEFAULT_OUTPUT_DIR

    composed = compose_run(args, output_dir=output_dir)
    agent, collector = composed.agent, composed.collector
    if isinstance(agent, R2DreamerAgent):
        print(f"adapter {args.adapter!r} -> embed_size {agent.embed_size}")
        if not training:
            print(f"Loaded checkpoint from step {agent.checkpoint_step}")

    tcfg = _trainer_config(args=args, output_dir=output_dir)
    resume_step = 0
    if training and tcfg.resume_from is not None:
        assert isinstance(agent, R2DreamerAgent)
        resume_step = apply_resume(agent, tcfg.resume_from)

    logger = RunLogger(composed.agent_config, tcfg, resume=resume_step > 0)
    recorder = None
    if not training:
        recorder = EvalRecorder(
            env=collector.env,
            output_dir=output_dir,
            render_topdown=args.render_topdown,
            video_episodes=args.log_video_episodes,
            wandb_module=logger.wandb_module,
        )

    rng_key = jax.random.PRNGKey(tcfg.seed)
    with run_session(logger, collector, hard_exit=tcfg.hard_exit_on_finish):
        if training:
            assert isinstance(agent, R2DreamerAgent)
            if resume_step > 0:
                # The trained policy re-collects on-policy transitions until
                # the buffer covers a batch; no random prefill on resume.
                print(f"Resume mode: skipping prefill, jumping to {resume_step}")
            else:
                rng_key = prefill(
                    collector,
                    num_steps=tcfg.prefill_steps,
                    num_actions=composed.agent_config.num_actions,
                    rng_key=rng_key,
                )
            if tcfg.overfit_one_batch:
                overfit(agent, collector, tcfg, logger, rng_key)
            else:
                run_loop(
                    agent,
                    collector,
                    composed.agent_config,
                    tcfg,
                    logger,
                    rng_key,
                    training=True,
                    start_step=resume_step,
                )
                logger.log_adapter_summary(collector.diagnostics(), tcfg.total_steps)
                logger.close_metrics_file()
                save_checkpoint(agent, tcfg.total_steps, tcfg.output_dir)
        else:
            run_loop(
                agent,
                collector,
                composed.agent_config,
                tcfg,
                logger,
                rng_key,
                training=False,
                episodes=args.episodes,
                max_episode_steps=args.max_episode_steps,
                recorder=recorder,
            )
            assert recorder is not None
            recorder.finalize(checkpoint=args.checkpoint, random_agent=args.random)


if __name__ == "__main__":
    main(sys.argv[1:])
