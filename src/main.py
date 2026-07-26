"""Single public entry point: the composition root for training and evaluation.

Everything variant-specific is read off one row of ``src.adapters.ADAPTERS``:
the env's render resolution, whether a frozen VGGT extractor is needed, the
branch overrides for the composite encoder, and which train-CLI knobs the
variant consumes. The architecture itself comes from one live adapter call on
the first frame - the adapter's routed fields tell the agent which encoder
branch consumes which observation. There is no encoder-type string to dispatch
on.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import jax

from src.adapters import ADAPTERS
from src.adapters.contract import AdapterFn, AdapterOutput
from src.buffer.replay_buffer import ReplayBuffer
from src.configs.config import LATENT_PRESETS, R2DreamerConfig, TrainerConfig
from src.environments.crafter import CrafterEnv
from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv
from src.environments.habitat_metrics import HabitatEpisodeMetrics
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.experience import Env, ExperienceCollector
from src.r2dreamer.launch.evaluate import (
    arch_overrides_from_manifest,
    resolve_eval_settings,
    run_evaluation,
)
from src.r2dreamer.launch.loops import apply_resume, run_training
from src.r2dreamer.launch.parser import _build_parser_eval, _build_parser_train
from src.shared.dtypes import compute_jnp_dtype

ENVS = ("habitat", "crafter")

# Scratch run directory for an ad-hoc train launch that names neither a
# --output_dir flag nor a shim kwarg.
_DEFAULT_OUTPUT_DIR = "output/dev"


@dataclass
class TrainingRun:
    """Handle returned to programmatic (notebook) callers after a run.

    Attributes:
        agent: The trained agent (params/opt state as of run end).
        experience: The train collector (env + adapter + replay buffer).
        val_experience: The val collector, when validation was wired.
        agent_config: Effective agent config.
        trainer_config: Effective loop-control config.
    """

    agent: Any
    experience: ExperienceCollector
    val_experience: ExperienceCollector | None
    agent_config: R2DreamerConfig
    trainer_config: TrainerConfig


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
        flag for cls in ADAPTERS.values() for flag in getattr(cls, "RUN_FLAGS", ())
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
    writes to, not the raw flag. The flag is only one of the sources that
    directory is resolved from, and a preset launch supplies it as a kwarg.

    Args:
        adapter_cls: The variant's adapter class.
        args: Parsed CLI namespace. Flags missing from it (the eval parser
            defines fewer) count as unset.
        output_dir: Resolved run directory, or ``None`` for a rollout that owns
            no artifacts (the validation collector).

    Returns:
        Keyword arguments for the adapter constructor.

    Raises:
        ValueError: If a variant-scoped flag is set that this adapter does not
            claim. These knobs drive diagnostics, and a diagnostic that quietly
            does not run costs a whole cluster job to discover.
    """
    claimed = tuple(getattr(adapter_cls, "RUN_FLAGS", ()))
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


def _without_variant_flags(args: Any) -> Any:
    """Return ``args`` with every variant-scoped knob cleared.

    The validation collector builds a second adapter from the same namespace.
    Run diagnostics belong to the recording rollout: a val adapter accumulating
    a different episode set would otherwise write over the train artifacts.
    """
    stripped = copy.copy(args)
    for flag in _variant_flags():
        setattr(stripped, flag, None)
    return stripped


def make_adapter(
    adapter_cls: type, args: Any, *, output_dir: str | None = None
) -> AdapterFn:
    """Build one adapter, with its own extractor when the variant needs one.

    Called once per collector, so train and validation never share adapter or
    extractor state. The extractor's ``reset_mode`` (from the adapter's
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
    curriculum: str | None,
    render_resolution: int,
    mode: str = "train",
    seed: int = 0,
    max_episode_steps: int = 500,
    semantic: bool = False,
) -> HabitatObjectNavEnv | CrafterEnv:
    """Build one env instance at the resolution the adapter needs.

    Args:
        env: Environment name (``habitat`` or ``crafter``).
        curriculum: Habitat curriculum level (``L1``..``L4``); ignored by crafter.
        render_resolution: Frame side length the adapter consumes.
        mode: Curriculum split (``train`` or ``eval``).
        seed: Env seed.
        max_episode_steps: Habitat episode step budget.
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
                curriculum=curriculum if curriculum is not None else "L1",
                mode=mode,
                semantic=semantic,
            ),
            seed=seed,
        )
    if env == "crafter":
        return CrafterEnv(size=(64, 64), seed=seed)
    raise ValueError(f"Unknown environment: {env!r}. Available: {list(ENVS)}")


def _fresh_env(
    *,
    env: str,
    adapter_cls: type,
    args: Any,
    curriculum: str | None,
    mode: str,
) -> HabitatObjectNavEnv | CrafterEnv:
    """Build one env for a collector to own, sized by the adapter.

    Every collector owns its own env instance (train, validation, evaluation),
    so the "which resolution, which sensors, which seed" plumbing is resolved
    here once instead of at each of the three call sites.

    Args:
        env: Environment name.
        adapter_cls: The variant's adapter class, whose ``RENDER_RESOLUTION``
            applies unless a CLI flag overrides it.
        args: Parsed CLI namespace. Only ``seed`` is required of it; the flags
            the two parsers do not share are read defensively, so a train and an
            eval namespace are both acceptable.
        curriculum: Effective habitat curriculum level.
        mode: Curriculum split (``train`` or ``eval``).

    Returns:
        The env instance.
    """
    return make_env(
        env,
        curriculum=curriculum,
        render_resolution=(
            getattr(args, "render_resolution", None) or adapter_cls.RENDER_RESOLUTION
        ),
        mode=mode,
        seed=args.seed,
        # Only the eval parser offers the semantic sensor; training never
        # renders it.
        semantic=getattr(args, "semantic", False),
    )


def _effective_curriculum(*, env: str, args: Any, curriculum: str | None) -> str | None:
    if env not in ENVS:
        raise KeyError(f"Unknown env {env!r}. Available: {list(ENVS)}")
    effective = args.curriculum if args.curriculum is not None else curriculum
    has_curriculum = effective is not None or args.curriculum_path is not None
    if env == "habitat" and not has_curriculum:
        raise ValueError(
            "Habitat env requires a curriculum. Pass curriculum='L1'..'L4', "
            "--curriculum, or --curriculum_path."
        )
    if env == "crafter" and has_curriculum:
        raise ValueError("Crafter env does not use a curriculum.")
    return effective


def _effective_run_metadata(
    *,
    args: Any,
    output_dir: str | None,
    wandb_name: str | None,
    wandb_tags: list[str] | None,
) -> tuple[str, str | None, list[str]]:
    # CLI value (non-None) wins over shim kwarg; neither set falls back to the
    # scratch dir, so an ad-hoc `python -m src.main train` still has somewhere
    # to write.
    eff_output_dir = args.output_dir or output_dir or _DEFAULT_OUTPUT_DIR
    eff_wandb_name = args.wandb_name if args.wandb_name is not None else wandb_name
    eff_wandb_tags: list[str] = list(wandb_tags) if wandb_tags is not None else []
    if args.wandb_tags:
        eff_wandb_tags.extend(t.strip() for t in args.wandb_tags.split(","))
    return eff_output_dir, eff_wandb_name, eff_wandb_tags


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
    adapter: str,
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
        adapter=adapter,
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


def _trainer_config(
    *,
    args: Any,
    output_dir: str,
    wandb_name: str | None,
    wandb_tags: list[str],
) -> TrainerConfig:
    """Loop-control config: CLI fields plus the few that are renamed or derived."""
    return _config_from_args(
        TrainerConfig,
        args,
        output_dir=output_dir,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        wandb_name=wandb_name,
        wandb_tags=wandb_tags,
        # Opt-in via the SLURM launcher: hard-exit a completed run before the
        # habitat_sim GL teardown can SIGABRT and poison the exit code.
        hard_exit_on_finish=os.environ.get("R2DREAMER_HARD_EXIT_ON_FINISH") == "1",
    )


@dataclass(frozen=True)
class _ComposedRun:
    """What a run needs before its loop starts, whatever the loop then does.

    Attributes:
        adapter_cls: Resolved adapter class, kept so the caller can build a
            second collector (validation) on the same variant.
        collector: Env + adapter, plus a replay buffer when the run records.
        fields: The first frame's routed fields - the architecture description
            the agent's encoder was composed from.
        agent: The agent, freshly initialised or restored from a checkpoint.
        agent_config: The config the agent was built with.
    """

    adapter_cls: type
    collector: ExperienceCollector
    fields: AdapterOutput
    agent: R2DreamerAgent
    agent_config: R2DreamerConfig


def _compose_run(
    *,
    env: str,
    adapter: str,
    args: Any,
    curriculum: str | None,
    mode: str,
    output_dir: str,
    checkpoint: str | None = None,
) -> _ComposedRun:
    """Compose env, adapter, collector and agent for one run.

    Training and evaluation do not build different agents - they differ only in
    where the parameters come from. The architecture is discovered the same way
    in both: one adapter call on the first frame, whose routed fields compose
    the encoder. Only the last step branches on ``checkpoint``, so any change to
    the routing plumbing reaches both entry points at once.

    Args:
        env: Environment name.
        adapter: Variant name, a key of ``src.adapters.ADAPTERS``.
        args: Parsed CLI namespace of the calling entry point.
        curriculum: Effective habitat curriculum level.
        mode: Curriculum split the env rolls out in.
        output_dir: Run directory. Handed to a variant that claims it on both
            paths, so an adapter always sees the directory the run writes to;
            additionally recorded as the fresh config's ``logdir`` when there is
            no checkpoint, since a checkpoint's config comes from its manifest.
        checkpoint: Parameters to load. ``None`` initialises them fresh from the
            CLI config and gives the collector a replay buffer, because a run
            that owns its parameters is the run that trains them.

    Returns:
        The composed pieces.
    """
    adapter_cls = resolve_adapter(adapter)
    env_instance = _fresh_env(
        env=env, adapter_cls=adapter_cls, args=args, curriculum=curriculum, mode=mode
    )
    try:
        num_actions = env_instance.num_actions
        agent_config = None
        buffer = None
        if checkpoint is None:
            agent_config = _agent_config(
                args=args,
                adapter=adapter,
                num_actions=num_actions,
                output_dir=output_dir,
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
            episode_metrics_fn=(
                # Only a recording run needs the rolling trackers: the eval loop
                # reads success/SPL off the final frame itself.
                HabitatEpisodeMetrics(env_instance)
                if env == "habitat" and buffer is not None
                else None
            ),
        )

        # One adapter call on the first frame supplies the routing and field
        # shapes the agent needs at init. The loop resets again when it starts;
        # that second reset costs one episode of the iterator and keeps the
        # collector the only owner of the rollout.
        first_fields = collector.reset_fields()
        encoder_overrides = dict(adapter_cls.ENCODER_OVERRIDES)
        if checkpoint is not None:
            agent = R2DreamerAgent.from_checkpoint(
                checkpoint,
                num_actions=num_actions,
                seed=args.seed,
                fields=first_fields,
                encoder_overrides=encoder_overrides,
                adapter=adapter,
                **arch_overrides_from_manifest(checkpoint),
            )
        else:
            assert agent_config is not None  # set above whenever checkpoint is None
            _rng_key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
            agent = R2DreamerAgent(
                agent_config,
                init_key,
                fields=first_fields,
                encoder_overrides=encoder_overrides,
            )
    except BaseException:
        # A half-composed run must not leak the env: an abandoned habitat sim
        # keeps its GL context, and the next env build then fails for a reason
        # that has nothing to do with the actual error.
        env_instance.close()
        raise

    return _ComposedRun(
        adapter_cls=adapter_cls,
        collector=collector,
        fields=first_fields,
        agent=agent,
        agent_config=agent.cfg,
    )


def train(
    *,
    env: str,
    adapter: str,
    curriculum: str | None = None,
    output_dir: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: list[str] | None = None,
    argv: list[str] | None = None,
) -> TrainingRun:
    """Compose env, adapter, buffer, agent and collectors, then run training.

    Kwargs (output_dir, wandb_name, wandb_tags) are shim-supplied defaults - CLI
    flags from argparse override if provided.

    Args:
        env: Environment name.
        adapter: Variant name, a key of ``src.adapters.ADAPTERS``.
        curriculum: Habitat curriculum level.
        output_dir: Run directory for checkpoints, metrics and the manifest.
        wandb_name: W&B run name.
        wandb_tags: W&B tags.
        argv: Train flags; defaults to ``sys.argv[1:]``.

    Returns:
        A ``TrainingRun`` handle (agent + collectors + effective configs) for
        programmatic (notebook) callers.
    """
    args = _build_parser_train().parse_args(
        argv if argv is not None else sys.argv[1:]
    )
    eff_curriculum = _effective_curriculum(env=env, args=args, curriculum=curriculum)
    eff_output_dir, eff_wandb_name, eff_wandb_tags = _effective_run_metadata(
        args=args,
        output_dir=output_dir,
        wandb_name=wandb_name,
        wandb_tags=wandb_tags,
    )
    composed = _compose_run(
        env=env,
        adapter=adapter,
        args=args,
        curriculum=eff_curriculum,
        mode=args.mode,
        output_dir=eff_output_dir,
    )
    agent, collector = composed.agent, composed.collector
    print(f"adapter {adapter!r} -> embed_size {agent.embed_size}")

    val_collector = None
    if env == "habitat" and args.val_every > 0 and args.mode == "train":
        # Own env, adapter and tracker so val rollouts never disturb the train
        # VGGT cache or the train rolling means. No buffer (val never records)
        # and no auto-reset (an extra reset would advance the pinned eval order).
        val_env = _fresh_env(
            env=env,
            adapter_cls=composed.adapter_cls,
            args=args,
            curriculum=eff_curriculum,
            mode="eval",
        )
        val_collector = ExperienceCollector(
            env=val_env,
            observe=make_adapter(
                composed.adapter_cls, _without_variant_flags(args), output_dir=None
            ),
            num_actions=collector.num_actions,
            buffer=None,
            episode_metrics_fn=HabitatEpisodeMetrics(val_env, track_collision_rate=True),
            auto_reset=False,
        )

    trainer_config = _trainer_config(
        args=args,
        output_dir=eff_output_dir,
        wandb_name=eff_wandb_name,
        wandb_tags=eff_wandb_tags,
    )
    resume_step = 0
    if trainer_config.resume_from is not None:
        resume_step = apply_resume(agent, trainer_config.resume_from)

    run_training(
        agent,
        collector,
        composed.agent_config,
        trainer_config,
        val_experience=val_collector,
        resume_step=resume_step,
    )
    return TrainingRun(
        agent=agent,
        experience=collector,
        val_experience=val_collector,
        agent_config=composed.agent_config,
        trainer_config=trainer_config,
    )


def evaluate(
    *,
    env: str,
    adapter: str,
    curriculum: str | None = None,
    checkpoint: str | None = None,
    output_dir: str | None = None,
    argv: list[str] | None = None,
) -> dict:
    """Load a checkpoint and run inference episodes with it.

    The architecture is not read from the checkpoint: the adapter is called once
    on the first frame, the encoder is rebuilt from that routing, and only the
    parameters come from disk (guarded by a param-tree comparison).

    Args:
        env: Environment name.
        adapter: Variant name, must match the one the checkpoint was trained with.
        curriculum: Habitat curriculum level.
        checkpoint: Checkpoint path; required unless ``--random``.
        output_dir: Directory for ``eval_results.json`` and artifacts.
        argv: Eval flags; defaults to ``sys.argv[1:]``.

    Returns:
        Metrics dict with ``results`` and ``meta`` keys.
    """
    args = _build_parser_eval().parse_args(argv if argv is not None else sys.argv[1:])
    eff_checkpoint, eff_output_dir = resolve_eval_settings(
        args, checkpoint=checkpoint, output_dir=output_dir
    )
    eff_curriculum = args.curriculum if args.curriculum is not None else curriculum
    env_instance: Env | None = None
    try:
        agent: Any
        observe: AdapterFn
        if args.random:
            # The baseline has no encoder, so there are no routed fields to
            # learn and no first-frame reset: one less env.reset() than the
            # checkpoint path, which is what pins the episodes it is scored on.
            from src.baselines.random_agent import RandomAgent

            adapter_cls = resolve_adapter(adapter)
            env_instance = _fresh_env(
                env=env,
                adapter_cls=adapter_cls,
                args=args,
                curriculum=eff_curriculum,
                mode="eval",
            )
            observe = make_adapter(adapter_cls, args, output_dir=eff_output_dir)
            print("Using random agent")
            agent = RandomAgent(
                env=env_instance, num_actions=env_instance.num_actions, seed=args.seed
            )
        else:
            assert eff_checkpoint is not None  # resolve_eval_settings guarantees it
            composed = _compose_run(
                env=env,
                adapter=adapter,
                args=args,
                curriculum=eff_curriculum,
                mode="eval",
                output_dir=eff_output_dir,
                checkpoint=eff_checkpoint,
            )
            env_instance = composed.collector.env
            observe = composed.collector.observe
            agent = composed.agent
            print(f"Loaded checkpoint from step {agent.checkpoint_step}")
        return run_evaluation(
            args=args,
            env_instance=env_instance,
            observe=observe,
            agent=agent,
            checkpoint=eff_checkpoint,
            output_dir=eff_output_dir,
        )
    finally:
        # _compose_run closes the env it built if composition itself fails, so
        # env_instance is only set once there is something to close.
        if env_instance is not None:
            env_instance.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 3D ObjectNav pipeline from a single entry point."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("train", "Train R2Dreamer end to end."),
        ("evaluate", "Evaluate a trained agent."),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--env", default="habitat", choices=list(ENVS))
        # Default is the appearance-only control baseline.
        sub.add_argument("--adapter", default="rgb", choices=sorted(ADAPTERS))
        # No default: the level must stay unset for envs that reject one, and
        # _effective_curriculum is what enforces habitat's requirement.
        sub.add_argument("--curriculum", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> object:
    """Dispatch to train/evaluate, forwarding workflow-specific flags."""
    parser = _build_parser()
    args, rest = parser.parse_known_args(list(argv) if argv is not None else None)
    if args.command == "train":
        return train(
            env=args.env,
            adapter=args.adapter,
            curriculum=args.curriculum,
            argv=rest,
        )
    if args.command == "evaluate":
        return evaluate(
            env=args.env,
            adapter=args.adapter,
            curriculum=args.curriculum,
            argv=rest,
        )
    parser.error(f"Unknown command: {args.command}")
    return None


if __name__ == "__main__":
    main(sys.argv[1:])
