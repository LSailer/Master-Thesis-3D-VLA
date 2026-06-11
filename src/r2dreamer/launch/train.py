"""Public train() entry point for the r2dreamer launcher."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.r2dreamer.trainer import Trainer


def _resolve_curriculum_inputs(
    *, env: str, args: Any, curriculum: str | None, env_registry: dict[str, Any],
) -> str | None:
    from src.r2dreamer.launch._helpers import resolve_curriculum_path

    if env not in env_registry:
        raise KeyError(f"Unknown env {env!r}. Available: {list(env_registry)}")

    curriculum_path = resolve_curriculum_path(args.curriculum_path, curriculum)
    if env == "habitat" and curriculum_path is None:
        raise ValueError(
            "Habitat env requires a curriculum. "
            "Pass curriculum='L1'..'L4' to train() or --curriculum_path on CLI."
        )
    if env == "crafter" and curriculum_path is not None:
        raise ValueError("Crafter env does not use a curriculum.")
    return curriculum_path


def _make_encoder_bundle(encoder: str, args: Any, encoder_registry: dict[str, Any]):
    if encoder not in encoder_registry:
        raise KeyError(f"Unknown encoder {encoder!r}. Available: {list(encoder_registry)}")
    enc = encoder_registry[encoder].from_train_args(args)
    return enc, enc.make_adapter(), enc.spec()


def _effective_run_metadata(
    *,
    args: Any,
    output_dir: str | None,
    wandb_name: str | None,
    wandb_tags: list[str] | None,
) -> tuple[str, str | None, list[str]]:
    # CLI value (non-None) wins over shim kwarg.
    eff_output_dir = args.output_dir if args.output_dir is not None else output_dir
    if eff_output_dir is None:
        raise ValueError("output_dir must be set via train(..., output_dir=...) or --output_dir")

    eff_wandb_name = args.wandb_name if args.wandb_name is not None else wandb_name
    eff_wandb_tags: list[str] = list(wandb_tags) if wandb_tags is not None else []
    if args.wandb_tags:
        eff_wandb_tags.extend(t.strip() for t in args.wandb_tags.split(","))
    return eff_output_dir, eff_wandb_name, eff_wandb_tags


def _make_env_instances(
    *,
    env: str,
    args: Any,
    curriculum_path: str | None,
    encoder_spec: Any,
    env_registry: dict[str, Any],
) -> tuple[Any, Any | None, int]:
    env_fn = env_registry[env]
    val_env_instance = None
    if env == "habitat":
        env_instance = env_fn(
            curriculum_path=curriculum_path,
            curriculum_mode=args.curriculum_mode,
            seed=args.seed,
            render_resolution=encoder_spec.env_render_resolution,
        )
        # Val env: same curriculum JSON, eval-key set. Skip when val is off
        # or the user is in train-of-train mode (curriculum_mode != "train").
        if args.val_every > 0 and args.curriculum_mode == "train":
            val_env_instance = env_fn(
                curriculum_path=curriculum_path,
                curriculum_mode="eval",
                seed=args.seed,
                render_resolution=encoder_spec.env_render_resolution,
            )
        return env_instance, val_env_instance, 4

    return env_fn(seed=args.seed), None, 17


def _agent_overrides_from_args(args: Any, encoder_spec: Any, latent_presets: dict[str, dict]):
    agent_overrides = dict(encoder_spec.agent_overrides)
    # Diagnostic CLI overrides (None => keep config default / encoder override).
    if args.actor_loss_weight is not None:
        agent_overrides["scale_policy"] = args.actor_loss_weight
    if args.value_loss_weight is not None:
        agent_overrides["scale_value"] = args.value_loss_weight
    if args.repval_loss_weight is not None:
        agent_overrides["scale_repval"] = args.repval_loss_weight
    if args.barlow_grad_to_encoder:
        agent_overrides["barlow_stop_grad"] = False
    if args.batch_size is not None:
        agent_overrides["batch_size"] = args.batch_size
    if args.seq_len is not None:
        agent_overrides["seq_len"] = args.seq_len
    if args.lr is not None:
        agent_overrides["lr"] = args.lr
    if args.mlp_layers is not None:
        agent_overrides["vggt_mlp_layers"] = args.mlp_layers
    if getattr(args, "train_ratio", None) is not None:
        agent_overrides["train_ratio"] = args.train_ratio
    if getattr(args, "buffer_capacity", None) is not None:
        agent_overrides["buffer_capacity"] = args.buffer_capacity

    # Latent-size ablation (3D-50): preset from the LATENT_PRESETS table, then
    # explicit flags win.
    preset = getattr(args, "latent_preset", "default")
    agent_overrides.update(latent_presets.get(preset, {}))
    for name in ("deter_size", "stoch_classes", "stoch_discrete", "mlp_vggt_hidden", "mlp_vggt_layers"):
        value = getattr(args, name, None)
        if value is not None:
            agent_overrides[name] = value
    if getattr(args, "decoder", False):
        agent_overrides["decoder"] = True
    if getattr(args, "scale_decoder", None) is not None:
        agent_overrides["scale_decoder"] = args.scale_decoder
    return agent_overrides


def _make_agent_config(
    *, args: Any, encoder_spec: Any, num_actions: int, output_dir: str,
    agent_overrides: dict[str, Any], config_cls: type,
):
    return config_cls(
        encoder_type=encoder_spec.encoder_type,
        encoder_module_cls=encoder_spec.module_cls,
        obs_shape=encoder_spec.obs_shape,
        num_actions=num_actions,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        act_entropy=args.act_entropy,
        seed=args.seed,
        log_every=args.log_every,
        logdir=output_dir,
        design_notes=encoder_spec.design_notes,
        **agent_overrides,
    )


def _make_trainer_config(
    *, args: Any, output_dir: str, wandb_name: str | None,
    wandb_tags: list[str], trainer_config_cls: type,
):
    return trainer_config_cls(
        output_dir=output_dir,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_name=wandb_name,
        wandb_tags=wandb_tags,
        wandb_id=args.wandb_id,
        video_log_every=args.video_log_every,
        video_log_episodes=args.video_log_episodes,
        val_every=args.val_every,
        val_episodes=args.val_episodes,
        val_video_episodes=args.val_video_episodes,
        val_max_episode_steps=args.val_max_episode_steps,
        resume_from=args.resume_from,
        overfit_one_batch=args.overfit_one_batch,
        overfit_steps=args.overfit_steps,
        overfit_batch_size=args.overfit_batch_size,
        overfit_seq_len=args.overfit_seq_len,
        overfit_min_loss_drop=args.overfit_min_loss_drop,
        # Opt-in via the SLURM launcher: hard-exit a completed run before the
        # habitat_sim GL teardown can SIGABRT and poison the exit code.
        hard_exit_on_finish=os.environ.get("R2DREAMER_HARD_EXIT_ON_FINISH") == "1",
    )


def _make_trainer(
    *,
    env: str,
    enc: Any,
    agent: Any,
    env_instance: Any,
    val_env_instance: Any | None,
    agent_config: Any,
    trainer_config: Any,
    adapter: Any,
    trainer_cls: type,
    habitat_defaults_fn: Any,
):
    if env != "habitat":
        return trainer_cls(
            agent=agent,
            env=env_instance,
            agent_config=agent_config,
            trainer_config=trainer_config,
            obs_adapter=adapter,
        )

    hab = habitat_defaults_fn(env_instance)
    val_kwargs: dict[str, object] = {}
    if val_env_instance is not None:
        # Val-Episode-Loop (3D-36) wiring: own adapter so the train
        # VGGT video buffer isn't disturbed; own tracker so val
        # rolling means stay independent of train rollouts.
        val_adapter = enc.make_adapter()
        val_hab = habitat_defaults_fn(val_env_instance, track_collision_rate=True)
        val_kwargs = {
            "val_env": val_env_instance,
            "val_obs_adapter": val_adapter,
            "val_episode_metrics_fn": val_hab["episode_metrics_fn"],
        }
    return trainer_cls(
        agent=agent,
        env=env_instance,
        agent_config=agent_config,
        trainer_config=trainer_config,
        obs_adapter=adapter,
        episode_metrics_fn=hab["episode_metrics_fn"],
        **val_kwargs,
    )


def train(
    *,
    env: str,
    encoder: str,
    curriculum: str | None = None,
    output_dir: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: list[str] | None = None,
    argv: list[str] | None = None,
) -> "Trainer":
    """Resolve (env, encoder, curriculum) via registries; parse CLI; run Trainer.run().

    Kwargs (output_dir, wandb_name, wandb_tags) are shim-supplied defaults — CLI
    flags from argparse override if provided.

    Returns the Trainer for programmatic (notebook) callers.
    """
    import jax

    from src.r2dreamer.agent import R2DreamerAgent
    from src.r2dreamer.config import R2DreamerConfig, LATENT_PRESETS
    from src.r2dreamer.launch.parser import _build_parser_train
    from src.r2dreamer.launch.registries import env_registry, encoder_registry
    from src.r2dreamer.trainer import Trainer, TrainerConfig, habitat_defaults

    parser = _build_parser_train()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    curriculum_path = _resolve_curriculum_inputs(
        env=env, args=args, curriculum=curriculum, env_registry=env_registry,
    )
    enc, adapter, encoder_spec = _make_encoder_bundle(encoder, args, encoder_registry)
    eff_output_dir, eff_wandb_name, eff_wandb_tags = _effective_run_metadata(
        args=args, output_dir=output_dir, wandb_name=wandb_name, wandb_tags=wandb_tags,
    )
    env_instance, val_env_instance, num_actions = _make_env_instances(
        env=env,
        args=args,
        curriculum_path=curriculum_path,
        encoder_spec=encoder_spec,
        env_registry=env_registry,
    )
    agent_overrides = _agent_overrides_from_args(args, encoder_spec, LATENT_PRESETS)
    agent_config = _make_agent_config(
        args=args,
        encoder_spec=encoder_spec,
        num_actions=num_actions,
        output_dir=eff_output_dir,
        agent_overrides=agent_overrides,
        config_cls=R2DreamerConfig,
    )

    # --- Build agent ---
    _rng_key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    agent = R2DreamerAgent(agent_config, init_key)

    trainer_config = _make_trainer_config(
        args=args,
        output_dir=eff_output_dir,
        wandb_name=eff_wandb_name,
        wandb_tags=eff_wandb_tags,
        trainer_config_cls=TrainerConfig,
    )
    trainer = _make_trainer(
        env=env,
        enc=enc,
        agent=agent,
        env_instance=env_instance,
        val_env_instance=val_env_instance,
        agent_config=agent_config,
        trainer_config=trainer_config,
        adapter=adapter,
        trainer_cls=Trainer,
        habitat_defaults_fn=habitat_defaults,
    )

    trainer.run()
    return trainer
