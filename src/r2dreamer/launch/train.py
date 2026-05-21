"""Public train() entry point for the r2dreamer launcher."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.r2dreamer.trainer import Trainer


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
    from src.r2dreamer.config import R2DreamerConfig
    from src.r2dreamer.launch.curricula import CURRICULA
    from src.r2dreamer.launch.parser import _build_parser_train
    from src.r2dreamer.launch.registries import env_registry, encoder_registry
    from src.r2dreamer.trainer import Trainer, TrainerConfig, habitat_defaults

    parser = _build_parser_train()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # --- Resolve env ---
    if env not in env_registry:
        raise KeyError(f"Unknown env {env!r}. Available: {list(env_registry)}")

    # --- Resolve curriculum path ---
    # CLI --curriculum_path is the escape hatch; otherwise use registry lookup.
    if args.curriculum_path is not None:
        curriculum_path = args.curriculum_path
    elif curriculum is not None:
        if curriculum not in CURRICULA:
            raise KeyError(f"Unknown curriculum {curriculum!r}. Available: {list(CURRICULA)}")
        curriculum_path = str(CURRICULA[curriculum])
    else:
        curriculum_path = None

    if env == "habitat" and curriculum_path is None:
        raise ValueError(
            "Habitat env requires a curriculum. "
            "Pass curriculum='L1'..'L4' to train() or --curriculum_path on CLI."
        )
    if env == "crafter" and curriculum_path is not None:
        raise ValueError("Crafter env does not use a curriculum.")

    # --- Resolve encoder and its observation spec ---
    if encoder not in encoder_registry:
        raise KeyError(f"Unknown encoder {encoder!r}. Available: {list(encoder_registry)}")

    encoder_cls = encoder_registry[encoder]
    enc = encoder_cls.from_train_args(args)
    adapter = enc.make_adapter()
    encoder_spec = enc.spec()

    # --- Resolve effective output_dir / wandb_name / wandb_tags ---
    # CLI value (non-None) wins over shim kwarg.
    eff_output_dir = args.output_dir if args.output_dir is not None else output_dir
    if eff_output_dir is None:
        raise ValueError("output_dir must be set via train(..., output_dir=...) or --output_dir")

    eff_wandb_name = args.wandb_name if args.wandb_name is not None else wandb_name

    # Build tags: start with shim defaults, extend with CLI extras if provided.
    eff_wandb_tags: list[str] = list(wandb_tags) if wandb_tags is not None else []
    if args.wandb_tags:
        eff_wandb_tags.extend(t.strip() for t in args.wandb_tags.split(","))

    # --- Build env ---
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
        num_actions = 4
    else:
        env_instance = env_fn(seed=args.seed)
        num_actions = 17

    # --- Build agent config ---
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

    agent_config = R2DreamerConfig(
        encoder_type=encoder_spec.encoder_type,
        encoder_module_cls=encoder_spec.module_cls,
        obs_shape=encoder_spec.obs_shape,
        num_actions=num_actions,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        act_entropy=args.act_entropy,
        seed=args.seed,
        log_every=args.log_every,
        logdir=eff_output_dir,
        design_notes=encoder_spec.design_notes,
        **agent_overrides,
    )

    # --- Build agent ---
    _rng_key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    agent = R2DreamerAgent(agent_config, init_key)

    # --- Build trainer config ---
    trainer_config = TrainerConfig(
        output_dir=eff_output_dir,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_name=eff_wandb_name,
        wandb_tags=eff_wandb_tags,
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
    )

    # --- Build trainer ---
    if env == "habitat":
        hab = habitat_defaults(env_instance)
        val_kwargs: dict[str, object] = {}
        if val_env_instance is not None:
            # Val-Episode-Loop (3D-36) wiring: own adapter so the train
            # VGGT video buffer isn't disturbed; own tracker so val
            # rolling means stay independent of train rollouts.
            val_adapter = enc.make_adapter()
            val_hab = habitat_defaults(val_env_instance, track_collision_rate=True)
            val_kwargs = {
                "val_env": val_env_instance,
                "val_obs_adapter": val_adapter,
                "val_episode_metrics_fn": val_hab["episode_metrics_fn"],
            }
        trainer = Trainer(
            agent=agent,
            env=env_instance,
            agent_config=agent_config,
            trainer_config=trainer_config,
            obs_adapter=adapter,
            episode_metrics_fn=hab["episode_metrics_fn"],
            **val_kwargs,
        )
    else:
        trainer = Trainer(
            agent=agent,
            env=env_instance,
            agent_config=agent_config,
            trainer_config=trainer_config,
            obs_adapter=adapter,
        )

    trainer.run()
    return trainer
