"""Public train() entry point for the r2dreamer launcher."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.r2dreamer.trainer import Trainer


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

    from modules.r2dreamer.agent import R2DreamerAgent
    from modules.r2dreamer.config import R2DreamerConfig
    from modules.r2dreamer.launch.curricula import CURRICULA
    from modules.r2dreamer.launch.parser import _build_parser_train
    from modules.r2dreamer.launch.registries import env_registry, encoder_registry
    from modules.r2dreamer.trainer import Trainer, TrainerConfig, habitat_defaults

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
    if env == "habitat":
        env_instance = env_fn(
            curriculum_path=curriculum_path,
            curriculum_mode=args.curriculum_mode,
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
        val_data=args.val_data,
        val_loss_every=args.val_loss_every,
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
        trainer = Trainer(
            agent=agent,
            env=env_instance,
            agent_config=agent_config,
            trainer_config=trainer_config,
            obs_adapter=adapter,
            episode_metrics_fn=hab["episode_metrics_fn"],
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
