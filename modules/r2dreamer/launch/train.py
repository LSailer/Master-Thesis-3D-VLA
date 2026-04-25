"""Public train() entry point for the r2dreamer launcher."""

from __future__ import annotations

import os
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

    from modules.r2dreamer.launch.parser import _build_parser_train
    from modules.r2dreamer.launch.registries import env_registry, encoder_registry
    from modules.r2dreamer.launch.curricula import CURRICULA
    from modules.r2dreamer.agent import R2DreamerAgent
    from modules.r2dreamer.config import R2DreamerConfig
    from modules.r2dreamer.adapters import VGGT_FEATURE_DIM
    from modules.r2dreamer.trainer import Trainer, TrainerConfig, habitat_defaults

    parser = _build_parser_train()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # --- Resolve env ---
    if env not in env_registry:
        raise KeyError(f"Unknown env {env!r}. Available: {list(env_registry)}")

    # --- Resolve curriculum path ---
    # CLI --curriculum_path is the escape hatch; otherwise use registry lookup.
    if args.curriculum_path is not None:
        # explicit CLI override
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

    # --- Resolve encoder ---
    if encoder not in encoder_registry:
        raise KeyError(f"Unknown encoder {encoder!r}. Available: {list(encoder_registry)}")

    encoder_cls = encoder_registry[encoder]
    if encoder == "vggt":
        enc = encoder_cls(resolution=args.render_resolution)
    else:
        enc = encoder_cls()
    adapter = enc.make_adapter()

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
            render_resolution=args.render_resolution if encoder == "vggt" else 64,
        )
    else:
        # crafter
        env_instance = env_fn(seed=args.seed)

    # --- Build agent config ---
    if encoder == "vggt":
        agent_config = R2DreamerConfig(
            encoder_type="vggt",
            obs_shape=(VGGT_FEATURE_DIM,),
            num_actions=4,
            total_steps=args.steps,
            prefill_steps=args.prefill,
            buffer_capacity=1_000_000,
            act_entropy=3e-2,
            seed=args.seed,
            log_every=args.log_every,
            logdir=eff_output_dir,
        )
    elif env == "habitat":
        agent_config = R2DreamerConfig(
            obs_shape=(3, 64, 64),
            num_actions=4,
            total_steps=args.steps,
            prefill_steps=args.prefill,
            buffer_capacity=1_000_000,
            act_entropy=3e-2,
            seed=args.seed,
            log_every=args.log_every,
            logdir=eff_output_dir,
        )
    else:
        # crafter
        agent_config = R2DreamerConfig(
            obs_shape=(3, 64, 64),
            num_actions=17,
            total_steps=args.steps,
            prefill_steps=args.prefill,
            seed=args.seed,
            log_every=args.log_every,
            logdir=eff_output_dir,
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
