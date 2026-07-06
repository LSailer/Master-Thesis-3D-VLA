"""Composition root for evaluate(): checkpoint/manifest resolution + build.

This module is allowed to know the concrete classes (``HabitatObjectNavEnv``,
``RandomAgent``, ``R2DreamerAgent``) because it is where evaluate()'s objects
are constructed. It resolves the checkpoint manifest into architecture
overrides, assembles the agent-config kwargs, and builds the env / encoder /
agent. The rollout in ``eval_loop.py`` consumes these objects through Protocols.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.baselines.random_agent import RandomAgent
from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.observation_preparation import recover_encoder_input_contract


def _find_manifest_for_checkpoint(checkpoint: str | Path) -> Path | None:
    """Locate a run's ``MANIFEST.json`` given a checkpoint path.

    Args:
      checkpoint: Path to a checkpoint file.

    Returns:
      The resolved manifest path (next to the checkpoint or one directory up),
      or ``None`` if neither exists.
    """
    ckpt = Path(checkpoint).resolve()
    for candidate in (
        ckpt.parent / "MANIFEST.json",
        ckpt.parent.parent / "MANIFEST.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _make_eval_env(*, args: Any, curriculum: str | None, eff_encoder: str):
    """Build the eval Habitat env, resolving render resolution from the encoder.

    Args:
      args: Parsed argparse namespace.
      curriculum: Shim-supplied curriculum default.
      eff_encoder: The resolved encoder name.

    Returns:
      A ``(env_instance, needs_hires, render_resolution)`` tuple.
    """
    # All VGGT readouts (wp_cp, aggregator, dense-WP CNN) AND the hybrid encoder
    # need 518x518 frames; the plain CNN baseline uses 64. Everything else is
    # driven off the EncoderSpec below.
    needs_hires = eff_encoder.startswith("vggt") or eff_encoder == "hybrid"
    default_resolution = 518 if needs_hires else 64
    render_resolution = (
        args.render_resolution
        if args.render_resolution is not None
        else default_resolution
    )
    effective_curriculum = (
        args.curriculum if args.curriculum is not None else curriculum
    )
    hab_config = HabitatEnvConfig(
        obs_shape=(3, render_resolution, render_resolution),
        max_episode_steps=500,
        split=args.split,
        reward_type="geodesic_delta",
        curriculum=effective_curriculum,
        curriculum_path=args.curriculum_path,
        curriculum_mode="eval",
    )
    env_instance = HabitatObjectNavEnv(
        hab_config,
        semantic=args.semantic,
        seed=args.seed,
    )
    return env_instance, needs_hires, render_resolution


def _make_eval_encoder(
    eff_encoder: str, encoder_registry: dict, needs_hires: bool, render_resolution: int
):
    """Instantiate the encoder and derive its adapter + spec.

    Args:
      eff_encoder: The resolved encoder name (a registry key).
      encoder_registry: The encoder-class registry.
      needs_hires: Whether the encoder needs the high-res resolution.
      render_resolution: The render resolution to pass to hi-res encoders.

    Returns:
      A ``(encoder, adapter, spec)`` tuple.
    """
    encoder_cls = encoder_registry[eff_encoder]
    enc = encoder_cls(resolution=render_resolution) if needs_hires else encoder_cls()
    return enc, enc.make_adapter(), enc.spec()


def _load_arch_overrides_from_manifest(eff_checkpoint: str | None) -> dict:
    """Recover architecture-shaping config overrides from a checkpoint manifest.

    Args:
      eff_checkpoint: The resolved checkpoint path, or ``None``.

    Returns:
      A dict of architecture fields from the manifest's saved config (plus a
      recovered ``encoder_input_contract`` when present); empty when no manifest
      or checkpoint is available.
    """
    if eff_checkpoint is None:
        return {}
    manifest = _find_manifest_for_checkpoint(eff_checkpoint)
    if manifest is None:
        return {}
    try:
        saved = json.loads(manifest.read_text()).get("config", {})
    except (ValueError, OSError):
        return {}

    arch_fields = (
        "deter_size",
        "hidden_size",
        "stoch_classes",
        "stoch_discrete",
        "blocks",
        "dyn_layers",
        "obs_layers",
        "img_layers",
        "encoder_depth",
        "encoder_kernel",
        "encoder_mults",
        "vggt_embed_dim",
        "vggt_mlp_layers",
        "mlp_vggt_hidden",
        "vggt_token_transformer_layers",
        "vggt_token_transformer_heads",
        "vggt_token_projection_dim",
        "vggt_token_transformer_mlp_ratio",
        "vggt_token_transformer_dropout",
        "vggt_keep_register_tokens",
        "vggt_token_count",
        "vggt_token_dim",
        "mlp_vggt_layers",
        "mlp_units",
        "mlp_layers_reward",
        "mlp_layers_cont",
        "mlp_layers_actor",
        "mlp_layers_critic",
        "twohot_bins",
        "decoder",
    )
    overrides = {
        key: tuple(saved[key]) if key == "encoder_mults" else saved[key]
        for key in arch_fields
        if key in saved
    }
    contract_snapshot = saved.get("encoder_input_contract")
    if contract_snapshot is not None:
        contract = recover_encoder_input_contract(contract_snapshot)
        overrides.update(
            encoder_type=contract.encoder_type,
            encoder_module_cls=contract.encoder_module_cls,
            obs_shape=contract.encoder_input.buffer_shape(),
            encoder_input_contract=contract_snapshot,
        )
    return overrides


def _agent_config_kwargs(
    encoder_spec: Any, *, args: Any, eff_checkpoint: str | None
) -> dict:
    """Assemble the R2DreamerConfig kwargs for the eval agent.

    Args:
      encoder_spec: The encoder spec (source of the base contract kwargs).
      args: Parsed argparse namespace.
      eff_checkpoint: The resolved checkpoint path, or ``None``.

    Returns:
      The config kwargs dict (base contract fields plus manifest overrides for
      non-random agents).

    Raises:
      ValueError: If the checkpoint's encoder contract disagrees with the
        CLI/registry-resolved encoder.
    """
    agent_config_kwargs: dict = {
        "encoder_type": encoder_spec.encoder_type,
        "encoder_module_cls": encoder_spec.module_cls,
        "obs_shape": encoder_spec.obs_shape,
    }
    if not args.random:
        overrides = _load_arch_overrides_from_manifest(eff_checkpoint)
        checkpoint_encoder = overrides.get("encoder_type")
        if (
            checkpoint_encoder is not None
            and checkpoint_encoder != encoder_spec.encoder_type
        ):
            raise ValueError(
                "checkpoint encoder contract mismatch: CLI/registry resolved "
                f"{encoder_spec.encoder_type!r}, checkpoint has "
                f"{checkpoint_encoder!r}"
            )
        agent_config_kwargs.update(overrides)
    return agent_config_kwargs


def _make_eval_agent(
    args: Any,
    eff_checkpoint: str | None,
    agent_config_kwargs: dict,
    env_instance: HabitatObjectNavEnv,
):
    """Build the eval agent: a RandomAgent or a checkpoint-restored R2Dreamer.

    Args:
      args: Parsed argparse namespace.
      eff_checkpoint: The resolved checkpoint path, or ``None``.
      agent_config_kwargs: Config kwargs for the learned agent.
      env_instance: The eval environment (needed by ``RandomAgent``).

    Returns:
      The constructed agent.

    Raises:
      ValueError: If not ``--random`` and no checkpoint is available.
    """
    if args.random:
        print("Using random agent")
        return RandomAgent(env=env_instance, num_actions=4, seed=args.seed)
    if eff_checkpoint is None:
        raise ValueError("checkpoint is required unless --random is set")
    agent = R2DreamerAgent.from_checkpoint(
        eff_checkpoint,
        num_actions=4,
        seed=args.seed,
        **agent_config_kwargs,
    )
    print(f"Loaded checkpoint from step {agent.checkpoint_step}")
    return agent
