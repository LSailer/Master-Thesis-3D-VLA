"""Public evaluate() entry point for the r2dreamer launcher."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jax
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from src.baselines.random_agent import RandomAgent
from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv
from src.environments.observation import ObservationFrame
from src.r2dreamer.launch.parser import _build_parser_eval
from src.r2dreamer.launch.registries import env_registry
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.obs_batch import ObservationPacker
from src.r2dreamer.observation_preparation import recover_encoder_input_contract
from src.shared.video_utils import (
    compose_frame,
    log_episode_video,
    render_topdown_frame,
)

_ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def _obs_value(obs, name: str):
    return obs[name] if isinstance(obs, dict) else getattr(obs, name)


def _extract_goal_positions(env):
    goal_positions = []
    for goal in env.current_episode.goals:
        if goal.view_points:
            for vp in goal.view_points:
                pos = vp.agent_state.position
                goal_positions.append(
                    pos.tolist() if hasattr(pos, "tolist") else list(pos)
                )
                break
        else:
            pos = goal.position
            goal_positions.append(pos.tolist() if hasattr(pos, "tolist") else list(pos))
    return goal_positions


def _get_agent_heading(env):
    """Extract agent heading (yaw in radians) from habitat sim state."""
    state = env._env.sim.get_agent_state()
    quat = state.rotation
    r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    euler = r.as_euler("yxz")
    return float(euler[0])


def _find_manifest_for_checkpoint(checkpoint: str | Path) -> Path | None:
    ckpt = Path(checkpoint).resolve()
    for candidate in (
        ckpt.parent / "MANIFEST.json",
        ckpt.parent.parent / "MANIFEST.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _resolve_eval_settings(
    args, *, encoder: str, checkpoint: str | None, output_dir: str | None
):
    # CLI --encoder overrides shim kwarg if user passed it explicitly.
    eff_encoder = args.encoder if args.encoder is not None else encoder

    eff_checkpoint = args.checkpoint if args.checkpoint is not None else checkpoint
    if not args.random and eff_checkpoint is None:
        raise ValueError(
            "checkpoint must be set via evaluate(..., checkpoint=...) or --checkpoint"
        )

    eff_output_dir = args.output_dir if args.output_dir is not None else output_dir
    if eff_output_dir is None:
        raise ValueError(
            "output_dir must be set via evaluate(..., output_dir=...) or --output_dir"
        )
    return eff_encoder, eff_checkpoint, eff_output_dir


def _init_eval_wandb(args):
    if args.wandb_project is None or args.log_video_episodes <= 0:
        return None
    import wandb

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    return wandb


def _make_eval_env(*, args, curriculum: str | None, eff_encoder: str):
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
    encoder_cls = encoder_registry[eff_encoder]
    enc = encoder_cls(resolution=render_resolution) if needs_hires else encoder_cls()
    return enc, enc.make_adapter(), enc.spec()


def _load_arch_overrides_from_manifest(eff_checkpoint: str | None) -> dict:
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


def _agent_config_kwargs(encoder_spec, *, args, eff_checkpoint: str | None) -> dict:
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
    args,
    eff_checkpoint: str | None,
    agent_config_kwargs: dict,
    env_instance: HabitatObjectNavEnv,
):
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


def _start_eval_episode(env_instance, adapter, packer: ObservationPacker):
    obs = env_instance.reset()
    if adapter.on_episode_reset:
        adapter.on_episode_reset()
    prepared = adapter.prepare_env_step(obs, packer)
    encoder_obs = prepared.encoder_obs
    is_first = prepared.is_first

    start_pos = env_instance._env.sim.get_agent_state().position.tolist()
    goal_positions = _extract_goal_positions(env_instance)
    scene_id = env_instance.current_episode.scene_id
    object_category = env_instance.current_episode.object_category
    trajectory = [start_pos]
    headings = [_get_agent_heading(env_instance)]
    return (
        obs,
        encoder_obs,
        is_first,
        start_pos,
        goal_positions,
        scene_id,
        object_category,
        trajectory,
        headings,
    )


def _initial_eval_video_frames(
    env_instance, obs: ObservationFrame, trajectory, goal_positions, record_video: bool
):
    if not record_video:
        return []
    topdown = render_topdown_frame(env_instance, trajectory, goal_positions)
    return [compose_frame(_obs_value(obs, "image"), topdown)]


def _make_eval_episode_result(
    *,
    ep_idx: int,
    scene_id: str,
    object_category: str,
    actions_taken: list[int],
    rewards: list[float],
    obs: ObservationFrame,
    start_pos: list[float],
    goal_positions: list[list[float]],
    trajectory: list[list[float]],
    headings: list[float],
) -> dict:
    return {
        "episode": ep_idx,
        "scene_id": scene_id,
        "object_category": object_category,
        "steps": len(actions_taken),
        "reward": sum(rewards),
        "success": float(_obs_value(obs, "success")),
        "spl": float(_obs_value(obs, "spl")),
        "actions": actions_taken,
        "action_counts": {
            name: actions_taken.count(idx) for idx, name in _ACTIONS.items()
        },
        "start_position": start_pos,
        "goal_positions": goal_positions,
        "trajectory": trajectory,
        "headings": headings,
    }


def _write_eval_episode_artifacts(
    *,
    args,
    env_instance,
    output_dir: str,
    ep_idx: int,
    trajectory: list[list[float]],
    goal_positions: list[list[float]],
    record_video: bool,
    wandb_module,
    video_frames: list[np.ndarray],
) -> None:
    if args.render_topdown:
        topdown_dir = os.path.join(output_dir, "topdown")
        os.makedirs(topdown_dir, exist_ok=True)
        topdown_path = os.path.join(topdown_dir, f"episode_{ep_idx:03d}.png")
        Image.fromarray(
            render_topdown_frame(env_instance, trajectory, goal_positions)
        ).save(topdown_path)
    if record_video:
        log_episode_video(
            wandb_module, f"eval/episode_video_{ep_idx}", video_frames, ep_idx
        )


def _run_eval_episode(
    *,
    ep_idx: int,
    args,
    env_instance,
    adapter,
    agent,
    rng_key,
    config,
    wandb_module,
    output_dir: str,
) -> tuple[dict, jax.Array]:
    packer = ObservationPacker(config)
    (
        obs,
        encoder_obs,
        is_first,
        start_pos,
        goal_positions,
        scene_id,
        object_category,
        trajectory,
        headings,
    ) = _start_eval_episode(env_instance, adapter, packer)
    actions_taken = []
    rewards = []
    record_video = wandb_module is not None and ep_idx < args.log_video_episodes
    video_frames = _initial_eval_video_frames(
        env_instance,
        obs,
        trajectory,
        goal_positions,
        record_video,
    )
    act_state = None if isinstance(agent, RandomAgent) else agent.initial_act_state()

    for _step in range(500):
        if isinstance(agent, RandomAgent):
            action = agent.sample_action()
            next_obs = agent.act(action)
        else:
            rng_key, act_key = jax.random.split(rng_key)
            action, act_state = agent.act_with_state(
                encoder_obs, is_first, act_state, act_key, training=False
            )
            next_obs = env_instance.step(action)
        next_prepared = adapter.prepare_env_step(next_obs, packer)
        next_encoder_obs = next_prepared.encoder_obs
        next_is_first = next_prepared.is_first
        actions_taken.append(int(action))
        rewards.append(float(_obs_value(next_obs, "reward")))

        pos = env_instance._env.sim.get_agent_state().position.tolist()
        trajectory.append(pos)
        headings.append(_get_agent_heading(env_instance))
        if record_video:
            topdown = render_topdown_frame(env_instance, trajectory, goal_positions)
            video_frames.append(compose_frame(_obs_value(next_obs, "image"), topdown))

        if _obs_value(next_obs, "done"):
            obs = next_obs
            break
        obs = next_obs
        encoder_obs = next_encoder_obs
        is_first = next_is_first

    ep_result = _make_eval_episode_result(
        ep_idx=ep_idx,
        scene_id=scene_id,
        object_category=object_category,
        actions_taken=actions_taken,
        rewards=rewards,
        obs=obs,
        start_pos=start_pos,
        goal_positions=goal_positions,
        trajectory=trajectory,
        headings=headings,
    )
    _write_eval_episode_artifacts(
        args=args,
        env_instance=env_instance,
        output_dir=output_dir,
        ep_idx=ep_idx,
        trajectory=trajectory,
        goal_positions=goal_positions,
        record_video=record_video,
        wandb_module=wandb_module,
        video_frames=video_frames,
    )
    print(
        f"Episode {ep_idx}: steps={len(actions_taken):3d}  "
        f"reward={sum(rewards):.2f}  "
        f"success={float(_obs_value(obs, 'success')):.0f}  "
        f"category={object_category}"
    )
    return ep_result, rng_key


def _print_eval_summary(results: list[dict], episodes: int) -> None:
    print(f"\n--- Summary ({episodes} episodes) ---")
    print(f"Success: {np.mean([r['success'] for r in results]) * 100:.1f}%")
    print(f"SPL: {np.mean([r['spl'] for r in results]):.3f}")
    print(f"Mean reward: {np.mean([r['reward'] for r in results]):.2f}")
    print(f"Mean steps: {np.mean([r['steps'] for r in results]):.0f}")


def evaluate(
    *,
    env: str,
    encoder: str,
    curriculum: str | None = None,
    checkpoint: str | None = None,
    output_dir: str | None = None,
    argv: list[str] | None = None,
) -> dict:
    """Resolve (env, encoder, curriculum) via registries; parse CLI; run eval loop.

    Kwargs (checkpoint, output_dir) are shim-supplied defaults — CLI flags override.

    Returns metrics dict with 'results' and 'meta' keys.
    """
    from src.r2dreamer.launch.registries import encoder_registry

    parser = _build_parser_eval()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    eff_encoder, eff_checkpoint, eff_output_dir = _resolve_eval_settings(
        args,
        encoder=encoder,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )
    if eff_encoder not in encoder_registry:
        raise KeyError(
            f"Unknown encoder {eff_encoder!r}. Available: {list(encoder_registry)}"
        )

    os.makedirs(eff_output_dir, exist_ok=True)
    output_path = os.path.join(eff_output_dir, "eval_results.json")

    wandb_module = None
    env_instance = None
    try:
        wandb_module = _init_eval_wandb(args)

        # --- Build env ---
        if env not in env_registry:
            raise KeyError(f"Unknown env {env!r}. Available: {list(env_registry)}")
        env_instance, needs_hires, render_resolution = _make_eval_env(
            args=args,
            curriculum=curriculum,
            eff_encoder=eff_encoder,
        )

        # --- Build encoder + adapter ---
        _enc, adapter, encoder_spec = _make_eval_encoder(
            eff_encoder,
            encoder_registry,
            needs_hires,
            render_resolution,
        )

        agent_config_kwargs = _agent_config_kwargs(
            encoder_spec,
            args=args,
            eff_checkpoint=eff_checkpoint,
        )

        config = R2DreamerConfig(num_actions=4, **agent_config_kwargs)
        rng_key = jax.random.PRNGKey(args.seed)

        agent = _make_eval_agent(
            args, eff_checkpoint, agent_config_kwargs, env_instance
        )
        if not isinstance(agent, RandomAgent):
            # Match the rng_key split the inline init_key used to consume so the
            # downstream act_key chain stays identical.
            rng_key, _ = jax.random.split(rng_key)

        # --- Evaluate ---
        results = []
        for ep_idx in range(args.episodes):
            ep_result, rng_key = _run_eval_episode(
                ep_idx=ep_idx,
                args=args,
                env_instance=env_instance,
                adapter=adapter,
                agent=agent,
                rng_key=rng_key,
                config=config,
                wandb_module=wandb_module,
                output_dir=eff_output_dir,
            )
            results.append(ep_result)

        _print_eval_summary(results, args.episodes)

        meta = {"agent": "random" if args.random else eff_checkpoint}
        output = {"meta": meta, "results": results}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")
        return output
    finally:
        if wandb_module is not None:
            wandb_module.finish()
        if env_instance is not None:
            env_instance.close()
