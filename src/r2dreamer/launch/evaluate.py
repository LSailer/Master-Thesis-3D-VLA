"""Public evaluate() entry point for the r2dreamer launcher.

This is the thin orchestration layer. The heavy lifting is split across:

* ``eval_cli.py`` -- composition root: checkpoint/manifest resolution and
  concrete env/encoder/agent construction.
* ``eval_loop.py`` -- rollout Protocols and side-effect-free helpers.
* ``eval_artifacts.py`` -- video/top-down/JSON writing and the W&B lifecycle.
* ``settings.py`` -- the shared CLI-vs-shim precedence resolution.

The stateful rollout driver (``_run_eval_episode`` + its ``_start_eval_episode``
/ ``_get_agent_heading`` collaborators) stays here so cross-references and test
monkeypatches resolve in a single namespace. The manifest/artifact helpers are
re-exported below for the callers (``src.main``, tests) that import them from
this module.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import jax
from scipy.spatial.transform import Rotation

from src.configs.config import R2DreamerConfig
from src.baselines.random_agent import RandomAgent
from src.r2dreamer.launch.eval_artifacts import (
    init_eval_wandb as _init_eval_wandb,
    initial_video_frames as _initial_eval_video_frames,
    append_video_frame as _append_eval_video_frame,
    obs_value as _obs_value,
    print_eval_summary as _print_eval_summary,
    write_episode_artifacts as _write_eval_episode_artifacts,
    write_eval_results as _write_eval_results,
)
from src.r2dreamer.launch.eval_cli import (
    _agent_config_kwargs,
    _find_manifest_for_checkpoint,
    _load_arch_overrides_from_manifest,
    _make_eval_agent,
    _make_eval_encoder,
    _make_eval_env,
)
from src.r2dreamer.launch.eval_loop import (
    _extract_goal_positions,
    _make_eval_episode_result,
)
from src.r2dreamer.launch.parser import _build_parser_eval
from src.r2dreamer.launch.registries import env_registry
from src.r2dreamer.launch.settings import resolve_eval_settings as _resolve_eval_settings

__all__ = [
    "evaluate",
    "_find_manifest_for_checkpoint",
    "_load_arch_overrides_from_manifest",
    "_run_eval_episode",
    "_start_eval_episode",
    "_get_agent_heading",
]


def _get_agent_heading(env):
    """Extract agent heading (yaw in radians) from habitat sim state.

    Args:
      env: The eval environment exposing ``_env.sim.get_agent_state()``.

    Returns:
      The agent yaw in radians.
    """
    state = env._env.sim.get_agent_state()
    quat = state.rotation
    r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    euler = r.as_euler("yxz")
    return float(euler[0])


def _start_eval_episode(env_instance, adapter):
    """Reset the env, prime the adapter, and snapshot the episode's start state.

    Args:
      env_instance: The eval environment.
      adapter: The observation adapter for this rollout.

    Returns:
      A tuple of (obs, encoder_obs, is_first, start_pos, goal_positions,
      scene_id, object_category, trajectory, headings).
    """
    obs = env_instance.reset()
    if adapter.on_episode_reset:
        # Pass the reset frame's scene_id so PERSIST_SCENE adapters save/restore
        # the right per-scene cache during evaluation (mirrors the trainer).
        adapter.on_episode_reset(getattr(obs, "scene_id", None) or "scene")
    prepared = adapter.prepare_env_step(obs)
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


def _run_eval_episode(
    *,
    ep_idx: int,
    args: Any,
    env_instance,
    adapter,
    agent,
    rng_key,
    config,
    wandb_module,
    output_dir: str,
) -> tuple[dict, jax.Array]:
    """Roll out one eval episode, writing artifacts, and return its result.

    Args:
      ep_idx: Episode index.
      args: Parsed argparse namespace.
      env_instance: The eval environment.
      adapter: The observation adapter for this rollout.
      agent: The agent (``RandomAgent`` or a learned agent).
      rng_key: The JAX PRNG key threaded through learned-agent steps.
      config: The resolved ``R2DreamerConfig`` (kept for signature parity).
      wandb_module: The active W&B module, or ``None``.
      output_dir: Directory for episode artifacts.

    Returns:
      A ``(episode_result, rng_key)`` tuple; ``rng_key`` is advanced when a
      learned agent consumed act-keys.

    Raises:
      RuntimeError: If a random-agent step observation lacks ``previous_action``.
    """
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
    ) = _start_eval_episode(env_instance, adapter)
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
            next_obs = agent.act()
            action = next_obs.previous_action
            if action is None:
                raise RuntimeError(
                    "random-agent step observation is missing previous_action"
                )
        else:
            rng_key, act_key = jax.random.split(rng_key)
            action, act_state = agent.act_with_state(
                encoder_obs, is_first, act_state, act_key, training=False
            )
            next_obs = env_instance.step(action)
        next_prepared = adapter.prepare_env_step(next_obs)
        next_encoder_obs = next_prepared.encoder_obs
        next_is_first = next_prepared.is_first
        actions_taken.append(int(action))
        rewards.append(float(_obs_value(next_obs, "reward")))

        pos = env_instance._env.sim.get_agent_state().position.tolist()
        trajectory.append(pos)
        headings.append(_get_agent_heading(env_instance))
        if record_video:
            _append_eval_video_frame(
                video_frames, env_instance, next_obs, trajectory, goal_positions
            )

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

    Args:
      env: Registry key for the environment.
      encoder: Shim-supplied encoder default (CLI ``--encoder`` overrides).
      curriculum: Shim-supplied curriculum default.
      checkpoint: Shim-supplied checkpoint default.
      output_dir: Shim-supplied output-dir default.
      argv: Explicit argv for programmatic callers; falls back to ``sys.argv``.

    Returns:
      Metrics dict with ``results`` and ``meta`` keys.

    Raises:
      KeyError: If ``env`` or the resolved encoder is not registered.
      ValueError: If required settings (checkpoint/output_dir) are missing.
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
        return _write_eval_results(output_path, meta, results)
    finally:
        if wandb_module is not None:
            wandb_module.finish()
        if env_instance is not None:
            env_instance.close()
