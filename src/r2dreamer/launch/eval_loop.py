"""Rollout-support types and pure helpers for the evaluate() loop.

The rollout is typed against small ``Protocol``s (``EvalEnv``, ``EvalAdapter``,
``EvalAgent``) rather than the concrete ``HabitatObjectNavEnv`` /
``R2DreamerAgent`` classes -- concrete construction stays in ``eval_cli.py``.

The stateful rollout driver (``_run_eval_episode``, ``_start_eval_episode``,
``_get_agent_heading``) lives in ``evaluate.py`` so that its cross-references
stay in one namespace -- callers monkeypatch ``_start_eval_episode`` /
``_get_agent_heading`` on the ``evaluate`` module and expect ``_run_eval_episode``
to pick that up. This module holds only the side-effect-free pieces.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.environments.observation import ObservationFrame
from src.r2dreamer.launch.eval_artifacts import obs_value

_ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


class EvalEnv(Protocol):
    """The environment surface the eval rollout needs."""

    _env: Any
    current_episode: Any

    def reset(self) -> Any: ...
    def step(self, action: int) -> Any: ...


class EvalAdapter(Protocol):
    """The observation-adapter surface the eval rollout needs."""

    on_episode_reset: Any

    def prepare_env_step(self, obs: Any) -> Any: ...


class EvalAgent(Protocol):
    """The learned-agent surface the eval rollout needs (non-random branch)."""

    def initial_act_state(self) -> Any: ...
    def act_with_state(
        self, encoder_obs: Any, is_first: Any, act_state: Any, act_key: Any, training: bool
    ) -> tuple[Any, Any]: ...


def _extract_goal_positions(env: EvalEnv) -> list[list[float]]:
    """Collect one goal position per goal (first view-point, else raw position).

    Args:
      env: The eval environment exposing ``current_episode.goals``.

    Returns:
      A list of ``[x, y, z]`` goal positions.
    """
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
    """Assemble the per-episode result record.

    Args:
      ep_idx: Episode index.
      scene_id: Scene identifier for the episode.
      object_category: Target object category.
      actions_taken: Actions taken during the episode.
      rewards: Per-step rewards.
      obs: The final observation.
      start_pos: The agent start position.
      goal_positions: Goal view-point positions.
      trajectory: Full agent trajectory.
      headings: Per-step agent headings.

    Returns:
      The episode-result dict.
    """
    return {
        "episode": ep_idx,
        "scene_id": scene_id,
        "object_category": object_category,
        "steps": len(actions_taken),
        "reward": sum(rewards),
        "success": float(obs_value(obs, "success")),
        "spl": float(obs_value(obs, "spl")),
        "actions": actions_taken,
        "action_counts": {
            name: actions_taken.count(idx) for idx, name in _ACTIONS.items()
        },
        "start_position": start_pos,
        "goal_positions": goal_positions,
        "trajectory": trajectory,
        "headings": headings,
    }
