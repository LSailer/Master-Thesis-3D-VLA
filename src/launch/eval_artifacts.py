"""Evaluation artifacts and checkpoint-architecture recovery.

The run loop in ``src.main`` is the same for train and eval; what eval adds is
bookkeeping around it: per-episode trajectories, top-down renders, W&B videos,
and ``eval_results.json``. ``EvalRecorder`` owns that bookkeeping - main calls
it at episode start, per step, and at episode end, and never touches PIL,
scipy, or the JSON layout itself.

The manifest-architecture helpers live here too: evaluation is the consumer
that must rebuild the trained run's RSSM/head shapes before parameters load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from src.r2dreamer.experience import EpisodeSummary
from src.shared.video_utils import log_episode_video, render_topdown_frame

_ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}

# Architecture fields that must match the trained run: the adapter supplies the
# encoder routing, but RSSM/head widths come from the run's own config. The
# compute-dtype gate belongs here too - it leaves param shapes untouched, so
# ``_assert_params_match`` cannot catch a run rebuilt in the wrong precision.
_ARCH_FIELDS = (
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
    "mlp_layers",
    "mlp_units",
    "mlp_layers_reward",
    "mlp_layers_cont",
    "mlp_layers_actor",
    "mlp_layers_critic",
    "twohot_bins",
    "decoder",
    "compute_dtype",
    "full_bf16",
)


def find_manifest_for_checkpoint(checkpoint: str | Path) -> Path | None:
    """Return the run manifest next to a checkpoint, or in its parent dir."""
    ckpt = Path(checkpoint).resolve()
    for candidate in (
        ckpt.parent / "MANIFEST.json",
        ckpt.parent.parent / "MANIFEST.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def arch_overrides_from_manifest(checkpoint: str | None) -> dict[str, Any]:
    """Recover the trained run's architecture fields from its manifest.

    Args:
        checkpoint: Checkpoint path, or ``None`` for a random agent.

    Returns:
        Config overrides for the fields in ``_ARCH_FIELDS`` that the manifest
        records. Empty when there is no readable manifest, in which case the
        config defaults apply and the param-tree guard in
        ``R2DreamerAgent.from_checkpoint`` catches any real mismatch.
    """
    if checkpoint is None:
        return {}
    manifest = find_manifest_for_checkpoint(checkpoint)
    if manifest is None:
        return {}
    try:
        saved = json.loads(manifest.read_text()).get("config", {})
    except (ValueError, OSError):
        return {}
    return {
        key: tuple(saved[key]) if key == "encoder_mults" else saved[key]
        for key in _ARCH_FIELDS
        if key in saved
    }


def _get_agent_heading(env: Any) -> float:
    """Extract agent heading (yaw in radians) from habitat sim state."""
    state = env.agent_state
    quat = state.rotation
    r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    euler = r.as_euler("yxz")
    return float(euler[0])


def _extract_goal_positions(env: Any) -> list[list[float]]:
    goal_positions: list[list[float]] = []
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


# One recorder owns every eval artifact sink by design (see the
# launch-main-orchestrator prototype); splitting it would re-scatter them.
class EvalRecorder:  # pylint: disable=too-many-instance-attributes
    """Per-episode eval bookkeeping, driven by the run loop in ``src.main``.

    The loop owns acting and env stepping; this class only records. Habitat
    internals (positions, headings, goals) are read directly off the env the
    loop steps, and episode-level metrics come from the collector's
    ``EpisodeSummary`` - the same ``HabitatEpisodeMetrics`` source training
    logs from.

    Args:
        env: The env instance the loop steps (source of poses and renders).
        output_dir: Directory for ``eval_results.json`` and topdown renders.
        render_topdown: Whether to save a topdown PNG per episode.
        video_episodes: How many leading episodes to log as W&B videos
            (0 disables; requires ``wandb_module``).
        wandb_module: Attached ``wandb`` module from the logger, or ``None``.
    """

    def __init__(
        self,
        *,
        env: Any,
        output_dir: str,
        render_topdown: bool = False,
        video_episodes: int = 0,
        wandb_module: Any | None = None,
    ) -> None:
        self._env = env
        self._output_dir = output_dir
        self._render_topdown = render_topdown
        self._video_episodes = video_episodes if wandb_module is not None else 0
        self._wandb = wandb_module
        self.results: list[dict[str, Any]] = []
        self._trajectory: list[list[float]] = []
        self._headings: list[float] = []
        self._actions: list[int] = []
        self._goal_positions: list[list[float]] = []
        self._scene_id = ""
        self._object_category = ""
        self._start_pos: list[float] = []
        os.makedirs(output_dir, exist_ok=True)

    @property
    def record_video(self) -> bool:
        """Whether the loop should start video capture for the next episode."""
        return len(self.results) < self._video_episodes

    def start_episode(self) -> None:
        """Capture the fresh episode's start pose, goals and identity."""
        env = self._env
        self._start_pos = env.agent_state.position.tolist()
        self._goal_positions = _extract_goal_positions(env)
        self._scene_id = env.current_episode.scene_id
        self._object_category = env.current_episode.object_category
        self._trajectory = [self._start_pos]
        self._headings = [_get_agent_heading(env)]
        self._actions = []

    def record_step(self, action: int) -> None:
        """Record one step's action and the resulting pose."""
        self._actions.append(action)
        self._trajectory.append(self._env.agent_state.position.tolist())
        self._headings.append(_get_agent_heading(self._env))

    def finish_episode(
        self, summary: EpisodeSummary, *, step: int
    ) -> dict[str, Any]:
        """Build the episode's result row and write its artifacts.

        Args:
            summary: The collector's episode summary; ``metrics/sr`` and
                ``metrics/spl`` are the success/SPL the run is scored on.
            step: Global rollout step the episode ended on. Videos log under
                this step: the run loop already logged the episode metrics at
                it, and W&B silently drops writes below its current step, so
                an episode-indexed axis would lose every video after the
                first multi-step episode.

        Returns:
            The result row appended to :attr:`results`.
        """
        ep_idx = len(self.results)
        result = {
            "episode": ep_idx,
            "scene_id": self._scene_id,
            "object_category": self._object_category,
            "steps": summary.steps,
            "reward": summary.reward,
            "success": float(summary.metrics.get("metrics/sr", 0.0)),
            "spl": float(summary.metrics.get("metrics/spl", 0.0)),
            "actions": list(self._actions),
            "action_counts": {
                name: self._actions.count(idx) for idx, name in _ACTIONS.items()
            },
            "start_position": self._start_pos,
            "goal_positions": self._goal_positions,
            "trajectory": list(self._trajectory),
            "headings": list(self._headings),
        }
        self.results.append(result)

        if self._render_topdown:
            topdown_dir = os.path.join(self._output_dir, "topdown")
            os.makedirs(topdown_dir, exist_ok=True)
            Image.fromarray(
                render_topdown_frame(self._env, self._trajectory, self._goal_positions)
            ).save(os.path.join(topdown_dir, f"episode_{ep_idx:03d}.png"))
        if summary.video_frames is not None and self._wandb is not None:
            log_episode_video(
                self._wandb,
                f"eval/episode_video_{ep_idx}",
                summary.video_frames,
                step,
            )
        print(
            f"Episode {ep_idx}: steps={summary.steps:3d}  "
            f"reward={summary.reward:.2f}  "
            f"success={result['success']:.0f}  "
            f"category={self._object_category}"
        )
        return result

    def finalize(self, *, checkpoint: str | None, random_agent: bool) -> dict[str, Any]:
        """Print the summary and write ``eval_results.json``.

        Args:
            checkpoint: Checkpoint path recorded in the results metadata.
            random_agent: Whether the run scored the random baseline.

        Returns:
            The full output dict (``meta`` + ``results``).
        """
        results = self.results
        print(f"\n--- Summary ({len(results)} episodes) ---")
        if results:
            print(f"Success: {np.mean([r['success'] for r in results]) * 100:.1f}%")
            print(f"SPL: {np.mean([r['spl'] for r in results]):.3f}")
            print(f"Mean reward: {np.mean([r['reward'] for r in results]):.2f}")
            print(f"Mean steps: {np.mean([r['steps'] for r in results]):.0f}")
        output = {
            "meta": {"agent": "random" if random_agent else checkpoint},
            "results": results,
        }
        output_path = os.path.join(self._output_dir, "eval_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")
        return output
