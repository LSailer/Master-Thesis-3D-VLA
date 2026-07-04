"""Per-episode Habitat ObjectNav metrics for experiment dashboards.

Separated from :mod:`src.environments.habitat` so the simulation env stays
free of experiment-reporting concerns: this module owns the windowed
``EpisodeTracker`` wiring and the W&B-keyed metric dicts consumed by the
trainer's ``episode_metrics_fn`` hook.
"""

from typing import Any

import numpy as np

from src.environments.observation import ObservationFrame


class HabitatEpisodeMetrics:
    """Per-episode Habitat ObjectNav metrics callback bound to one env.

    Owns a private ``EpisodeTracker`` (windowed running averages) and reads
    navigation stats off the Habitat env it is constructed with. Use one
    instance per env: the train rollout leaves ``track_collision_rate`` False so
    the dashboard isn't doubly-noisy, while the val loop passes
    ``track_collision_rate=True`` for its independent tracker.

    Attributes:
        _env: The Habitat env whose ``current_episode``/path stats are read.
        _tracker: The instance-owned windowed metrics tracker.
        _action_names: Discrete action index to dashboard label mapping.
    """

    def __init__(self, env: Any, *, track_collision_rate: bool = False) -> None:
        """Bind the metrics callback to a Habitat env.

        Args:
            env: Habitat ObjectNav env exposing ``current_episode``,
                ``_path_length``, and ``_start_geodesic``.
            track_collision_rate: Whether the owned tracker records collision
                rate; pass True for the val loop, False for train rollouts.
        """
        from src.shared.wandb_utils import EpisodeTracker

        self._env = env
        self._tracker = EpisodeTracker(
            window=100, track_collision_rate=track_collision_rate
        )
        self._action_names = {0: "stop", 1: "forward", 2: "left", 3: "right"}

    def __call__(
        self,
        last_obs: ObservationFrame,
        episode_reward: float,
        episode_steps: int,
        action_counts: np.ndarray,
    ) -> dict[str, Any]:
        """Record one finished episode and return its dashboard metrics.

        Args:
            last_obs: Final ``ObservationFrame`` of the episode.
            episode_reward: Total reward accumulated over the episode.
            episode_steps: Number of environment steps taken.
            action_counts: Per-action step counts for the episode.

        Returns:
            Dict of windowed and per-episode metrics keyed for W&B logging.
        """
        env = self._env
        success = last_obs.success
        spl = last_obs.spl
        softspl = last_obs.softspl
        dtg = last_obs.dtg
        collision_rate = last_obs.collision_rate
        episode = env.current_episode
        category = getattr(episode, "object_category", "unknown")
        scene_raw = getattr(episode, "scene_id", "")
        path_length = env._path_length
        shortest_path = env._start_geodesic
        path_ratio = path_length / shortest_path if shortest_path > 0 else 0.0

        tracked = self._tracker.record(
            reward=episode_reward,
            success=success,
            spl=spl,
            category=category,
            scene_id=scene_raw,
            softspl=softspl,
            dtg=dtg,
            collision_rate=collision_rate,
        )

        action_pcts = action_counts / max(episode_steps, 1)
        return {
            **tracked,
            "episode/steps": episode_steps,
            "episode/path_length": path_length,
            "episode/shortest_path": shortest_path,
            "episode/path_ratio": path_ratio,
            "episode_reset": 1,
            **{
                f"action/{self._action_names[i]}_pct": float(action_pcts[i])
                for i in range(len(action_counts))
            },
        }
