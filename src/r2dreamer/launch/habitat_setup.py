"""Factory for HabitatObjectNavEnv used by the launcher."""

from __future__ import annotations

from pathlib import Path

from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv


def make_habitat_env(
    *,
    config: HabitatEnvConfig | None = None,
    curriculum: str | None = None,
    curriculum_path: str | Path | None = None,
    mode: str = "train",
    seed: int = 0,
    render_resolution: int = 64,
    **_kwargs,
) -> HabitatObjectNavEnv:
    """Construct a HabitatObjectNavEnv with standard training defaults."""
    if config is None:
        config = HabitatEnvConfig(
            obs_shape=(3, render_resolution, render_resolution),
            max_episode_steps=500,
            reward_type="geodesic_delta",
            curriculum=curriculum,
            curriculum_path=curriculum_path,
            mode=mode,
        )
    return HabitatObjectNavEnv(config, seed=seed)
