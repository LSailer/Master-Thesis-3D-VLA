"""Factory for HabitatObjectNavEnv used by the launcher."""

from __future__ import annotations

from pathlib import Path

from src.environments.habitat import HabitatObjectNavEnv
from src.shared.configs import DreamerConfig


def make_habitat_env(
    *,
    curriculum_path: str | Path | None = None,
    curriculum_mode: str = "train",
    seed: int = 0,
    render_resolution: int = 64,
    **kwargs,
) -> HabitatObjectNavEnv:
    """Construct a HabitatObjectNavEnv with standard training defaults."""
    config = DreamerConfig(
        obs_shape=(3, render_resolution, render_resolution),
        max_episode_steps=500,
        split="train",
        reward_type="geodesic_delta",
    )
    return HabitatObjectNavEnv(
        config,
        curriculum_path=str(curriculum_path) if curriculum_path is not None else None,
        curriculum_mode=curriculum_mode,
        seed=seed,
    )
