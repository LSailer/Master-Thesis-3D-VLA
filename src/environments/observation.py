"""Typed environment observation frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservationFrame:
    """Raw observation returned by environment wrappers."""

    image: np.ndarray
    is_first: bool
    reward: float = 0.0
    done: bool = False
    success: float = 0.0
    spl: float = 0.0
    softspl: float = 0.0
    dtg: float = 0.0
    collision_rate: float = 0.0
    scene_id: str | None = None
    episode_id: str | None = None
    step: int | None = None
    invalid_goal_distance: float = 0.0
    invalid_goal_distance_raw: str | None = None
    is_last: bool = False
    is_terminal: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
