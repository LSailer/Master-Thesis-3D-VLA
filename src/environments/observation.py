"""Typed environment observation frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


import jax.numpy as jnp

@dataclass(frozen=True)
class ObservationFrame:
    """Raw observation returned by environment wrappers.

    ``previous_action`` is the discrete action that produced this frame. It is
    ``None`` for reset frames because no environment action preceded them.
    """

    image: jnp.ndarray
    is_first: bool
    previous_action: int | None = None
    reward: float = 0.0
    done: bool = False
    success: float = 0.0
    spl: float = 0.0
    softspl: float = 0.0
    dtg: float = 0.0
    collision_rate: float = 0.0
    scene_id: str = ""
    episode_id: str | None = None
    step: int | None = None
    invalid_goal_distance: float = 0.0
    invalid_goal_distance_raw: str | None = None

    @property
    def is_episode_end(self) -> bool:
        """Whether this frame is the final frame before an environment reset."""
        return self.done

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Return an attribute value with a mapping-style default."""
        return getattr(self, key, default)
