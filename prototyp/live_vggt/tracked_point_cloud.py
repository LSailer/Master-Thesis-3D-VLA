"""Tracked point-cloud state for the live VGGT prototype."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class TrackedPointCloud:
    """Global scene point cloud plus per-step visibility mask."""

    point_xyz: jnp.ndarray  # (N, 3)
    visible_steps: jnp.ndarray  # (N, max_steps), bool

    def visible_points(self, step_id: int) -> jnp.ndarray:
        """Return points visible at one recorded environment step."""

        if self.point_xyz.ndim != 2 or self.point_xyz.shape[1] != 3:
            raise ValueError(f"expected point_xyz shape (N, 3), got {self.point_xyz.shape}")
        if self.visible_steps.ndim != 2:
            raise ValueError(
                f"expected visible_steps shape (N, max_steps), got {self.visible_steps.shape}"
            )
        if self.point_xyz.shape[0] != self.visible_steps.shape[0]:
            raise ValueError(
                "point_xyz and visible_steps disagree on point count: "
                f"{self.point_xyz.shape[0]} != {self.visible_steps.shape[0]}"
            )
        if step_id < 0 or step_id >= self.visible_steps.shape[1]:
            raise IndexError(
                f"step_id {step_id} outside visible_steps width {self.visible_steps.shape[1]}"
            )

        return self.point_xyz[self.visible_steps[:, step_id]]
