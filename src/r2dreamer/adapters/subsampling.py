"""Input subsampling policy bounding per-step house-buffer add cost.

Hosts :class:`InputSubsamplingPolicy`, the even-stride subsampling collaborator
the ``VGGTHousePointsPoseObsAdapter`` composes, plus the small VGGT
output-field accessors it shares with the adapter's camera-pose extraction.
Split out of ``hybrid_adapter.py``; ``hybrid_adapter`` re-exports these names
for backward compatibility.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.observation_keys import CAMERA_POSE_KEY
from src.r2dreamer.observation_preparation.vggt_readouts import VGGTOutputLike


def _vggt_output_field(out: VGGTOutputLike, field_name: str) -> Any | None:
    """Return one VGGT output field from object or legacy mapping outputs."""
    if isinstance(out, Mapping):
        value = out.get(field_name)
        if value is None and field_name == "world_points":
            value = out.get("dense_world_points")
        if value is None and field_name == "camera_pose":
            value = out.get(CAMERA_POSE_KEY)
        return value
    return getattr(out, field_name, None)


def _required_vggt_output_field(out: VGGTOutputLike, field_name: str) -> Any:
    """Return a required VGGT output field or raise a contract error."""
    value = _vggt_output_field(out, field_name)
    if value is None:
        raise ValueError(f"VGGT output field {field_name!r} is required")
    return value


class InputSubsamplingPolicy:
    """Even-stride subsampling policy bounding per-step buffer-add cost.

    By default every VGGT point-map pixel is fed to the buffer each step
    (``max_input_points <= 0``); when a positive budget is set, the ``(H, W)``
    map is strided down to approximately that many points before ``add``.
    """

    def __init__(self, max_input_points: int = 0):
        self._max_input_points = int(max_input_points)

    @property
    def max_input_points(self) -> int:
        """Configured input-point budget (``<= 0`` disables subsampling)."""
        return self._max_input_points

    def stride(self, height: int, width: int) -> int:
        """Return the even stride that caps a ``(height, width)`` map to budget.

        Args:
          height: Point-map height in pixels.
          width: Point-map width in pixels.

        Returns:
          ``1`` if no subsampling is needed, else the smallest stride whose
          strided grid area is at or below ``max_input_points``.
        """
        total = int(height) * int(width)
        if self._max_input_points <= 0 or total <= self._max_input_points:
            return 1
        return max(1, int(math.ceil(math.sqrt(total / self._max_input_points))))

    def subsample(
        self, out: VGGTOutputLike, env_obs: ObservationFrame
    ) -> tuple[Any, ObservationFrame]:
        """Return ``(vggt_output, observation)`` strided to bound ``add`` cost.

        Strides the ``(H, W, 3)`` world map, ``(H, W)`` confidence and
        ``(3, H, W)`` image together so they stay pixel-aligned, then wraps
        them in lightweight stand-ins that expose exactly the fields ``add``
        reads.

        Args:
          out: Raw VGGT extractor output (object or legacy mapping).
          env_obs: Environment frame that produced ``out``.

        Returns:
          ``(vggt_output, observation)``, strided if the configured budget
          requires it; otherwise ``out``/``env_obs`` are passed through
          (only re-wrapped into a shim when ``out`` is a legacy mapping).
        """
        world_points = _required_vggt_output_field(out, "world_points")
        confidence = _required_vggt_output_field(out, "confidence")
        stride = self.stride(world_points.shape[0], world_points.shape[1])
        if stride == 1:
            if not isinstance(out, Mapping):
                return out, env_obs
            shim_out = SimpleNamespace(world_points=world_points, confidence=confidence)
            return shim_out, env_obs
        strided_points = world_points[::stride, ::stride, :]
        strided_conf = jnp.asarray(confidence)[::stride, ::stride]
        strided_image = np.asarray(env_obs.image)[:, ::stride, ::stride]
        shim_out = SimpleNamespace(
            world_points=strided_points, confidence=strided_conf
        )
        return shim_out, dataclasses.replace(env_obs, image=strided_image)
