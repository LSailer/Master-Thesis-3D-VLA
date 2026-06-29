"""Observation/replay layout configuration for R2Dreamer input modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ReplayComponent = Literal["image", "world_points", "wp_cp", "tokens", "features"]
_ALLOWED_REPLAY_COMPONENTS = {"image", "world_points", "wp_cp", "tokens", "features"}


@dataclass(frozen=True)
class ObservationDims:
    """Dimension knobs for observation/replay layouts.

    Store knobs here, derive shapes from properties. This avoids independent
    config fields like ``wp_side=37`` and ``vggt_feature_dim=4116`` drifting.
    """

    render_size: int = 518
    replay_image_size: int = 64
    wp_side: int = 37
    camera_pose_dim: int = 9
    xyz_channels: int = 3
    token_count: int = 1374
    token_dim: int = 1024

    @property
    def render_shape(self) -> tuple[int, int, int]:
        """Return CHW shape requested from the environment renderer."""
        return (3, self.render_size, self.render_size)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Return CHW RGB replay image shape."""
        return (3, self.replay_image_size, self.replay_image_size)

    @property
    def world_points_shape(self) -> tuple[int, int, int]:
        """Return CHW world-point replay shape."""
        return (self.xyz_channels, self.wp_side, self.wp_side)

    @property
    def camera_pose_shape(self) -> tuple[int]:
        """Return flattened camera-pose replay shape."""
        return (self.camera_pose_dim,)

    @property
    def wp_cp_dim(self) -> int:
        """Return flattened world-points-plus-camera-pose feature width."""
        return self.xyz_channels * self.wp_side * self.wp_side + self.camera_pose_dim

    @property
    def wp_cp_shape(self) -> tuple[int]:
        """Return shape for flattened WP/CP replay features."""
        return (self.wp_cp_dim,)

    @property
    def token_shape(self) -> tuple[int, int]:
        """Return structured VGGT token replay shape."""
        return (self.token_count, self.token_dim)

    @property
    def flat_token_shape(self) -> tuple[int]:
        """Return flattened VGGT token replay shape."""
        return (self.token_count * self.token_dim,)


@dataclass(frozen=True)
class ReplayObservationConfig:
    """Replay fields requested by a run.

    ``components`` controls what buffer_obs stores. Single-component configs map
    to the existing array replay path; multi-component configs map to dict replay.
    """

    components: tuple[ReplayComponent, ...] = ("image",)
    image_dtype: str = "uint8"
    feature_dtype: str = "float32"
    normalize_image: bool = True
    feature_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        invalid = set(self.components) - _ALLOWED_REPLAY_COMPONENTS
        if invalid:
            raise ValueError(f"unknown replay components: {sorted(invalid)}")
        if "features" in self.components and self.feature_shape is None:
            raise ValueError("feature_shape is required for generic 'features'")


@dataclass(frozen=True)
class ObservationRunConfig:
    """Observation/replay config for one run, independent of model hyperparams."""

    encoder: str = "cnn"
    dims: ObservationDims = field(default_factory=ObservationDims)
    replay: ReplayObservationConfig = field(default_factory=ReplayObservationConfig)

    def replay_field_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return replay storage shapes keyed by field name."""
        fields: dict[str, tuple[int, ...]] = {}
        dims = self.dims
        if "image" in self.replay.components:
            fields["image"] = dims.image_shape
        if "world_points" in self.replay.components:
            fields["world_points"] = dims.world_points_shape
            fields["camera_pose"] = dims.camera_pose_shape
        if "wp_cp" in self.replay.components:
            fields["wp_cp"] = dims.wp_cp_shape
        if "tokens" in self.replay.components:
            fields["tokens"] = dims.flat_token_shape
        if "features" in self.replay.components:
            fields["features"] = self.replay.feature_shape or ()
        return fields

    def replay_field_dtypes(self) -> dict[str, str]:
        """Return replay storage dtypes keyed by field name."""
        return {
            name: (
                self.replay.image_dtype
                if name == "image"
                else self.replay.feature_dtype
            )
            for name in self.replay_field_shapes()
        }

    def replay_field_normalize(self) -> dict[str, bool]:
        """Return replay sample-time normalization flags keyed by field name."""
        return {
            name: name == "image" and self.replay.normalize_image
            for name in self.replay_field_shapes()
        }
