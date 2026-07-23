"""Plot helpers for the live VGGT point-cloud prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
from matplotlib.figure import Figure
import numpy as np
import plotly.graph_objects as go

from prototyp.live_vggt.tracked_point_cloud import TrackedPointCloud

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "prototype_live_vggt"
DEFAULT_CHANGE_PLOT_PATH = DEFAULT_OUTPUT_DIR / "different_points_new_counts.png"
DEFAULT_TRACKED_POINT_CLOUD_DIR = DEFAULT_OUTPUT_DIR / "tracked_point_cloud"


class TrackedPointCloudPlotPaths(NamedTuple):
    """Output files written by one tracked-cloud record call."""

    snapshot_png: Path
    interactive_html: Path


class _PointGroups(NamedTuple):
    """Point groups for one step in tracked-cloud time."""

    inactive: np.ndarray
    retained: np.ndarray
    new: np.ndarray

    @property
    def axis_points(self) -> np.ndarray:
        """Return all points visible in the plot frame."""

        populated = [
            points
            for points in (self.inactive, self.retained, self.new)
            if points.size
        ]
        if not populated:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(populated, axis=0)


class _MatplotlibPointStyle(NamedTuple):
    """Matplotlib scatter styling for one point group."""

    color: str
    label: str
    size: float
    alpha: float


@dataclass
class PointChangePlotter:
    """Record point-change counts and save a PNG after each step."""

    output_path: Path = DEFAULT_CHANGE_PLOT_PATH
    steps: list[int] = field(default_factory=list)
    added_counts: list[int] = field(default_factory=list)
    removed_counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Start each prototype run with no stale plot image."""

        if self.output_path.exists():
            self.output_path.unlink()

    def record(
        self,
        step_id: int,
        different_points_new: jnp.ndarray | np.ndarray,
        different_mask: jnp.ndarray | np.ndarray | None = None,
    ) -> None:
        """Record one step.

        ``different_points_new`` is plotted as added points. If passed,
        ``different_mask`` is plotted as removed/replaced points.
        """

        self.steps.append(int(step_id))
        self.added_counts.append(_first_dim(different_points_new))
        self.removed_counts.append(
            _true_count(different_mask) if different_mask is not None else 0
        )
        self.save()

    def save(self) -> None:
        """Save the accumulated point-change count plot."""

        if not self.steps:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = Figure(figsize=(9, 4))
        ax = fig.add_subplot(111)
        ax.plot(
            self.steps,
            self.added_counts,
            marker="o",
            label="added: different_points_new",
        )
        ax.plot(
            self.steps,
            self.removed_counts,
            marker="o",
            label="removed/replaced: different_mask",
        )
        ax.set_xlabel("step")
        ax.set_ylabel("point count")
        ax.set_title("VGGT point changes, one scene")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(self.output_path, dpi=160)


@dataclass
class TrackedPointCloudPlotter:
    """Render tracked point-cloud evolution as PNG snapshots and Plotly HTML."""

    output_dir: Path = DEFAULT_TRACKED_POINT_CLOUD_DIR
    max_points_per_group: int = 8_000
    include_plotlyjs: bool | str = True
    clear_on_init: bool = True
    recorded_steps_by_scene: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Prepare the output directory and remove stale generated plots."""

        self.output_dir = Path(self.output_dir)
        if self.clear_on_init:
            self._clear_generated_outputs()

    def record(
        self,
        step_id: int,
        tracked_point_cloud: TrackedPointCloud,
        scene_id: str = "scene",
    ) -> TrackedPointCloudPlotPaths:
        """Write a step PNG and refresh the scene-level interactive HTML."""

        steps = self.recorded_steps_by_scene.setdefault(scene_id, [])
        if step_id not in steps:
            steps.append(step_id)
            steps.sort()

        snapshot_png = self._save_step_png(scene_id, step_id, tracked_point_cloud)
        interactive_html = self._write_interactive_html(
            scene_id, steps, step_id, tracked_point_cloud
        )
        return TrackedPointCloudPlotPaths(snapshot_png, interactive_html)

    def _clear_generated_outputs(self) -> None:
        if not self.output_dir.exists():
            return
        for pattern in ("step_*.png", "tracked_point_cloud.html"):
            for output_path in self.output_dir.rglob(pattern):
                if output_path.is_file():
                    output_path.unlink()

    def _save_step_png(
        self,
        scene_id: str,
        step_id: int,
        tracked_point_cloud: TrackedPointCloud,
    ) -> Path:
        groups = _point_groups_for_step(tracked_point_cloud, step_id)
        output_dir = self._scene_output_dir(scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"step_{step_id:05d}.png"

        fig = Figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        self._draw_snapshot_groups(ax, groups)
        _set_equal_xyz_axes(ax, groups.axis_points)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(_step_title(scene_id, step_id, groups))
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        return output_path

    def _draw_snapshot_groups(self, ax: object, groups: _PointGroups) -> None:
        _draw_matplotlib_points(
            ax,
            _deterministic_subsample(groups.inactive, self.max_points_per_group),
            _MatplotlibPointStyle("lightgray", "inactive, seen before", 0.5, 0.12),
        )
        _draw_matplotlib_points(
            ax,
            _deterministic_subsample(groups.retained, self.max_points_per_group),
            _MatplotlibPointStyle("#1f77b4", "visible, already tracked", 2.0, 0.9),
        )
        _draw_matplotlib_points(
            ax,
            _deterministic_subsample(groups.new, self.max_points_per_group),
            _MatplotlibPointStyle("#ff7f0e", "new this step", 6.0, 1.0),
        )

    def _write_interactive_html(
        self,
        scene_id: str,
        steps: list[int],
        active_step: int,
        tracked_point_cloud: TrackedPointCloud,
    ) -> Path:
        output_dir = self._scene_output_dir(scene_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "tracked_point_cloud.html"

        frames = [
            _plotly_frame_for_step(
                scene_id,
                step,
                tracked_point_cloud,
                self.max_points_per_group,
            )
            for step in steps
        ]
        active_index = steps.index(active_step)
        fig = go.Figure(data=frames[active_index].data, frames=frames)
        active_groups = _point_groups_for_step(tracked_point_cloud, active_step)
        fig.update_layout(
            title=_step_title(scene_id, active_step, active_groups, line_break="<br>"),
            margin={"l": 0, "r": 0, "b": 0, "t": 60},
            showlegend=True,
            scene=_plotly_scene_layout(tracked_point_cloud, steps),
            sliders=[_plotly_slider(steps, active_index)],
            updatemenus=[_plotly_buttons()],
        )
        fig.write_html(output_path, include_plotlyjs=self.include_plotlyjs)
        return output_path

    def _scene_output_dir(self, scene_id: str) -> Path:
        return self.output_dir / _safe_path_name(scene_id)


def _point_groups_for_step(
    tracked_point_cloud: TrackedPointCloud, step_id: int
) -> _PointGroups:
    point_xyz = _point_xyz_numpy(tracked_point_cloud)
    visible_steps = _visible_steps_numpy(tracked_point_cloud, point_xyz.shape[0])
    if step_id < 0 or step_id >= visible_steps.shape[1]:
        raise IndexError(
            f"step_id {step_id} outside visible_steps width {visible_steps.shape[1]}"
        )

    first_visible_steps = _first_visible_steps(visible_steps)
    finite_mask = np.isfinite(point_xyz).all(axis=1)
    visible_now = visible_steps[:, step_id] & finite_mask
    seen_before = (first_visible_steps >= 0) & (first_visible_steps < step_id)

    new_mask = visible_now & (first_visible_steps == step_id)
    retained_mask = visible_now & seen_before
    inactive_mask = (~visible_now) & seen_before & finite_mask

    return _PointGroups(
        inactive=point_xyz[inactive_mask],
        retained=point_xyz[retained_mask],
        new=point_xyz[new_mask],
    )


def _point_xyz_numpy(tracked_point_cloud: TrackedPointCloud) -> np.ndarray:
    point_xyz = np.asarray(tracked_point_cloud.point_xyz)
    if point_xyz.ndim != 2 or point_xyz.shape[1] != 3:
        raise ValueError(f"expected point_xyz shape (N, 3), got {point_xyz.shape}")
    return point_xyz


def _visible_steps_numpy(
    tracked_point_cloud: TrackedPointCloud, point_count: int
) -> np.ndarray:
    visible_steps = np.asarray(tracked_point_cloud.visible_steps, dtype=bool)
    if visible_steps.ndim != 2:
        raise ValueError(
            f"expected visible_steps shape (N, max_steps), got {visible_steps.shape}"
        )
    if point_count != visible_steps.shape[0]:
        raise ValueError(
            "point_xyz and visible_steps disagree on point count: "
            f"{point_count} != {visible_steps.shape[0]}"
        )
    return visible_steps


def _first_visible_steps(visible_steps: np.ndarray) -> np.ndarray:
    has_visible_step = visible_steps.any(axis=1)
    first_visible_steps = np.argmax(visible_steps, axis=1).astype(np.int64)
    first_visible_steps[~has_visible_step] = -1
    return first_visible_steps


def _plotly_frame_for_step(
    scene_id: str,
    step_id: int,
    tracked_point_cloud: TrackedPointCloud,
    max_points_per_group: int,
) -> go.Frame:
    groups = _point_groups_for_step(tracked_point_cloud, step_id)
    traces = _plotly_traces(groups, max_points_per_group)
    return go.Frame(
        data=traces,
        name=str(step_id),
        layout=go.Layout(
            title=_step_title(scene_id, step_id, groups, line_break="<br>")
        ),
    )


def _plotly_traces(
    groups: _PointGroups, max_points_per_group: int
) -> list[go.Scatter3d]:
    return [
        _plotly_trace(
            _deterministic_subsample(groups.inactive, max_points_per_group),
            name="inactive, seen before",
            color="lightgray",
            size=2,
            opacity=0.18,
        ),
        _plotly_trace(
            _deterministic_subsample(groups.retained, max_points_per_group),
            name="visible, already tracked",
            color="#1f77b4",
            size=3,
            opacity=0.9,
        ),
        _plotly_trace(
            _deterministic_subsample(groups.new, max_points_per_group),
            name="new this step",
            color="#ff7f0e",
            size=5,
            opacity=1.0,
        ),
    ]


def _plotly_trace(
    points: np.ndarray,
    name: str,
    color: str,
    size: int,
    opacity: float,
) -> go.Scatter3d:
    if points.size:
        x_values = points[:, 0]
        y_values = points[:, 1]
        z_values = points[:, 2]
    else:
        x_values = []
        y_values = []
        z_values = []
    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="markers",
        name=name,
        marker={"size": size, "color": color, "opacity": opacity},
    )


def _plotly_scene_layout(
    tracked_point_cloud: TrackedPointCloud, steps: list[int]
) -> dict[str, object]:
    axis_points = _axis_points_for_steps(tracked_point_cloud, steps)
    x_range, y_range, z_range = _xyz_axis_ranges(axis_points)
    return {
        "aspectmode": "data",
        "xaxis": {"title": "x", "range": x_range},
        "yaxis": {"title": "y", "range": y_range},
        "zaxis": {"title": "z", "range": z_range},
    }


def _axis_points_for_steps(
    tracked_point_cloud: TrackedPointCloud, steps: list[int]
) -> np.ndarray:
    groups_by_step = [_point_groups_for_step(tracked_point_cloud, step) for step in steps]
    populated = [groups.axis_points for groups in groups_by_step if groups.axis_points.size]
    if not populated:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(populated, axis=0)


def _xyz_axis_ranges(points: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    if not points.size:
        return [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]

    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    radius = float(np.max(xyz_max - xyz_min) / 2.0)
    if not np.isfinite(radius) or radius == 0.0:
        radius = 1.0

    lower = center - radius
    upper = center + radius
    return (
        [float(lower[0]), float(upper[0])],
        [float(lower[1]), float(upper[1])],
        [float(lower[2]), float(upper[2])],
    )


def _plotly_slider(steps: list[int], active_index: int) -> dict[str, object]:
    return {
        "active": active_index,
        "currentvalue": {"prefix": "step "},
        "steps": [
            {
                "args": [
                    [str(step)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": str(step),
                "method": "animate",
            }
            for step in steps
        ],
    }


def _plotly_buttons() -> dict[str, object]:
    return {
        "type": "buttons",
        "showactive": False,
        "x": 0.05,
        "y": 0.0,
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 350, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    },
                ],
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
            },
        ],
    }


def _draw_matplotlib_points(
    ax: Any, points: np.ndarray, style: _MatplotlibPointStyle
) -> None:
    if not points.size:
        return
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=style.color,
        s=style.size,
        alpha=style.alpha,
        depthshade=False,
        label=style.label,
    )


def _deterministic_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices]


def _set_equal_xyz_axes(ax: Any, points: np.ndarray) -> None:
    x_range, y_range, z_range = _xyz_axis_ranges(points)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_range[0], z_range[1])


def _step_title(
    scene_id: str, step_id: int, groups: _PointGroups, line_break: str = "\n"
) -> str:
    return (
        f"Tracked VGGT point cloud: {scene_id}, step {step_id}"
        f"{line_break}new={groups.new.shape[0]} "
        f"retained={groups.retained.shape[0]} "
        f"inactive={groups.inactive.shape[0]}"
    )


def _safe_path_name(name: str) -> str:
    safe_chars = [char if char.isalnum() or char in "._-" else "_" for char in name]
    safe_name = "".join(safe_chars).strip("._")
    return safe_name or "scene"


def _first_dim(array: jnp.ndarray | np.ndarray) -> int:
    return int(array.shape[0])


def _true_count(mask: jnp.ndarray | np.ndarray) -> int:
    return int(np.asarray(mask).sum())
