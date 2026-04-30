"""Utilities for loading and iterating over the ALFRED JSON dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "alfred" / "json_2.1.0"

SPLITS = ("train", "valid_seen", "valid_unseen", "tests_seen", "tests_unseen")


@dataclass
class Trajectory:
    """Parsed ALFRED traj_data.json."""

    task_id: str
    task_type: str
    scene: str  # e.g. "FloorPlan28"
    goal: str  # high-level natural language goal
    instructions: list[str]  # step-by-step annotator instructions
    plan: list[dict]  # high-level action plan from PDDL
    low_actions: list[dict]  # low-level API actions
    turk_annotations: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)


def load_trajectory(traj_dir: str | Path) -> Trajectory:
    """Load a single trajectory from its directory (containing traj_data.json)."""
    traj_dir = Path(traj_dir)
    traj_path = traj_dir / "traj_data.json"
    with open(traj_path) as f:
        data = json.load(f)

    turk = data.get("turk_annotations", {}).get("anns", [])
    # Use first annotator's instructions by default
    instructions = turk[0]["high_descs"] if turk else []
    goal = turk[0]["task_desc"] if turk else data.get("task_desc", "")

    return Trajectory(
        task_id=data["task_id"],
        task_type=data["task_type"],
        scene=data["scene"]["scene_num"] if isinstance(data["scene"], dict) else str(data["scene"]),
        goal=goal,
        instructions=instructions,
        plan=data.get("plan", {}).get("high_pddl", []),
        low_actions=data.get("plan", {}).get("low_actions", []),
        turk_annotations=turk,
        raw=data,
    )


def iter_dataset(
    split: str = "train",
    data_root: str | Path | None = None,
) -> Iterator[Trajectory]:
    """Iterate over all trajectories in a split."""
    root = Path(data_root) if data_root else DATA_ROOT
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    for task_type_dir in sorted(split_dir.iterdir()):
        if not task_type_dir.is_dir():
            continue
        for trial_dir in sorted(task_type_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            traj_path = trial_dir / "traj_data.json"
            if traj_path.exists():
                yield load_trajectory(trial_dir)


@dataclass
class DatasetStats:
    """Summary statistics for an ALFRED split."""

    split: str
    num_trajectories: int
    task_type_counts: dict[str, int]
    unique_scenes: set[str]
    avg_low_actions: float
    avg_instructions: float


def get_dataset_stats(
    split: str = "train",
    data_root: str | Path | None = None,
) -> DatasetStats:
    """Compute summary statistics for a split."""
    task_type_counts: dict[str, int] = {}
    scenes: set[str] = set()
    total_actions = 0
    total_instructions = 0
    n = 0

    for traj in iter_dataset(split, data_root):
        n += 1
        task_type_counts[traj.task_type] = task_type_counts.get(traj.task_type, 0) + 1
        scenes.add(traj.scene)
        total_actions += len(traj.low_actions)
        total_instructions += len(traj.instructions)

    return DatasetStats(
        split=split,
        num_trajectories=n,
        task_type_counts=task_type_counts,
        unique_scenes=scenes,
        avg_low_actions=total_actions / max(n, 1),
        avg_instructions=total_instructions / max(n, 1),
    )
