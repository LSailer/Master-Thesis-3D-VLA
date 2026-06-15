"""MANIFEST.json writer for training runs.

Captures git-sha, config snapshot, wandb-id, slurm-id, timestamps and
host info on run start; appends end-time and exit-status on run end.

Per recap decision #9 of 2026-04-26-output-restructure.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "MANIFEST.json"


def public_config_snapshot(config: dict) -> dict:
    """Return the public config snapshot written to new run manifests."""
    snapshot = dict(config)
    if "encoder_type" in snapshot:
        snapshot["observation_preparation_type"] = snapshot.pop("encoder_type")
    return snapshot


def _git(*args: str) -> str:
    """Run a git command and return stdout stripped. Lets git failures propagate."""
    return subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _wandb_id() -> str | None:
    """Best-effort lookup: env var first, then active wandb run."""
    env = os.environ.get("WANDB_RUN_ID")
    if env:
        return env
    try:
        import wandb  # noqa: WPS433 — optional dep
        run = getattr(wandb, "run", None)
        return run.id if run is not None else None
    except Exception:
        return None


def write_manifest_start(run_dir: Path, config: dict) -> Path:
    """Write initial MANIFEST.json into run_dir; return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config": public_config_snapshot(config),
        "wandb_id": _wandb_id(),
        "slurm_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
    }
    path = run_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def write_manifest_end(run_dir: Path, status: str) -> None:
    """Append ended_at + status to MANIFEST.json. Tolerates missing start."""
    path = Path(run_dir) / MANIFEST_NAME
    manifest: dict[str, Any] = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = status
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
