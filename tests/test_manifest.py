"""Unit tests for src.r2dreamer.manifest."""

from __future__ import annotations

import json
from pathlib import Path

from src.r2dreamer.manifest import write_manifest_end, write_manifest_start


def test_write_manifest_start(tmp_path: Path) -> None:
    cfg = {"lr": 1e-4, "batch_size": 16, "tag": "smoke"}
    path = write_manifest_start(tmp_path, cfg)

    assert path == tmp_path / "MANIFEST.json"
    assert path.exists()

    data = json.loads(path.read_text())
    expected_keys = {
        "git_sha", "git_branch", "git_dirty", "config", "wandb_id",
        "slurm_id", "started_at", "hostname", "python_version",
    }
    assert expected_keys.issubset(data.keys())
    assert data["config"] == cfg
    assert isinstance(data["git_sha"], str) and len(data["git_sha"]) == 40
    assert isinstance(data["git_dirty"], bool)


def test_write_manifest_end_appends(tmp_path: Path) -> None:
    write_manifest_start(tmp_path, {"k": "v"})
    write_manifest_end(tmp_path, "completed")

    data = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert data["status"] == "completed"
    assert "ended_at" in data
    # Existing keys preserved.
    assert data["config"] == {"k": "v"}
    assert "started_at" in data


def test_write_manifest_end_without_start(tmp_path: Path) -> None:
    # Should not crash if MANIFEST.json doesn't exist yet.
    write_manifest_end(tmp_path, "failed")

    path = tmp_path / "MANIFEST.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["status"] == "failed"
    assert "ended_at" in data
