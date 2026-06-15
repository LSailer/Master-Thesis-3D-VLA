"""Public observation-preparation launcher language tests."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "r2dreamer"))
import _run_configs  # noqa: E402

from src.r2dreamer.manifest import write_manifest_start  # noqa: E402


def test_run_configs_use_observation_preparation_key():
    for run_id, cfg in _run_configs.RUN_CONFIGS.items():
        assert "observation_preparation" in cfg, run_id
        assert "encoder" not in cfg, run_id


def test_manifest_config_snapshot_uses_observation_preparation_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("src.r2dreamer.manifest._git", lambda *args: "test")

    path = write_manifest_start(
        tmp_path,
        {
            "encoder_type": "vggt",
            "obs_shape": (4116,),
            "total_steps": 1,
        },
    )

    manifest = json.loads(path.read_text())
    assert manifest["config"]["observation_preparation_type"] == "vggt"
    assert "encoder_type" not in manifest["config"]
