"""Aggregation + rendering for the 3D-46 JAX-vs-external comparison builder.

Pure-stdlib (no torch/jax), so it runs in any venv.
"""

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "r2dreamer" / "build_offline_comparison.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_offline_comparison", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(dyn, rep, rew, k1, k5, k15, recon=float("nan")):
    return {
        "reconstruction_nll": recon,
        "dynamics_kl": dyn,
        "representation_kl": rep,
        "reward_mse": rew,
        "k_step_rollout_k1": k1,
        "k_step_rollout_k5": k5,
        "k_step_rollout_k15": k15,
    }


def test_aggregate_mean_std():
    mod = _load()
    rows = {
        "0": _row(10.0, 10.0, 1.0, 0.1, 0.5, 1.5),
        "1": _row(12.0, 12.0, 2.0, 0.2, 0.6, 1.6),
        "2": _row(14.0, 14.0, 3.0, 0.3, 0.7, 1.7),
    }
    agg = mod.aggregate(rows)
    assert agg["dynamics_kl"]["mean"] == 12.0
    assert agg["dynamics_kl"]["n"] == 3
    assert abs(agg["dynamics_kl"]["std"] - 2.0) < 1e-9  # sample stdev of 10,12,14
    assert abs(agg["reward_mse"]["mean"] - 2.0) < 1e-9
    # recon NLL must not be a comparable column.
    assert "reconstruction_nll" not in agg


def test_aggregate_ignores_nan_and_missing():
    mod = _load()
    rows = {
        "0": _row(10.0, 10.0, float("nan"), 0.1, 0.5, 1.5),
        "1": _row(12.0, 12.0, 2.0, 0.2, 0.6, 1.6),
    }
    agg = mod.aggregate(rows)
    # reward_mse has one finite value (seed 1); n=1, std=0.
    assert agg["reward_mse"]["n"] == 1
    assert agg["reward_mse"]["mean"] == 2.0
    assert agg["reward_mse"]["std"] == 0.0


def test_seed_from_path():
    mod = _load()
    assert mod._seed_from_path("output/3d46/wp_cp-seed2/run-99/heldout_table_row.json") == "2"
    assert mod._seed_from_path("ext-wp_cp-seed0") == "0"


def test_end_to_end_glob_and_outputs(tmp_path):
    mod = _load()
    # Synthesize 3 external seed dirs with table rows.
    for s in range(3):
        d = tmp_path / f"wp_cp-seed{s}" / "run-1"
        d.mkdir(parents=True)
        (d / "heldout_table_row.json").write_text(
            json.dumps(_row(10.0 + s, 10.0 + s, 1.0 + s, 0.1, 0.5, 1.5))
        )
    ext_glob = str(tmp_path / "wp_cp-seed*" / "run-*" / "heldout_table_row.json")
    out_md = tmp_path / "cmp.md"
    out_csv = tmp_path / "cmp.csv"

    mod.main([
        "--external-glob", ext_glob,
        "--out-md", str(out_md),
        "--out-csv", str(out_csv),
    ])

    md = out_md.read_text()
    assert "external PyTorch (3D-46)" in md
    assert "11.0000 ± 1.0000 (n=3)" in md  # dynamics_kl mean of 10,11,12
    assert "_pending_" in md  # JAX column has no data yet
    # CSV has the external per-seed rows (3 seeds × 6 comparable metrics).
    csv_lines = out_csv.read_text().strip().splitlines()
    assert csv_lines[0] == "framework,seed,metric,value"
    assert sum(1 for ln in csv_lines if ln.startswith("pytorch_external")) == 18
