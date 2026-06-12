import json

import pytest

from src.shared.profiling import (
    block_until_ready_tree,
    init_phase_times,
    make_synthetic_rgb_frame,
    measure_ms,
    render_phase_table,
    summarize_phase_times,
    summarize_values_ms,
    write_json,
)


def test_init_phase_times_creates_empty_lists():
    assert init_phase_times(["a", "b"]) == {"a": [], "b": []}


def test_summarize_values_ms_empty_has_stable_keys():
    assert summarize_values_ms([]) == {
        "n": 0,
        "mean_ms": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
        "total_s": 0.0,
    }


def test_summarize_values_ms_uses_existing_percentile_indexing():
    stats = summarize_values_ms([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean_ms"] == pytest.approx(2.5)
    assert stats["p50_ms"] == pytest.approx(3.0)
    assert stats["p95_ms"] == pytest.approx(4.0)
    assert stats["min_ms"] == pytest.approx(1.0)
    assert stats["max_ms"] == pytest.approx(4.0)
    assert stats["total_s"] == pytest.approx(0.010)


def test_summarize_phase_times_preserves_phase_keys():
    out = summarize_phase_times({"env_step": [2.0], "train_step": []})
    assert out["env_step"]["mean_ms"] == pytest.approx(2.0)
    assert out["train_step"]["n"] == 0


def test_measure_ms_validates_counts_and_returns_ms_stats():
    with pytest.raises(ValueError):
        measure_ms(lambda: None, n=0)
    mean_ms, std_ms = measure_ms(lambda: None, n=2, warmup=1)
    assert mean_ms >= 0.0
    assert std_ms >= 0.0


def test_block_until_ready_tree_walks_nested_containers():
    class Leaf:
        def __init__(self):
            self.called = False

        def block_until_ready(self):
            self.called = True
            return "ready"

    left = Leaf()
    right = Leaf()
    out = block_until_ready_tree({"x": [left, (right, 3)]})
    assert out == {"x": ["ready", ("ready", 3)]}
    assert left.called
    assert right.called


def test_write_json_creates_parent_directory(tmp_path):
    out = write_json(tmp_path / "nested" / "payload.json", {"ok": True})
    assert json.loads(out.read_text()) == {"ok": True}


def test_render_phase_table_formats_requested_columns():
    table = render_phase_table(
        {"env_step": {"mean_ms": 1.2345, "n": 2}},
        ["env_step"],
        [("mean", "mean_ms", ".2f"), ("calls", "n", "d")],
        col_width=8,
    )
    assert "   phase |     mean |    calls" in table
    assert "env_step |     1.23 |        2" in table


def test_make_synthetic_rgb_frame_is_deterministic_uint8_chw():
    first = make_synthetic_rgb_frame(7, size=4)
    second = make_synthetic_rgb_frame(7, size=4)
    other = make_synthetic_rgb_frame(8, size=4)
    assert first.shape == (3, 4, 4)
    assert first.dtype.name == "uint8"
    assert (first == second).all()
    assert not (first == other).all()
