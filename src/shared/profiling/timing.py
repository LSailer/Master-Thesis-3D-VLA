"""Dependency-light timing and summary helpers."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence


PhaseTimes = dict[str, list[float]]


@contextmanager
def timed(phase_times: dict[str, list[float]], phase: str):
    """Wall-clock timer, records milliseconds into ``phase_times[phase]``.

    The caller is responsible for bracketing work that has already synchronised
    with the device (e.g. ``int()`` cast on a JAX scalar, or a manual
    ``block_until_ready``) -- otherwise the recorded time will miss async work.
    """
    t0 = time.perf_counter()
    yield
    phase_times[phase].append((time.perf_counter() - t0) * 1000.0)


def init_phase_times(phases: Iterable[str]) -> PhaseTimes:
    """Return an empty millisecond accumulator for each named phase."""
    return {phase: [] for phase in phases}


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize_values_ms(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize already-synchronized millisecond timings."""
    vals = [float(v) for v in values]
    if not vals:
        return {
            "n": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "total_s": 0.0,
        }
    return {
        "n": len(vals),
        "mean_ms": mean(vals),
        "p50_ms": _percentile(vals, 0.50),
        "p95_ms": _percentile(vals, 0.95),
        "min_ms": min(vals),
        "max_ms": max(vals),
        "total_s": sum(vals) / 1000.0,
    }


def summarize_phase_times(
    phase_times: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float | int]]:
    """Summarize a phase-to-samples mapping."""
    return {phase: summarize_values_ms(values) for phase, values in phase_times.items()}


def measure_ms(
    fn: Callable[[], Any], n: int = 20, warmup: int = 3
) -> tuple[float, float]:
    """Run ``fn`` after warmup and return ``(mean_ms, std_ms)``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    avg = mean(samples)
    variance = mean([(x - avg) ** 2 for x in samples])
    return float(avg), float(math.sqrt(variance))


def block_until_ready_tree(value: Any) -> Any:
    """Call ``block_until_ready`` on every leaf that exposes it."""
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if isinstance(value, dict):
        return {k: block_until_ready_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [block_until_ready_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(block_until_ready_tree(v) for v in value)
    return value


def render_phase_table(
    rows: Mapping[str, Mapping[str, Any]],
    phase_order: Sequence[str],
    columns: Sequence[str | tuple[str, str, str]],
    *,
    phase_header: str = "phase",
    col_width: int = 16,
) -> str:
    """Render a fixed-width phase table.

    ``columns`` entries can be either a key string or ``(header, key, fmt)``.
    ``fmt`` is a normal format spec, for example ``".3f"``.
    """
    specs: list[tuple[str, str, str]] = []
    for column in columns:
        if isinstance(column, str):
            specs.append((column, column, ""))
        else:
            specs.append(column)

    headers = [phase_header, *[header for header, _, _ in specs]]
    lines = [
        " | ".join(f"{header:>{col_width}}" for header in headers),
        "-+-".join("-" * col_width for _ in headers),
    ]
    for phase in phase_order:
        row = rows.get(phase, {})
        cells = [phase]
        for _, key, fmt in specs:
            value = row.get(key, 0.0)
            cells.append(format(value, fmt) if fmt else str(value))
        lines.append(" | ".join(f"{cell:>{col_width}}" for cell in cells))
    return "\n".join(lines)
