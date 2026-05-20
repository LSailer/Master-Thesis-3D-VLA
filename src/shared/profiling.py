"""Shared wall-clock profiling helpers.

`StepTimer` accumulates per-phase wall-clock time inside a fixed-phase loop
(the offline-buffer collector). `timed()` is a contextmanager that appends a
single millisecond reading to a per-phase list (the JAX profiling scripts).
Both helpers used to live as duplicates inside individual scripts.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any


class StepTimer:
    """Accumulate per-component wall-clock timings during the collection loop.

    The first ``warmup`` steps are dropped from the means so JAX compile, the
    first habitat scene swap, and other one-shot costs don't dominate.
    """

    PHASES = (
        "vggt_extract",
        "fp16_cast",
        "npz_append",
        "resize",
        "agent_act",
        "env_step",
        "bookkeeping",
    )

    def __init__(self, warmup: int = 100) -> None:
        self.warmup = warmup
        self.sums = {p: 0.0 for p in self.PHASES}
        self.step = 0
        self._t: float | None = None

    def start(self) -> None:
        self._t = time.perf_counter()

    def lap(self, phase: str) -> None:
        if self._t is None:
            raise RuntimeError("StepTimer.start() not called")
        now = time.perf_counter()
        if self.step >= self.warmup:
            self.sums[phase] += now - self._t
        self._t = now

    def end_step(self) -> None:
        self.step += 1

    def summary(self) -> dict[str, Any]:
        active = max(self.step - self.warmup, 0)
        if active <= 0:
            return {"active_steps": 0, "warmup_steps": self.warmup}
        total = sum(self.sums.values())
        components_ms = {p: 1000.0 * self.sums[p] / active for p in self.PHASES}
        components_pct = {
            p: (100.0 * self.sums[p] / total) if total > 0 else 0.0
            for p in self.PHASES
        }
        return {
            "warmup_steps": self.warmup,
            "active_steps": active,
            "per_step_ms": 1000.0 * total / active,
            "components_ms": components_ms,
            "components_pct": components_pct,
        }


@contextmanager
def timed(phase_times: dict[str, list[float]], phase: str):
    """Wall-clock timer, records milliseconds into ``phase_times[phase]``.

    The caller is responsible for bracketing work that has already synchronised
    with the device (e.g. ``int()`` cast on a JAX scalar, or a manual
    ``block_until_ready``) — otherwise the recorded time will miss async work.
    """
    t0 = time.perf_counter()
    yield
    phase_times[phase].append((time.perf_counter() - t0) * 1000.0)
