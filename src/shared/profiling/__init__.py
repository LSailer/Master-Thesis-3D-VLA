"""Shared wall-clock profiling helpers.

The package facade keeps imports stable while implementation lives in focused
modules. These helpers stay dependency-light: no JAX, Torch, Habitat,
R2Dreamer, or VGGT internals are imported at module import time.
"""

from src.shared.profiling.artifacts import write_json
from src.shared.profiling.step_timer import StepTimer
from src.shared.profiling.synthetic import make_synthetic_rgb_frame
from src.shared.profiling.timing import (
    PhaseTimes,
    block_until_ready_tree,
    init_phase_times,
    measure_ms,
    render_phase_table,
    summarize_phase_times,
    summarize_values_ms,
    timed,
)

__all__ = [
    "PhaseTimes",
    "StepTimer",
    "block_until_ready_tree",
    "init_phase_times",
    "make_synthetic_rgb_frame",
    "measure_ms",
    "render_phase_table",
    "summarize_phase_times",
    "summarize_values_ms",
    "timed",
    "write_json",
]
