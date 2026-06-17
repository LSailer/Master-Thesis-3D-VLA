"""Synthetic inputs for profiling and benchmark scripts."""

from __future__ import annotations


def make_synthetic_rgb_frame(seed: int, size: int = 518):
    """Return a deterministic ``(3, size, size)`` uint8 RGB frame."""
    import numpy as np

    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(3, size, size), dtype=np.uint8)
