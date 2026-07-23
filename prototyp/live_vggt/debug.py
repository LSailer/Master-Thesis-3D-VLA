"""Reusable debug printing for manual prototypes."""

from __future__ import annotations

from os import PathLike
from pathlib import Path


class Debugger:
    """Flushed debug printer with a file-name prefix."""

    def __init__(self, file_name: str | PathLike[str], active: bool = True) -> None:
        self.file_name = Path(file_name).stem
        self.active = active

    def debug(self, message: str) -> None:
        """Print a debug message when this debugger is active."""

        if self.active:
            print(f"[{self.file_name}] {message}", flush=True)

    def __call__(self, message: str) -> None:
        """Print a debug message when this debugger is active."""

        self.debug(message)

    def set_active(self, active: bool) -> None:
        """Enable or disable debug printing."""

        self.active = active
