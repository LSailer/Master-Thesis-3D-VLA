#!/usr/bin/env python3
"""Run ``python -m src.main`` under cProfile, surviving ``os._exit``.

The trainer hard-exits via ``os._exit(0)`` on successful completion
(``hard_exit_on_finish``) to skip habitat_sim's aborting GL teardown, which
bypasses ``python -m cProfile``'s end-of-run stats dump. This wrapper patches
``os._exit`` so the profile is flushed to ``$PROF_OUT`` immediately before the
real hard exit, and also dumps on the normal return path.

Usage::

    PROF_OUT=output/profiling/foo.prof \
        python scripts/profiling/cprofile_run.py [train flags...]
"""
import cProfile
import os
import runpy
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Profiles the src.main entry point and dumps stats to $PROF_OUT.

    Returns:
      None. Exits with the training process's exit semantics; the profile is
      written to the path in the PROF_OUT environment variable.

    Raises:
      KeyError: If the PROF_OUT environment variable is not set.
    """
    prof_out = os.environ["PROF_OUT"]
    profiler = cProfile.Profile()
    dumped = False

    def dump_once():
        nonlocal dumped
        if not dumped:
            dumped = True
            profiler.disable()
            profiler.dump_stats(prof_out)
            print(f"[cprofile_run] profile written to {prof_out}", flush=True)

    real_exit = os._exit

    def exit_with_dump(code):
        dump_once()
        real_exit(code)

    os._exit = exit_with_dump

    # Module mode resolves src against the repo the profiler runs from.
    sys.path.insert(0, _REPO_ROOT)
    sys.argv = ["src.main", *sys.argv[1:]]
    profiler.enable()
    try:
        runpy.run_module("src.main", run_name="__main__")
    finally:
        dump_once()


if __name__ == "__main__":
    main()
