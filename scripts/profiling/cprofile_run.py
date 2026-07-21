#!/usr/bin/env python3
"""Run ``scripts/r2dreamer/run.py`` under cProfile, surviving ``os._exit``.

The trainer hard-exits via ``os._exit(0)`` on successful completion
(``hard_exit_on_finish``) to skip habitat_sim's aborting GL teardown, which
bypasses ``python -m cProfile``'s end-of-run stats dump. This wrapper patches
``os._exit`` so the profile is flushed to ``$PROF_OUT`` immediately before the
real hard exit, and also dumps on the normal return path.

Usage::

    PROF_OUT=output/profiling/foo.prof \
        python scripts/profiling/cprofile_run.py <run-id> [train flags...]
"""
import cProfile
import os
import runpy
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RUN_PY = os.path.join(_REPO_ROOT, "scripts", "r2dreamer", "run.py")


def main():
    """Profiles the run.py training entry point and dumps stats to $PROF_OUT.

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

    # run.py imports its sibling _run_configs; make that resolvable.
    sys.path.insert(0, os.path.dirname(_RUN_PY))
    sys.argv = [_RUN_PY, *sys.argv[1:]]
    profiler.enable()
    try:
        runpy.run_path(_RUN_PY, run_name="__main__")
    finally:
        dump_once()


if __name__ == "__main__":
    main()
