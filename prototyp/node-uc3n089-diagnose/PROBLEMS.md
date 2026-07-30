# Open problems / notes

- `scontrol show node uc3n089` prints nothing for this user on the login
  node - node metadata (drain state, reason) is not visible; judge only by
  job behavior.
- launch.py removed PYTHONFAULTHANDLER=1 from smokes because one of the old
  smoke-only knobs made habitat SIGABRT on good nodes (jobs 6056684/6056813
  vs 6056750). The probe re-enables it deliberately; if the control node's
  combined stage aborts WITH faulthandler but the plain smoke passed, the
  instrumentation itself is a suspect - compare against the control run
  before believing any uc3n089-only conclusion.
- Probing uc3n089 requires it to be free on gpu_h100_short; if PENDING
  forever the node may be drained/busy - check `squeue -w uc3n089`.
