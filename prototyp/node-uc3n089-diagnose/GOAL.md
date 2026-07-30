# Goal: why does uc3n089 kill habitat GL jobs with exit 134?

## Symptom

Since 2026-07-20, habitat training jobs on node uc3n089 die with exit 134
(SIGABRT) mid-prefill. Confirmed 2026-07-30: all four CNN smoke jobs
(6098260-6098263) landed on uc3n089 and died after ~8 min; the identical
resubmit (6098413-6098416) on uc3n088 passed in ~6 min. The node is
blocklisted via `sbatch.exclude` in `scripts/slurm/configs/_base.yaml`
until this is understood.

## Evidence so far

- Same GPU on both nodes: H100 NVL, 95830 MiB total.
- Failed jobs died at ~72.4 GB used = exactly XLA's default 75% preallocation.
  The good node peaks at ~75.2 GB and survives, so it is not a capacity gap.
- L1-L3 died after JAX allocation; L4 (6098263) died with 27 MiB used,
  i.e. before JAX ever allocated - two different death points.
- Dead jobs ran LONGER than a full green smoke (8 min vs 6 min): the node
  first slows down, then the job aborts.
- No abort message in stdout or stderr: the last stderr line is a harmless
  hwloc warning. The SIGABRT is silent, hence this staged probe.

## Hypotheses

- H1 GL/EGL driver state: habitat's GL context or sensor reads abort on this
  node regardless of JAX (test: stage habitat-cpu).
- H2 JAX/XLA on this GPU: allocation or compute fails alone (test: stage jax-gpu).
- H3 Interaction: only the prod ordering (XLA prealloc ~72 GB, then habitat
  GL needs device memory alongside) fails, e.g. degraded/remapped memory rows
  that only the upper region touches (test: stage combined; check ROW_REMAPPER).
- H4 User memory: "last time this was about JAX allocation needing 80 GB not
  70 GB" - watch total/used and XLA_PYTHON_CLIENT_MEM_FRACTION effects.

## Approach

`probe.sbatch` runs four stages on one pinned node (`submit.sh <node>`),
each as its own python process with PYTHONFAULTHANDLER=1 so a SIGABRT
finally yields a stack. A stage failing does not stop the next:

- env: nvidia-smi -q dump (ECC, ROW_REMAPPER, retired pages), driver, clocks.
- habitat-cpu: JAX_PLATFORMS=cpu run of src.main, prefill 300 - GL path only.
- jax-gpu: allocate + matmul at default prealloc, no habitat.
- combined: prod-ordering src.main smoke, prefill 2048, prod shape - the
  actual reproduction.

Run on uc3n089 and on a known-good control (uc3n088), diff the outputs under
`outputs/prototype/node-uc3n089-diagnose/`.
