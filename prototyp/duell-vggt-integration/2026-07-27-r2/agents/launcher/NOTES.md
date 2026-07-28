Launcher babysitting notes - Duell 2 r2

Protocol: 30-min --prod runs, expected end = TIMEOUT (success). Judge by
MANIFEST.json, not exit code (habitat GL teardown poisons exit codes).
metrics.csv is long-format (step,metric,value), not sorted - sort/filter per
metric before reading.

## Wave 1 (submitted ~13:21 UTC, all resolved by ~13:33 UTC)

| job | slot | config | state | exit | elapsed | node | max step | episodes | last SR |
|---|---|---|---|---|---|---|---|---|---|
| 6060402 | A | duell_l3_aggpool_lottery | TIMEOUT | 0:0 | 00:30:12 | uc3n105 | 19175 | 39 | 0.0256 |
| 6060403 | B | duell2_l3_aggpool_b200k_p2048 | FAILED | 2:0 | 00:00:31 | uc3n105 | never started | - | - |
| 6060404 | C | duell2_l3_aggpool_b200k_tr128 | TIMEOUT | 0:0 | 00:30:12 | uc3n105 | 21543 | 44 | 0.0227 |
| 6060405 | D | duell2_l3_pointmap_p2048 | FAILED | 134:0 | 00:10:04 | uc3n107 | 0 (no steps logged) | - | - |

### 6060403 (Slot B) FAILED early - uv-sync race, NOT habitat GL, NOT exit 134

Died in 31s, before python/training started - no run-6060403 dir was ever
created (no MANIFEST.json, no metrics.csv). .err log:

```
   Building master-thesis-3d-vla @ file:///.../duell-2-vggt-dreamer-1c1f56
      Built master-thesis-3d-vla @ file:///.../duell-2-vggt-dreamer-1c1f56
error: failed to remove directory `.../.venv/lib/python3.11/site-packages/flax/nnx/training`: No such file or directory (os error 2)
```
sacct: State FAILED, ExitCode 2:0, DerivedExitCode 0:0.

Root cause: scripts/slurm/launch.py only forces `uv run --no-sync python` for
--smoke mode (line 191-192); prod mode uses plain `uv run python`, which does
an implicit `uv sync` against the SHARED .venv. Wave 1's 4 jobs were
submitted near-simultaneously and 3 landed on uc3n105 within ~1 minute of
each other; job B's uv sync raced a concurrent sync/rebuild from another job
touching the same shared .venv tree and lost (tried to remove a directory
another process had already removed). Matches known memory item "uv run
re-syncs shared venv". Launcher/infra bug, not code or habitat GL fault.

### 6060405 (Slot D) FAILED exit 134 - habitat GL SIGABRT, before any training step

Ran 10:04, only got through habitat scene/GL setup (wandb run started
q8uoi1fv) before dying - no traceback in .err, just habitat GL init logs
followed by the SLURM job-feedback footer showing "State: FAILED (exit code
134)". No steps ever logged (metrics.csv is 0 bytes). Node was uc3n107 (not
previously known-bad uc3n089/uc3n105 in memory - new node to watch).
Classic habitat GL teardown/init SIGABRT signature.

### 6060402 (A) and 6060404 (C) - healthy, TIMEOUT = success

Both ran the full 30:12 walltime on uc3n105, MANIFEST.json has started_at,
metrics.csv growing throughout. Final (sorted) numbers:
- A (aggpool-lottery): step 19175, episode/count 39, metrics/sr 0.0256
- C (aggpool-b200k-tr128): step 21543, episode/count 44, metrics/sr 0.0227

Artifacts copied to
prototyp/duell-vggt-integration/2026-07-27-r2/runs/{6060402-aggpool-lottery,
6060403-aggpool-b200k-p2048,6060404-aggpool-b200k-tr128,6060405-pointmap-p2048}/
(MANIFEST.json + metrics.csv + slurm .out/.err; B and D have no
metrics.csv/MANIFEST since they never trained).

## Wave 2 / final wave (submitted ~14:12 UTC by orchestrator, expected end ~14:45 UTC)

- 6061173 = Slot E, duell2_l3_aggpool_b200k_tr128 SEED=43 (confirmation run of C)
- 6061174 = Slot F, duell2_l3_b200k_tr128_ent3em4
- 6061175 = Slot G, duell2_l3_b200k_tr128_ent3em3
- 6061176 = Slot H, duell2_l3_pointmap_p2048 retry (of D)

### 14:13 UTC check-in

| job | slot | state | node |
|---|---|---|---|
| 6061173 | E | RUNNING (started 16:13:07, elapsed ~2min) | uc3n088 |
| 6061174 | F | RUNNING (started 16:13:07, elapsed ~2min) | uc3n088 |
| 6061175 | G | PENDING (Resources) | - |
| 6061176 | H | PENDING (Priority) | - |

E and F share node uc3n088 - watching closely for the same uv-sync race as B
hit in wave 1 (both started at the exact same timestamp). Will check their
MANIFEST.json next poll to confirm they actually reached training.

Continuing to poll every few minutes until all 4 final-wave jobs end.

### 14:27 UTC check-in

All four final-wave jobs now RUNNING:
- E (6061173) uc3n088, started 14:16:12 UTC
- F (6061174) uc3n088, started 14:16:13 UTC
- G (6061175) uc3n104, started 14:26:08 UTC
- H (6061176) uc3n104, started 14:26:31 UTC

G and H share uc3n104 and started ~23s apart - checked MANIFEST.json for
both, both have started_at set (no uv-sync race repeat). All 4 confirmed
past the venv-sync stage and into training. Expected TIMEOUT for E/F around
14:46-14:47 UTC (30:xx min after their start), G/H around 14:56-14:57 UTC.
