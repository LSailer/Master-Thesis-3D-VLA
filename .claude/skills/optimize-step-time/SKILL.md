---
name: optimize-step-time
description: Interactive optimization loop for training step time (ms/step). Use whenever the user wants training to run faster, asks to optimize/profile step time, names a wandb run to speed up, or says things like "try variants to reduce step time", "why is training slow", "optimize the step cost". The loop measures a baseline from a wandb run, proposes candidate optimizations for the user to select, benchmarks each selected variant on SLURM, and opens a PR summarizing what was tried and measured.
---

# Optimize step time

A human-in-the-loop optimization cycle:

1. **Baseline** — pull the step-time breakdown of a reference wandb run.
2. **Propose** — derive candidate optimizations, notify the user, let them
   select which to try (AskUserQuestion, multiSelect).
3. **Experiment** — implement each selected variant minimally, benchmark it
   on SLURM with a prod-shaped probe, judge by MANIFEST.json.
4. **Report** — open a PR: results table, winners kept, losers documented.

Run the whole loop on one feature branch. Track state in
`.scratch/optimize-step-time/<run-id>.md` (repo issue-tracker convention,
see `docs/agents/issue-tracker.md`) so an interrupted loop can resume.

## Phase 0 — Baseline

Input: a wandb run path like `sailer-luca-university-ulm/3d-vla-objectnav/fvwuoux3`.
If the user gave none, ask which run is the reference before doing anything else.

Fetch the measured step time with the bundled script (login node is fine —
this is wandb API I/O, no GPU):

```bash
.venv/bin/python .claude/skills/optimize-step-time/scripts/fetch_step_time.py \
    <entity>/<project>/<run_id>
```

It prints post-warmup median/p10/p90 of `perf/ms_per_step_interval`, the run
config, and any timing keys in the run summary. If wandb is unreachable or the
run was offline, fall back to the run dir's `output/**/metrics.csv` — note it
is long-format `[step, key, value]` and **not step-sorted**; filter the key,
sort by step, drop the warmup prefix before computing stats.

The wandb ms/step is a whole-step number. To attribute it to components,
check `run.summary` timing keys first, then — only if attribution is genuinely
unclear — launch a profiling probe (`scripts/slurm/launch.sh
profile_training_vggt --smoke`, see `scripts/profiling/README.md`). Known
reference point: the house-points-pose production shape measured ~219 ms/step
≈ VGGT encoder 132 ms + `ReplayBuffer.sample` 59 ms amortized + rest. Re-derive
rather than assume — the codebase moves.

## Phase 1 — Propose variants and let the user pick

From the breakdown, write 3–6 candidate optimizations. Each candidate needs:

- **Name** — short slug, becomes the config/commit name.
- **Hypothesis** — which component it attacks and why it should help.
- **Expected saving** — rough ms/step, tied to the measured breakdown.
- **Risk** — what could regress (loss quality, memory, correctness).
- **Cost to test** — smoke-only vs. prod-shaped probe.

Candidates the user explicitly named in their request always go on the list.
Then hand the decision to the user:

1. Load `PushNotification` via ToolSearch and send a short notification that
   options are ready ("Step-time optimization: N candidates ready to pick").
2. Call `AskUserQuestion` with `multiSelect: true`, one option per candidate
   (label = name, description = hypothesis + expected saving + risk).

Do not start implementing before the user has selected. This gate is the
point of the skill — the user decides what GPU time gets spent on.

## Phase 2 — Try each selected variant

For each selected candidate, in sequence:

1. **Implement the minimal diff.** One candidate = one commit. Prefer a
   config-gated change (new flag defaulting to old behavior) over an
   unconditional rewrite — losers are then trivially revertible and winners
   can ship dark. Follow repo preferences (JAX over NumPy, bfloat16 defaults,
   Google-style docstrings on touched functions only).
2. **Create a probe config** `scripts/slurm/configs/<name>_probe.yaml`
   modeled on the existing `*_prodshape_probe_*.yaml` files: production
   shapes, short walltime. Smoke shapes understate real step cost (a past
   variant read 158 ms in smoke vs 219 ms at prod shape), so measure at prod
   shape whenever the change touches anything shape-dependent.
3. **Launch:** `bash scripts/slurm/launch.sh <name>_probe --time 01:00:00`
   (or `--smoke` for a cheap correctness gate first). Poll with a background
   Bash loop on `squeue -h -j <jobid>`; queue waits can be long, so keep
   working on the next candidate's implementation while a probe runs.
4. **Judge by MANIFEST.json, not the SLURM exit code** — habitat's GL
   teardown can poison exit codes after a fully successful run. The run dir's
   `MANIFEST.json` gains `"status": "completed"` on success. No manifest end
   or a non-completed status = failed probe; read the SLURM log for the
   traceback.
5. **Measure:** post-warmup median `perf/ms_per_step_interval` from the
   probe's metrics (same fetch script if the probe ran wandb online,
   otherwise `metrics.csv`). Also record the loss curves' sanity — a variant
   that is fast but diverges is a loser.
6. **Record** the result in the `.scratch/optimize-step-time/<run-id>.md`
   file immediately: name, commit SHA, job id, ms/step, Δ vs baseline,
   verdict (keep / revert / needs-longer-run), one-line note.

If a probe fails for a fixable reason (OOM, config typo), fix and relaunch
once; a second failure marks the candidate `failed` with the reason — don't
sink the loop into one stubborn variant.

## Phase 3 — PR

When all selected candidates have a verdict:

1. Revert or default-off the losers (keep their probe configs and the issue
   file — negative results are findings too).
2. Commit remaining work; push the branch; open a PR against `main` with
   `gh pr create`. PR body template:

```markdown
## Step-time optimization: <baseline run id>

Baseline: <ms/step> ms/step (median, post-warmup) — <wandb run link>

| Variant | Hypothesis | ms/step | Δ | Verdict |
|---|---|---|---|---|
| <name> | <one line> | <n> | <-x ms / +x ms> | kept / reverted / failed |

## Details
<per-variant: what was changed, probe job id, wandb link, caveats>

## Not tried
<candidates the user deselected, for the record>
```

3. Notify the user (PushNotification) with the PR URL and the headline
   number: best variant's ms/step vs baseline.

## Ground rules

- Never run VGGT/Habitat/JAX-GPU work on the login node — everything through
  `scripts/slurm/launch.sh` (canonical launcher; ignore legacy
  `scripts/r2dreamer/slurm/*.sbatch`).
- One wandb run of GPU time per candidate probe; don't launch prod-length
  (2M-step) runs from this skill — recommend one in the PR instead if a
  winner deserves final validation.
- Compare like with like: probe vs. probe at identical shapes, never probe
  vs. the baseline's full prod run without noting the shape difference.
