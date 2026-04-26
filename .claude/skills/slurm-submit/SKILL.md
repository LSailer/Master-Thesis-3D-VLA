---
name: slurm-submit
description: Submit the auto-pipeline SLURM job chain (train → verify → report, chained via --dependency=afterok) for an experiment whose .args file lives at scripts/pipeline/<name>.args. Use when the user types /slurm-submit <name>, when /engineer-team finishes Phase 5, or when the user wants to launch a pipeline run for an experiment that already has an .args file. For one-off sbatch jobs unrelated to the pipeline, use raw sbatch commands instead.
---

# SLURM Submit

## Quick start

> **User:** `/slurm-submit cnn-l2-baseline`
> **You:** "Reading scripts/pipeline/cnn-l2-baseline.args. Branch is pipeline/cnn-l2-baseline. Submitting chain via launch.sh:
>  - train → JOB 5012891 (gpu_h100, 36:00:00)
>  - verify → JOB 5012892 (afterok:5012891, claude-p Sonnet)
>  - report → JOB 5012893 (afterok:5012892, claude-p Sonnet)
> Monitor with `squeue -u $USER`. PR will open when verify passes; gh issue if it fails."

This skill is a thin wrapper around `scripts/pipeline/launch.sh`. The bash script is the source of truth for submission logic; this skill exists so `/engineer-team` and the user can invoke the chain through the skill mechanism rather than memorising the path.

## Workflow

### Validate

- [ ] Confirm `scripts/pipeline/<name>.args` exists — if not, fail with: "No args file at scripts/pipeline/<name>.args. /engineer-team writes this in Phase 3."
- [ ] Confirm current branch is `pipeline/<name>` (or `pipeline/<name>-vN`). If not, warn: "Branch is `<branch>`. The chain will tag this branch in commits. Continue anyway? (Y/N)"
- [ ] Confirm `scripts/pipeline/launch.sh` exists and is executable — if not, fail: "Pipeline scaffolding missing. Re-run /engineer-team."

### Submit

- [ ] Run `bash scripts/pipeline/launch.sh <name>` from the repo root
- [ ] Capture the three SLURM job IDs from launch.sh's stdout
- [ ] Confirm `launch.sh` exports/propagates `WANDB_RUN_ID` (and forwards `SLURM_JOB_ID`, which SLURM sets automatically) into the train job's environment via `--export=ALL,WANDB_RUN_ID=$WANDB_RUN_ID,...` or an equivalent `sbatch --export` clause. The trainer's `MANIFEST.json` emission reads both env vars at run start; if either is missing, the manifest will record `unknown` and lose provenance. If `launch.sh` does not currently propagate `WANDB_RUN_ID`, fail with: "launch.sh missing WANDB_RUN_ID propagation — fix before submitting (see docs/wiki/recaps/2026-04-26-output-restructure.md decisions #9, #10)."

### Report

- [ ] Print: branch name, three job IDs (train / verify / report) with their `afterok` dependency, monitoring command, and where the deliverables will appear:
  - On success → PR opened by report job + HTML at `output/reports/<name>.html`
  - On failure (verify exit non-zero) → GitHub issue created by verify; report job is skipped (afterok-blocked)
- [ ] Exit. The user can disconnect; the chain runs autonomously.

## Hard rules

- This skill **submits direct sbatch with --dependency=afterok**. It does NOT trigger GitHub Actions workflows.
- This skill is **not for ad-hoc sbatch jobs** — for one-off `sbatch foo.sbatch` calls, just run them directly.
- This skill **does not write the .args file** — that's `/engineer-team` Phase 3. If args don't exist, the user is in the wrong place in the workflow.
- This skill **does not run on a compute node** — it's an orchestrator step, intended to run on a login node where you have outbound network and `gh` CLI for the eventual PR.
- The submitted train job **must inherit `WANDB_RUN_ID`** from the calling shell so the trainer's auto-emitted `MANIFEST.json` (per recap 2026-04-26-output-restructure decisions #9 and #10) records a real wandb id, not `unknown`. SLURM auto-injects `SLURM_JOB_ID` into the job; `WANDB_RUN_ID` must be explicitly forwarded via `sbatch --export`. `launch.sh` is the source of truth for the actual export clause — this skill only verifies it is present.
