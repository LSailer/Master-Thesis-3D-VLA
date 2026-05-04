# Master-Thesis-3D-VLA — Domain Glossary

This file is the canonical glossary for the project. Terms are added here
as they're resolved during design discussions (see `.agents/skills/grill-with-docs`).
Architectural decisions live in `docs/adr/`.

## Agent-loop dispatcher

| Term                | Meaning |
| ------------------- | ------- |
| **agent loop**      | The tmux+srun+`claude -p` dispatcher in `scripts/agent_loop.sh`. Serial — one issue at a time, no parallelism, no worktree. Wraps an `srun --gres=gpu:1` shell in a tmux session on the login node so the loop survives SSH disconnects. See `docs/agents/agent-loop.md`. |
| **ready-for-agent** | The *only* label `agent_loop.sh` treats as a green light. An issue is eligible iff it has `ready-for-agent` and lacks `blocked` / `in-progress` / `needs-info`. See ADR 0001. |
| **AFK** / **HITL**  | Orthogonal *execution-mode* labels. Document expected human involvement. Do **not** gate the picker. (See `docs/agents/triage-labels.md`.) |
| **PRD branch**      | `agent/prd-<N>` — single branch where every child commit of PRD #N accumulates across loop iterations. One PR per PRD lands when the last child closes. |
| **orphan issue**    | A `ready-for-agent` issue whose body has no `## Parent #N` reference. Handled on its own `agent/issue-<M>` branch with an immediate PR. |
| **PRD ↔ child link**| The `## Parent\n#N` body reference written by `.agents/skills/to-issues`. The dispatcher uses `gh issue list --search 'in:body "Parent #N"'` to enumerate siblings. |
| **halt**            | Hard-stop the loop on first failure. The failing issue keeps `in-progress` and gets a stderr-tail comment; the tmux pane prints `LOOP HALTED: see #<N>` and exits 1. |
| **drain**           | Loop exits cleanly with code 0 when the picker returns no eligible issues. |
| **implement pass**  | First `claude -p --model claude-opus-4-7 --dangerously-skip-permissions` call per issue. Reads issue body, writes code, runs tests, commits. |
| **review pass**     | Second `claude -p` call per issue. Runs `pr-review-toolkit:review-pr` on the diff vs `main`, commits any blocker fixes, writes `APPROVED` or `BLOCKED: …` to `.agent_loop/review-verdict`. |

## Project domain

| Term                 | Meaning |
| -------------------- | ------- |
| **end-to-end run**   | A single training run of R2Dreamer on the HM3D ObjectNav curriculum with the VGGT (or CNN) encoder. Comprises *acting* (env_step → vggt_forward → wm_inference → buffer_add) and *world-model training* (replay-buffer windows). Profiled in `docs/wiki/methods/l4-profiling.md`. |
| **SLURM run window** | Hard wall-clock cap per slurm job on uc3 = **48 hours**. Jobs that run longer are killed with `TIMEOUT` (verified: job `4109486`, wandb run `4c5xd78l`, ran exactly 2-00:00:28 before cancellation). The thesis training plan splits longer experiments into chained jobs that resume from checkpoint. |
| **chained training run** | A sequence of slurm jobs covering a single end-to-end experiment, each ≤ the SLURM run window, resuming from the previous job's checkpoint. The 4-day full L4-VGGT plot at ~5.2 FPS = ~3.2 M env steps requires ~4 chained jobs at 24h each (or 2 at 48h). |
| **frame-skip (K)**   | Throughput lever filed as issue #97. Run VGGT every K env steps; reuse cached features in between. K=2 ≈ 2× throughput, K=4 ≈ ~4×. Trades feature freshness for steps-per-hour. Orthogonal to the JAX uniformity track (#98). |
| **eviction-onset cliff** | Latency regression in the JAX VGGT extractor: from ~frame 36 onwards every frame triggers a fresh XLA compile (~43 s) because `current_budgets_static` (a static jit arg) drifts after eviction first fires. Diagnosed 2026-05-04 in `docs/wiki/methods/vggt-jax-eviction-recompile.md`. |
