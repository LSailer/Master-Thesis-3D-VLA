---
name: engineer-team
description: Phase 2 of the auto-pipeline. Reads a /grill-me recap, orchestrates Sonnet sub-agents to implement the work in packages, runs an integration smoke test, and submits the SLURM verify+report chain via /slurm-submit. Use after a /grill-me session that produced a recap with eval criteria.
---

# Engineer Team

## Quick start

> **User:** `/engineer-team docs/wiki/recaps/2026-04-26-auto-pipeline.md`
> **You:** "Recap read. Proposing 3 packages: (P1) /grill-me + /engineer-team SKILL.md updates, (P2) scripts/pipeline/ bash glue, (P3) verify+report prompt templates. Run P2 and P3 in parallel after P1, since they depend on naming decisions in P1. Confirm split?"

You are Opus. Sub-agents are Sonnet via the `Task` tool. Three hard gates: package split, smoke pass, long-run go.

## Workflow

### Setup

- [ ] Read the recap path provided (or find it under `docs/wiki/recaps/`)
- [ ] Check GPU: `nvidia-smi` — if missing, prefix smoke commands with `srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00`
- [ ] Read related `docs/wiki/methods/` and `docs/wiki/experiments/` for context
- [ ] Skim `pyproject.toml` and `modules/` for existing patterns and missing dependencies
- [ ] Derive slug `<name>` from recap title; `git checkout -b pipeline/<name>` (next free `-vN` suffix if branch exists)

### Phase 1 — Package proposal (Gate 1)

- [ ] Decompose recap deliverables into 2–5 logical packages (one per module touched; tests separate only if they would dwarf the code)
- [ ] Identify dependencies between packages and group parallel-able ones
- [ ] Present split to user
- [ ] **Wait for user confirmation before dispatching**

### Phase 2 — Sub-agent dispatch

For each confirmed package, invoke `Task` with `subagent_type=general-purpose` (or `feature-dev:code-architect` for design-heavy work). Run independent packages in parallel (multiple `Task` calls in one message); sequential ones one at a time.

Sub-agent prompt MUST include:

- [ ] Package scope: files allowed, files explicitly NOT allowed
- [ ] Relevant excerpts from the recap (extract — never pass the full recap)
- [ ] The eval criteria from Phase B that this package supports
- [ ] **"Test only your block. Do not run end-to-end smoke. Commit with a clear message."**
- [ ] **"If unclear, stop and report — do not improvise."**

After each sub-agent completes:

- [ ] `git status` + `git diff HEAD~1` to verify whitelist compliance
- [ ] If files outside whitelist were touched: revert and re-dispatch

### Q&A relay (whenever a sub-agent reports a clarification request)

- [ ] Surface the question verbatim to the user
- [ ] Wait for the user's answer
- [ ] Append the Q&A pair to a "Lessons Learned" section of the recap (or experiment MD if one exists)
- [ ] Re-dispatch the sub-agent with the answer appended

### Phase 3 — Args file

- [ ] Write `scripts/pipeline/<name>.args` per [ARGS-FORMAT.md](ARGS-FORMAT.md) — only experiment-specific overrides; reuse the generic `scripts/slurm/train.sbatch`

### Phase 4 — Integration smoke (Gate 2)

- [ ] Run end-to-end smoke (1–5 min) using the args from `<name>.args` with a tiny step budget
- [ ] Verify metrics emit at the expected path; `sbatch --test-only` for sbatch syntax
- [ ] If smoke fails: fix in this session — never submit a long run on broken code
- [ ] On smoke green: ask user "Smoke passed. Go for long run?" and **wait for confirmation**

### Phase 5 — Submit (Gate 3 = user said "go")

- [ ] Invoke `/slurm-submit <name>` (skill call — wraps `scripts/pipeline/launch.sh`)
- [ ] Print branch name, three SLURM job IDs (train, verify, report), recap path
- [ ] Remind user: "On failure, verify creates a gh issue and report is skipped (afterok dependency)."

Do **not** update `docs/wiki/index.md` here — that is the reporter's job after the long run completes successfully.
