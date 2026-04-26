# Recap: Auto-Pipeline Design

**Date:** 2026-04-26
**Source:** /grill-me session
**Topic:** Automated experiment pipeline (Plan → Engineer → Test → Review)
**Outcome:** Two-skill design + bash glue, no new orchestrator skill needed

## Kontext

User wants an automated pipeline that walks a research idea through four phases — Planning, Developing, Test, Review — with minimal human babysitting. Goal: hand over an idea, get back a PR + HTML report (or a GitHub issue if verification fails). Triggered by frustration with manually re-typing the same `/plan` → `/engineer` → `sbatch` → `/reporter` sequence per experiment, and a hand-drawn pipeline diagram the user shared during the session, introducing a "Knowledge + HTML Page" artifact spanning all phases.

Pre-existing assets that constrain the design:
- `/grill-me`, `/engineer`, `/review`, `/reporter`, `/plan`, `/loop`, `/schedule` skills already exist
- `docs/wiki/{methods,experiments,meetings,research}/` is the established knowledge base
- `modules/vggt/autoresearch/run_autoresearch.sbatch` already calls `claude -p --dangerously-skip-permissions` from a SLURM compute node (proves outbound API works on cluster)
- `scripts/slurm/train.sbatch` exists as a reusable template
- Phased Orchestration Pattern (`docs/wiki/methods/phase-orchestration.md`) already codifies the orchestrator+sub-agents recipe

## Design Decisions

### D1 — Architecture form: NO new meta-skill, instead enhance two existing skills + bash glue
**Decision:** No `/auto-pipeline` skill. Enhance `/grill-me` (Phase 1) and create new `/engineer-team` (Phase 2). Phases 3+4 are bash/sbatch with `claude -p` calls.
**Why:** Skills are for interactive work, bash is for unattended. The "automation" between phases is just shell glue. Building a meta-skill would be ceremony for no gain.
**Rejected alternatives:** (A) single `/auto-pipeline` meta-skill that orchestrates everything in one Claude session — would block terminal for hours, can't survive cluster long-runs. (B) Wrapper script outside Claude that spawns `claude -p` per phase — over-engineered. (C) Cron-driven state machine — premature for one user.

### D2 — Phase 1+2 share a Claude-Opus session, Phase 3+4 run autonomously on SLURM
**Decision:** Unit A = `/grill-me` + `/engineer-team` in one live Opus session with the user. Unit B = SLURM dependency chain (train → verify → report) running unattended.
**Why:** Long-runs take hours/days. Claude session can't stay open. Splitting at the smoke-test/SLURM-submit boundary lets the user disconnect after Phase 2 and return to a finished PR.
**How to apply:** Phase 2 ends with `bash scripts/pipeline/launch.sh <name>` then Claude exits.

### D3 — Knowledge artifact lives in `docs/wiki/`, NOT as GitHub issues
**Decision:** Plan-phase output is a markdown file in the wiki, not a GH issue. PRD-as-issue (the default in `/plan`) is dropped for the pipeline workflow.
**Why:** User wants a single artifact that survives across phases and can be re-read by every phase. Wiki is already that store. GH issues are for failure-tracking, not specs.
**How to apply:** `/grill-me` writes `docs/wiki/recaps/<date>-<topic>.md`; `/engineer-team` may also write `docs/wiki/experiments/<name>.md` if the work is an experiment.

### D4 — Three-layer wiki structure: recaps / methods+experiments / log+index
**Decision:** New `docs/wiki/recaps/` folder for grill-session audit trails. Existing `methods/` and `experiments/` keep destillated end-products. `log.md` and `index.md` index both layers.
**Why:** Recaps preserve the *journey* (what was discussed, what was rejected, why); methods/experiments preserve the *destination*. Separating them lets re-grills on "No" find the original reasoning without rummaging through long markdowns.
**How to apply:** `/grill-me` writes recap + appends one-liner to `log.md`. `/grill-me` does NOT touch `index.md` (that's reserved for the destillated end-product).

### D5 — Verifier and Reporter are `claude -p` calls, not Python scripts
**Decision:** No deterministic Python verify layer. `verify.sbatch` calls `claude -p --model claude-sonnet-4-6 ...` with a prompt template. Same for `report.sbatch`.
**Why:** The autoresearch sbatch proves this pattern works on the cluster. Sonnet handles structured judgment + writing (gh issue body, PR body) in one tool. A Python pre-filter would be a second vocabulary to maintain for marginal cost savings (~$0.05/run).
**Trade-off:** Non-determinism + per-run token cost. Acceptable given infrequency of pipeline runs.
**Escalation rule (parked):** If Sonnet's verify is too lenient/strict in practice, add an Opus second-opinion pass on uncertain cases. Not building this now.

### D6 — Sub-agents are Sonnet via Task tool; Opus orchestrates
**Decision:** `/engineer-team` runs as Opus, dispatches per-package work to Sonnet sub-agents via the Task tool. Sub-agent ↔ user Q&A is relayed through Opus.
**Why:** Cost-efficient (Sonnet ~5× cheaper than Opus for code generation per `methods/phase-orchestration.md`), and Sub-agents protect the orchestrator's context window.
**How to apply:** Each package = one `Task` invocation with subagent_type=general-purpose (or feature-dev:code-architect for design-heavy packages).

### D7 — Sub-agent test scope = block-level only; Orchestrator does the integration smoke
**Decision:** Each Sonnet sub-agent writes + runs unit/component tests for its block. End-to-end pipeline smoke-test is the orchestrator's job before SLURM submit.
**Why:** Standard division. Sub-agents shouldn't see the full pipeline (context bloat); orchestrator already has it.
**How to apply:** Sub-agent prompts include "test only your block, do not run end-to-end". Orchestrator runs smoke after all sub-agents are done.

### D8 — Sub-agent Q&A gets logged into the experiment MD's "Lessons Learned" section
**Decision:** When a sub-agent asks Opus, who relays to user — the question + answer is appended to a "Lessons Learned" section of the experiment MD (or recap).
**Why:** Future re-grills (on "No") see what was unclear last time. Pipeline becomes lesson-accumulating over runs.
**How to apply:** `/engineer-team` keeps an in-memory list of Q&A pairs, dumps them at end of session into the MD.

### D9 — `/grill-me` becomes universal (no pipeline-mode), but always probes evaluation
**Decision:** Every `/grill-me` session — not just pipeline-bound — runs a two-phase structure: (Phase A) design grilling, (Phase B) eval pass that walks each decision and asks "auto / manual / N/A?".
**Why:** The act of asking "how do we evaluate this?" forces design rigor universally, not only when there's a pipeline downstream. A pipeline-only flag would be a missed leverage opportunity.
**Rejected alternative:** Per-decision inline eval-question — too noisy, doesn't allow holistic eval-design once full design is known.

### D10 — Three mid-session human gates in the live Opus session
**Decision:** User confirms three times during Unit A before SLURM submit:
1. After Phase 1 grilling: "Recap MD looks right?"
2. After Phase 2 sub-agent package proposal: "Package split looks right?"
3. After smoke test: "Go for long run?"
**Why:** Each gate corresponds to a hard-to-reverse next step (writing code, submitting expensive SLURM jobs).

### D11 — Branch strategy: `/engineer-team` auto-creates `pipeline/<name>` branch
**Decision:** Engineer-team checks out a new branch `pipeline/<name>` automatically. PR targets `main`. Re-grill after "No" → new branch `pipeline/<name>-v2`.
**Why:** Prevents accidentally committing pipeline output to `main` or to whatever feature-branch was current.

### D12 — Reporter handles both HTML rendering AND `gh pr create`
**Decision:** No separate `open_pr.sh`. `report.sbatch`'s claude call does HTML + PR opening in one shot.
**Why:** Both need the same input (the experiment MD + verify results). Splitting them would duplicate file-reading.

### D13 — Verifier handles both pass-write-results AND fail-create-issue
**Decision:** No separate `file_issue.sh`. `verify.sbatch`'s claude call writes results to MD frontmatter on pass OR runs `gh issue create` on fail.
**Why:** Same logic, same input, one place.

### D14 — SLURM chain uses `--dependency=afterok` to gate Reporter on Verify success
**Decision:** `verify.sbatch` exits non-zero on fail → `report.sbatch` (with `--dependency=afterok:verify_jobid`) never runs. Verify itself creates the gh issue. No `--dependency=afternotok` job needed.
**Why:** Cleaner than two parallel dependency chains; aligns with SLURM convention.

### D15 — Models: Sonnet 4.6 for both Verify and Report, Opus 4.7 for orchestration
**Decision:** `claude -p --model claude-sonnet-4-6` for `verify.sbatch` and `report.sbatch`. Live Opus session for `/grill-me` and `/engineer-team` orchestration. Sub-agents use Sonnet via Task tool.
**Why:** Verify is structured comparison; Report is templated rendering + prose — both Sonnet's comfort zone. Per-run cost ~$0.10. Opus for the live orchestration where reasoning depth matters.
**Parked:** "Opus second-opinion on verify-uncertainty" rule — add later if observed need.

### D16 — Existing `train.sbatch` is reusable; `.args` files per experiment
**Decision:** `scripts/slurm/train.sbatch` (or its successor) stays generic. `/engineer-team` writes `scripts/pipeline/<name>.args` with hyperparameters. `launch.sh` reads the args file and passes them to sbatch.
**Why:** 95% of sbatch lines are identical across experiments. Per-experiment data is hyperparams.
**How to apply:** Sub-agents do not generate sbatch files from scratch — they only write `.args`.

## Eval-Pass

| Decision | Eval mode | Notes |
|---|---|---|
| D1 (no meta-skill) | Manual | Felt right after design discussion; success = "we don't write a meta-skill SKILL.md file" |
| D2 (Phase split) | Auto | Success = launch.sh submits 3 jobs and exits cleanly; checked by smoke run |
| D3 (wiki not issues) | Manual | Success = no `gh issue create` in `/grill-me` skill; visible in skill diff |
| D4 (recaps folder) | Auto | Success = `docs/wiki/recaps/` exists, this file is written, log.md has new entry |
| D5 (claude -p verifier) | Auto | Success = `verify.sbatch` invokes `claude -p` not `python verify.py`; visible in file |
| D6 (Sonnet sub-agents) | Manual | Success = `/engineer-team` SKILL.md mentions Task tool with Sonnet model |
| D7 (test scope split) | Manual | Verified by reading `/engineer-team` skill's sub-agent prompt template |
| D8 (Q&A logged to MD) | Auto | Success = `/engineer-team` writes a "Lessons Learned" section when applicable |
| D9 (universal eval-pass in grill) | Manual | Verified by reading new `/grill-me` skill — Phase B section exists |
| D10 (three gates) | Manual | Verified in `/engineer-team` skill — three explicit confirmation prompts |
| D11 (branch auto-create) | Auto | Success = engineer-team's first action is `git checkout -b pipeline/<name>` |
| D12 (reporter does PR) | Auto | Success = `report_prompt.md` includes `gh pr create` instruction |
| D13 (verifier files issue) | Auto | Success = `verify_prompt.md` includes `gh issue create` instruction |
| D14 (afterok chain) | Auto | Success = `launch.sh` uses `--dependency=afterok` |
| D15 (Sonnet for verify/report) | Auto | Success = sbatch files contain `--model claude-sonnet-4-6` |
| D16 (.args per experiment) | Auto | Success = `launch.sh` reads from `scripts/pipeline/<name>.args` |

## Open Questions / Parked

- **OQ1:** Re-grill on "No" UX — how does the user signal "re-grill this experiment"? `/grill-me --replan <name>`? Manual restart? Defer until first real "No" event.
- **OQ2:** Multi-run pipelines — does `/auto-pipeline` make sense for sweeps (different seeds)? Current design assumes one-experiment-per-recap. Defer.
- **OQ3:** Opus second-opinion on verify-uncertainty — D15 parked rule. Implement only when observed need.
- **OQ4:** What does the criteria YAML schema actually look like? Frontmatter vs sidecar vs fenced block. Engineer-team will decide on first use; we'll codify the convention into `/grill-me` after one round of usage.

## Deliverables (next steps)

1. Update `.claude/skills/grill-me/SKILL.md` — add wiki-first read, web-search-on-uncertainty, two-phase structure, recap-write at end
2. Create `.claude/skills/engineer-team/SKILL.md` — orchestration + sub-agents + branch + smoke + launch
3. Create `scripts/pipeline/launch.sh` — bash glue submitting the 3-job chain
4. Create `scripts/pipeline/{verify,report}.sbatch` — sbatch wrappers around `claude -p`
5. Create `scripts/pipeline/{verify,report}_prompt.md` — prompt templates
