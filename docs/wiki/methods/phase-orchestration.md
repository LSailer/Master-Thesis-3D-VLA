# Phased Orchestration Pattern

**Date**: 2026-04-25
**Origin**: `#85` launcher refactor session — pattern emerged organically, codified after the fact.
**Related**: [methods/launcher-refactor.md](launcher-refactor.md) (the design doc this pattern delivered), [methods/workflow.md](workflow.md) (the broader pipeline)

## When to use this pattern

Use **phased orchestration** instead of plain `/engineer` when **all** of the following hold:

1. The work has **2+ independent phases** with clear commit-able boundaries.
2. A **design document already exists** (typically a wiki methods page produced by `/grill-me`) — phases and exit gates are written down, not improvised.
3. The user **explicitly opts in** for orchestrator-with-subagents (per `feedback_skills_over_agents.md`'s exception clause). Default remains `/engineer` for single-shot work.
4. Cost matters — Sonnet subagents implementing under Opus orchestration is ~3-5× cheaper than pure-Opus implementation while preserving review quality.

If any of those fail, use `/engineer` instead.

## The pattern at a glance

```
┌──────────────────────────┐
│  Wiki design doc (spec)  │   ← /grill-me produced this earlier
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Orchestrator (Opus)     │   ← user-facing, in main conversation
│  - reads spec            │
│  - drafts phase briefs   │
│  - spawns subagents      │
│  - runs verify gates     │
│  - proposes commits      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│ Phase 1 subagent         │ →  │ Phase 2 subagent         │ →  │ Phase 3 subagent         │
│ (Sonnet, run_in_bg=true) │    │ (Sonnet, run_in_bg=true) │    │ (Sonnet, run_in_bg=true) │
│ Strict file whitelist    │    │ Strict file whitelist    │    │ Strict file whitelist    │
└────────────┬─────────────┘    └────────────┬─────────────┘    └────────────┬─────────────┘
             │                               │                               │
             ▼                               ▼                               ▼
       Verify → Commit                 Verify → Commit                 Verify → Commit
                                                                             │
                                                                             ▼
                                                                    End-to-end smoke
                                                                    Issue closures
```

## The 7 operational rules

### 1. Wiki page is the spec, not the conversation

Subagents start with empty context. The orchestrator's conversation history is **not visible to them**. Therefore the design must live in a durable artifact (typically `docs/wiki/methods/<topic>.md`) the subagent reads as its first action.

If the spec is only in conversation context, you cannot delegate to subagents — the brief would have to re-explain everything every time. **Spec-as-document is non-negotiable.**

### 2. One subagent per phase, not finer

Don't decompose a phase into "wave 1, wave 2, wave 3" sub-tasks across parallel subagents. Coarse-grained matches the phase model and avoids coordination overhead. Within a phase, the subagent itself can run parallel operations (parallel file edits, parallel tests).

Empirical: phases tend to have strong sequential dependencies internally (e.g. moving a class before something imports it from its new location). The "intra-phase parallelism" gain is typically 2-5 minutes wallclock, not worth the briefing complexity.

### 3. Strict file whitelist in the brief — not "surgical changes"

Generic "make surgical changes only" guidance is **insufficient** to prevent scope creep. Subagents will help-fully clean up adjacent code unless the brief is structurally explicit:

```
## STRICT FILE WHITELIST

**Only modify, create, or move these. Touching anything else is out of scope.**
If you think another file needs editing, **stop and report**, do not modify.

### NEW (create):
- path/to/new_file.py
- ...

### REWRITE (replace contents):
- path/to/existing_file.py — what it becomes
- ...

### EDIT (line-level edits):
- path/to/sbatch_file.sbatch
- ...

**Do NOT touch:** [list of nearby-but-untouched files, especially those the subagent
might be tempted to "harmonize"].
```

Phase 1 of the launcher refactor (2026-04-25) demonstrated the cost: Phase 1 subagent commented out 4 BP debug-prints across 3 files outside its brief, "for consistency" with the BP#2 removal that *was* in scope. The user's `feedback_park_as_issue.md` rule was violated. Phase 2 + 3 used explicit whitelists and produced **zero scope-creep**.

### 4. Trust-but-verify after every subagent run

Subagent reports describe **intent**, not necessarily reality. Always run before accepting:

- `git status --short` + `git diff --stat HEAD` — confirms only whitelisted paths changed
- The full test command from the brief's exit gate — confirms green
- `ls` on any newly-created directories — confirms they actually exist
- Spot-check 1-2 files for content sanity (especially shim format consistency)

If any check disagrees with the report, dig in. Phase 1's BP scope creep was caught at this step (the report didn't mention `agent.py` and `networks.py` modifications; `git status` did).

### 5. End-to-end smoke is a REQUIRED gate, not a manual aftermath

This is the lesson the launcher refactor taught the hard way. Unit tests + integration tests can all be green while production invocation is broken — pytest configures `sys.path` differently than direct script invocation, IDEs configure environments differently than SLURM, etc.

For any refactor that changes invocation paths (entrypoint scripts, sbatch wrappers, CLI flags), one **direct production-equivalent invocation** must succeed before declaring "done":

- For training scripts: `python <shim> --steps 1000 --prefill 100 --output_dir /tmp/smoke_<id>`
- For services: actually `curl` the new endpoint
- For batch jobs: `sbatch --steps 1000` and wait for the SLURM out file

If smoke fails after issue-closures, **reopen** or comment, then add a Phase N+1 fix commit.

### 6. Phase boundary = commit boundary

Each phase ends with:
1. Verify gates green
2. Diff review with user (orchestrator stages explicitly-named files; never `git add -A` per `CLAUDE.md`)
3. User explicit approval to commit
4. `git commit` with conventional-commits message + phase-specific summary
5. (Final phase only) `git push` after explicit user approval

The orchestrator NEVER commits without user approval, even after a clean verify. The phase boundary IS the user's review opportunity. Don't shortcut it.

### 7. Test gaps surface as parked issues, not silent compromises

When a verification gate misses a class of bug (today's lesson: pytest doesn't simulate `python script.py`), park the gap as a GitHub issue **before** closing the parent. The Phase 4 fix commit is forensic evidence; the test gap is the recurring risk.

Per the user's `feedback_park_as_issue.md` memory, every out-of-scope or brainstorm item gets an issue before session ends. This is the inverse: every *test* that should have caught a bug but didn't, gets an issue too.

## Brief template for phase subagents

```
You are implementing Phase <N> of <issue/RFC reference> for <repo path>. Phases <N-1> just landed (commit `<sha>`). Your job is <one-line scope>.

## Required reading (do this first)

1. <wiki page>, especially the "Phase <N>" section.
2. CLAUDE.md — conventions (uv run, pytest -v, no emojis, surgical changes).
3. <2-4 specific source files the subagent will read or modify>.

## STRICT FILE WHITELIST

**Only modify these files. Touching anything else is out of scope.**

### NEW: ...
### REWRITE: ...
### EDIT: ...
### Do NOT touch: ...

## Concrete deliverables

[Section per file or feature, with specific signatures, exact code patterns
where helpful, and pointers to existing patterns to mirror.]

## Conventions
- uv run python ...
- pytest -v
- No emojis, no multi-paragraph docstrings, single-line WHY comments only
- No Co-Authored-By lines

## Honest-fit caveat (optional, for refactors with abstraction risk)

If extracting <X> requires inventing artificial abstractions (>N lines for a wrapper, >M lines for a "shared" helper used by 2 callers), STOP AND REPORT. We'd rather leave it flat than fabricate structure.

## Exit gate

```bash
[concrete pytest commands, including marker subsets like -m "not gpu"]
```

All must stay green.

## Reporting format

```
## Phase <N> Status: <COMPLETED|PARTIAL|BLOCKED>
### Files created/modified/archived: ...
### Test results: ...
### Honest-fit verdict (if applicable): ...
### Unresolved: ...
### Notes: ...
```

Begin.
```

## Lessons from the 2026-04-25 launcher refactor

The pattern was **not** clean from the start. Each phase taught a lesson:

| Phase | What broke | Fix |
|---|---|---|
| 1 | Brief said "surgical changes only" but subagent commented out 4 BP debug-prints across 3 files outside scope. Report didn't mention them. | Phase 2/3 briefs added STRICT FILE WHITELIST section with explicit "Do NOT touch" list. Zero scope creep after. |
| 2 | Subagent dropped `sys.path.insert(0, "../../..")` boilerplate from shims (looked like cruft). Pytest stayed green. Smoke would have failed at first sbatch. | Phase 4 fix added bootstrap back. Issue #91 filed for test-gap (subprocess invocation), implemented as `test_shim_invocation.py`. **End-to-end smoke is now a required gate.** |
| 3 | "Honest-fit check" worked — subagent stopped, reported, the abstraction did fit cleanly. | Pattern: include "STOP and report if abstraction doesn't fit" caveat in briefs that involve extracting shared code. |
| Verify | `git status` revealed Phase 1's scope creep that the report omitted. | Always run `git status` + `git diff --stat` after every subagent, regardless of how clean the report sounds. |

## Cost / time observations (n=1 session)

Launcher refactor, 2026-04-25:
- 3 phase subagents (Sonnet) total: ~30 min wallclock, ~195k tokens
- 1 fix subagent + manual edits: ~5 min
- 2 end-to-end smokes: ~12 min
- Total Opus orchestrator time: ~3 hours conversation including grill + verify gates
- Final delivered: 6 commits, ~1.4k lines net reduction in scripts/, 25 launch tests + 59 existing tests + 9 subprocess smoke tests = 93 tests green

The model split worked: Sonnet's coarse-grained implementations were perfectly serviceable; Opus orchestration caught the scope creep, the import bug, and the test gap. Pure-Sonnet would likely have shipped all three issues. Pure-Opus would have spent ~3× the tokens for the same code outcome.

## Anti-patterns to avoid

- **Don't** spawn parallel subagents for non-independent work just because parallelism feels productive.
- **Don't** trust subagent reports without `git status` + diff verification.
- **Don't** treat unit-tests-green as "done" — smoke-tests-green is the real bar.
- **Don't** close parent issues before the smoke-test passes (or, if you do, expect to reopen-and-comment when it fails).
- **Don't** let "surgical changes" be the only scope rule — explicit file whitelist prevents 80% of scope creep.
- **Don't** let "out of scope but I'll just do it real quick" creep into briefs — every brainstorm/cleanup item parks as a GitHub issue.

## Related

- [methods/workflow.md](workflow.md) — broader pipeline this pattern fits into
- [methods/launcher-refactor.md](launcher-refactor.md) — the design this pattern delivered
- `feedback_skills_over_agents.md` (memory) — when to use this pattern vs default `/engineer`
- `feedback_park_as_issue.md` (memory) — the issue-parking discipline
- Issue [#91](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/91) — the test-gap that motivated the smoke-test rule
