# Claude Code Workflow

**Tags**: #workflow #skills #pipeline
**Date**: 2026-04-13 (revised 2026-05-03 — autonomous trigger + actual flow)

---

## TL;DR — autonomous flow (current as of 2026-05-03)

```
gh issue create --template task.yml          # write issue, label `backlog`
gh issue edit N --add-label ready --add-label AFK
   ↓ (within 15 min)
.github/workflows/ralph-cron.yml             # scheduled trigger
   → dispatches .github/workflows/ralph.yml  # self-hosted BWUniCluster runner
       → Claude Code: worktree → TDD → smoke (srun dev_gpu_h100) → PR
   ↓ (you review PR, merge)
sbatch scripts/r2dreamer/slurm/<variant>.sbatch     # manual today
   → train.sbatch → verify.sbatch → report.sbatch  (deferred wiring)
```

**What's wired today**: cron-fires-Ralph and Ralph-creates-PR. **Deferred**: auto-launching `scripts/pipeline/launch.sh` from PR merge, Telegram, cross-repo paper-recap. See parked issues for each.

## Philosophy

Three sources inform this workflow:

- **Karpathy's coding guidelines** — behavioral guardrails that shape how Claude writes code at every step (clarify before coding, simplicity first, surgical changes, goal-driven execution). These are embedded in `CLAUDE.md`, not a separate skill.
- **Pocock's skill pipeline** — workflow stages that structure when Claude does what (plan, build, review, report). These are implemented as skills.
- **Karpathy's LLM Wiki** — a persistent, compounding knowledge base maintained by Claude in `docs/wiki/`. Knowledge is compiled once and kept current, not re-derived every session.

## Skills (7 total)

| Skill | Purpose | When to use |
|-------|---------|-------------|
| `/plan` | Grill → PRD → vertical-slice plan | Starting a new feature or experiment |
| `/engineer` | Implement plan with TDD for infrastructure | After `/plan` completes |
| `/review` | Find bugs, simplify code | After `/engineer` completes |
| `/reporter` | Wiki page + plot scripts + HTML slides | After experiments produce results |
| `/triage-issue` | Investigate bug → GitHub issue with fix plan | When a bug is reported or SLURM job fails |
| `/wiki` | Ingest, query, lint the knowledge base | Anytime — meeting notes, papers, questions |
| `/slurm-submit` | (planned, **not yet implemented**) Trigger GPU jobs via GitHub Actions | Today: hand-submit via `sbatch scripts/r2dreamer/slurm/<variant>.sbatch` |

## Pipelines

### Feature track (new experiments, architectural changes)

```
/plan → /engineer → /review → [submit to SLURM, wait] → /reporter → human chat review
```

1. **`/plan`** — Three phases: grill (interview relentlessly), document (PRD as GitHub issue), slice (vertical phases saved to `docs/plans/`). Always runs all three.
2. **`/engineer`** — Implements phase by phase. Uses TDD (test-first) for infrastructure code (env wrappers, networks, replay buffer, preprocessing). Implements experiment code (training loops, configs) directly.
3. **`/review`** — Two-pass: find bugs/logic errors, then simplify. Runs pytest.
4. **Submit training** — today, manual: `sbatch scripts/r2dreamer/slurm/<variant>.sbatch`. (`/slurm-submit` skill is planned, not built.) Wait for results.
5. **`/reporter`** — Creates three deliverables: plot scripts in `src/*/scripts/`, wiki experiment page in `docs/wiki/experiments/`, HTML slides in `docs/`. Updates wiki index and log.
6. **Human chat review** — Read the slides, ask questions. Claude answers from wiki and output data. Unresolved questions become GitHub issues via `gh issue create`.

### Fix track (bugs, small changes)

```
/triage-issue → /engineer → /review
```

1. **`/triage-issue`** — Investigates the bug, finds root cause, creates GitHub issue with TDD fix plan.
2. **`/engineer`** — Implements the fix following the issue's TDD plan.
3. **`/review`** — Validates the fix, simplifies.

### Wiki operations (anytime)

```
/wiki ingest    — add meeting notes, paper summaries, method decisions
/wiki query     — search wiki to answer questions
/wiki lint      — health check for contradictions, orphan pages, stale claims
```

## TDD Rules

TDD applies to **infrastructure code** — the parts other code depends on that break silently:
- Env wrappers (`crafter.py`, `habitat.py`) — observation shapes, action spaces, reset/step contracts
- Network building blocks (`networks.py`) — tensor shapes through encoder/decoder/RSSM
- Replay buffer — sampling, sequence lengths, dtype preservation
- Preprocessing pipelines — UNITE feature extraction, curriculum logic

TDD does **not** apply to experiment code — training loops, hyperparameter sweeps, SLURM scripts. These change too fast and their correctness is defined by experimental results.

## Wiki Conventions

- `/reporter` auto-writes experiment pages to `docs/wiki/experiments/`
- All other wiki pages (meetings, methods, research) are added via `/wiki ingest` or manual chat
- Both reporter and wiki follow the same conventions: update `index.md`, append to `log.md`
- Templates live in `docs/wiki/_templates/` (experiment template, generic template)
- Cross-reference pages using `[[page-name]]` in `## Related` sections

## Deliverable Formats

| Type | Format | Location |
|------|--------|----------|
| Wiki notes | Markdown | `docs/wiki/` |
| Experiment presentations | HTML slides (dark theme) | `docs/<experiment>.html` |
| Plots | PNG (publication-quality) | `output/figures/` |
| Plans | Markdown | `docs/plans/` |
| PRDs | GitHub issues | GitHub |
| Bug reports | GitHub issues | GitHub |

HTML slides are **replaced** (not appended) when an experiment is updated — git preserves history.

## Related

- [[dreamerv3-architecture]]
- [[2026-03-03-braun]]
