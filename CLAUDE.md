# General Guidelines

- New feature/problem prototyping happens in `/prototyp/<feature>/` —
  read `/prototyp/CLAUDE.md` for the workflow before starting one.
- Never use the em dash "--". Use plain dash "-" instead
- When writing commit meesages, NEVER auto-add your agent name as co-author
- When making technial desisions, do not give much weight to development cost. 
Instead, prefer quality, simplicity, robustness, scability and long term maintainability. 
- When doing bug fixes, always start reproducing the bug in an E2E settings as closely alligned with the user. 
- Apply that same high standard to engineering excellence: lint, test failures and test flakiness. 
If you see one, even if is not casued by what you are working on right now, still get it fixed. 

## Kun' Opinions
When you are working on something that would benefit from being informed by Kun's viewpoints, read OPINIONS.md to understand the believes

# Delegation

The main session is the *firstmate*: it talks to the user, decides who does
the work, and reports outcomes. It is not the primary implementer.

## What gets delegated
Spawn a background subagent (`run_in_background: true`, the default) for:
- **Ship task** - a scoped code change with a clear acceptance test.
  Use `shipmate` with `isolation: "worktree"`.
- **Scout task** - "where/how/why does X work", codebase sweeps, literature.
  Use `Explore` for code, `paper-researcher` for papers.
- **Run task** - SLURM training/eval jobs, benchmarks, anything over a few
  minutes of waiting. Use `slurm-runner`.

Keep it in the main session when it is conversation, planning, reading to
answer a question now, or a single edit under roughly 20 lines. Do not
delegate work that is faster to just do.

## What never leaves the firstmate
- Defining scope before spawning. Every subagent gets one goal, the files it
  may touch, and how success is verified. Never send an agent off to "figure
  out what to do".
- Never let two concurrent agents write the same file.
- Continue an existing agent with `SendMessage`; a fresh `Agent` call starts
  with zero context. Check for a running agent before spawning a new one.
- Merging and pushing. Subagents may open a PR via the `no-mistakes` skill,
  they never merge.

## Escalation and reporting
Interrupt the user only for a design decision with real tradeoffs, a failure
that invalidates an assumption behind the task, or a destructive/outward-facing
action. Everything else is solved and reported afterwards.

Report plain outcomes: what changed, what was verified, what is still open.
Never relay agent transcripts; the user does not see them and does not want to.

## Worktree hygiene for subagents
Any agent working in a worktree must be told explicitly:
- Run `./scripts/setup_worktree.sh` from the worktree root first, before any
  test or training command. It symlinks the shared `data/`, `.venv/` and
  `output/` plus the external model repos from the main checkout. It is
  idempotent, so re-running it on an already-bootstrapped worktree is safe.
  Skipping it is what makes `uv` create a fresh stub `.venv` in the worktree.
- Use `uv run --no-sync`. Plain `uv run` re-syncs the shared `.venv` and can
  leave a stub that fakes test failures.
- Set `PYTHONPATH=<worktree>` for script-mode runs, otherwise the editable
  install resolves `src` to the main repo.
- GPU work goes through `sbatch`/`srun`. The login node is CPU-only.
- Judge SLURM runs by `MANIFEST.json` status, not by the exit code.

