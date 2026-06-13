# AGENTS.md

Compact dispatcher for agents working in this repo on bwUniCluster.

## Context

- Read the nearest scoped `AGENTS.md` before editing:
  - `src/r2dreamer/AGENTS.md`
  - `src/vggt/AGENTS.md`
  - `scripts/r2dreamer/AGENTS.md`
  - `tests/r2dreamer/AGENTS.md`
- `CONTEXT.md` and `docs/adr/` hold durable project/design context; load them when the task needs architecture, data-flow, experiment-contract, or cross-module context.
- Issues are tracked in Linear on `3D-WM-ObjectNAV`; read the issue/spec before editing.
- Thesis prose and figures belong in the sibling `../writing/` repo; follow `../writing/AGENTS.md` and commit it separately.

## Worktrees

- The main checkout is for orchestration: status checks, SLURM/W&B/PR/Linear inspection, branch review, and worktree management.
- Implement in `worktrees/<linear-key>-<short-task-slug>/` on a matching branch unless the user explicitly says otherwise.
- Before editing, run `git rev-parse --show-toplevel`, `git status --short --branch`, and `git worktree list`.
- In fresh worktrees, run `./scripts/setup_worktree.sh` before training or eval.

## Naming

- Branch/worktree: `<linear-key>-<short-task-slug>`
- Commit/PR title: `<Linear issue key>: <summary>`
- Do not include agent/tool/model names in branch, worktree, commit, or PR titles.

## Workflow

- Before editing: read the Linear issue/spec, inspect relevant files, and check git status/worktree.
- Implementation: use the `$tdd` skill by default.
- Before handoff: run the narrowest useful verification for the files changed. If a check is skipped, state the skipped command and why.
- For PRs and reviews, follow `docs/agents/pr-workflow.md`.
