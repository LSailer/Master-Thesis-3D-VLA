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

## Architecture And Data Flow

The canonical repo-wide architecture/data-flow reference is
[`docs/architecture-data-flow.html`](docs/architecture-data-flow.html). Read it before
editing encoder routing, VGGT readouts, replay layouts, RSSM training flow, or
architecture-sensitive docs. When adding reports or notes, link to that page for
the shared pipeline and keep local docs focused on experiment-specific deltas,
evidence, and results.

## Thesis Writing

The written thesis is a separate LaTeX repo at [`../writing/`](../writing/)
(sibling of this checkout; absolute path
`/pfs/data6/home/ul/ul_student/ul_hfj15/writing/`). It is version-controlled
independently; commit there separately, never from this repo. When an experiment
here produces a reportable result, the prose and figures belong there, not only
in `docs/`:

- [`../writing/document.tex`](../writing/document.tex) - main file; `\input{content/...}` in reading order
- [`../writing/content/Experiments.tex`](../writing/content/Experiments.tex) - results, per-level SR/SPL, encoder comparisons
- `../writing/content/{Introduction,Related work,Method,Discussion,Appendix}.tex`
- `../writing/img/` - figures, referenced by bare filename in `\includegraphics`; drop generated PNGs here

Match the existing `\section`/`\subsection`, `\label{sec:|fig:|tab:}`, and
`natbib` (`splncs04`) conventions already in those files. The repo-side
`docs/notes/` writeups are staging drafts; the thesis is the destination.

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
