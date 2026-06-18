# AGENTS.md

Compact dispatcher for agents working in this repo on bwUniCluster.

## Conversational Style

- Keep answers short and concise.
- Use technical prose only; be direct.
- Avoid fluff, praise, or cheerful filler.
  - Prefer: "Thanks @user"
  - Avoid: "Thanks so much @user!"
- Do not use emojis in commits, issues, PR comments, or code.
- When the user asks a question, answer it first before making edits or running implementation commands.
- When responding to user feedback or analysis, explicitly state whether you agree or disagree before describing changes.

## Code Quality

- Read relevant files in full before broad changes, audits, or editing files not yet inspected. Do not rely only on search snippets.
- Avoid `any` and untyped code unless there is no practical typed alternative.
- Always ask before removing functionality or code that appears intentional.
- Do not silence pylint disables. Fix the warning. If disabling is genuinely needed, ask first and use the smallest possible scoped disable with a reason.
- Do not preserve backward compatibility unless explicitly requested.
- Do not hardcode key checks such as `matchesKey(keyData, "ctrl+x")`.
  Add defaults to `DEFAULT_EDITOR_KEYBINDINGS` or `DEFAULT_APP_KEYBINDINGS`
  so keybindings remain configurable.

## Commands

- After code changes, not docs-only changes, run the narrowest required check:
  - Python: `python -m pylint <changed paths>`
- Do not treat checks as tests unless they explicitly run tests.
- Run commands directly; do not pipe to `tail` or hide output.
- Fix all errors, warnings, and infos before committing.

## Context

- Read the nearest scoped `AGENTS.md` before editing:
  - `src/r2dreamer/AGENTS.md`
  - `src/vggt/AGENTS.md`
  - `scripts/r2dreamer/AGENTS.md`
  - `tests/r2dreamer/AGENTS.md`
- `CONTEXT.md` and `docs/adr/` hold durable project/design context; load them when the task needs architecture, data-flow, experiment-contract, or cross-module context.
- Issues are tracked in Linear on `3D-WM-ObjectNAV`; read the issue/spec before editing.
- Thesis prose and figures belong in the sibling `../writing/` repo; follow `../writing/AGENTS.md` and commit it separately.


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
- Implement medium/high-risk tasks in `worktrees/<linear-key>-<short-task-slug>/`. For small, single-purpose edits, use the current checkout when clean; otherwise create a worktree.
- Before editing, run `git rev-parse --show-toplevel`, `git status --short --branch`, and `git worktree list`.
- In fresh worktrees, run `./scripts/setup_worktree.sh` before training or eval.

## Naming

- Branch/worktree: `<linear-key>-<short-task-slug>`
- Commit/PR title: `<Linear issue key>: <summary>`
- Do not include agent/tool/model names in branch, worktree, commit, or PR titles.

- For PRs and reviews, follow `docs/agents/pr-workflow.md`.
