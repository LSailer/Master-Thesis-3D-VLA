# AGENTS.md

Repo-wide instructions for coding agents on bwUniCluster. Scoped `AGENTS.md`
files may add stricter rules.

## Core behavior

- Be brief, direct, and technical. No fluff or emojis.
- Answer questions before editing or running implementation commands.
- When responding to feedback or analysis, say whether you agree or disagree first.
- Give concise step-by-step reasoning with recommendations.
- Stay inside the user-requested scope; ask before removing intentional-looking behavior
  or making broad architectural changes.
- Do not include agent, tool, or model names in branch, worktree, commit, or PR titles.

## Editing rules

- Before direct repo edits, inspect repo state with `git rev-parse --show-toplevel`,
  `git status --short --branch`, and `git worktree list`.
- Read relevant files in full before broad edits; do not rely only on search snippets.
- Read the nearest scoped `AGENTS.md` before editing under `src/r2dreamer/`,
  `src/vggt/`, `scripts/r2dreamer/`, `tests/r2dreamer/`, or `src/prototyp/`.
- Prototype-skill work is scratchpad-only. Other repo edits are allowed when the
  user has requested or approved the target scope.
- Treat `src/prototyp/` as protected prototype output; edit it only when the user
  explicitly names it.
- Do not write secrets or credentials. Do not edit `.env`, `.git/`, or policy files
  unless the user explicitly requests that exact file.
- Avoid untyped code and `any` unless no practical typed alternative exists.
- Do not preserve backward compatibility unless explicitly requested.
- Keep keybindings configurable; add defaults instead of hardcoded key checks.

## Code quality

- Prefer clear names, local idioms, small cohesive modules, and straightforward code.
- Avoid cleverness, premature generalization, dead code, and unresolved TODOs.
- Avoid trivial wrapper functions whose body is only an optional docstring and a
  single `return`, unless the wrapper creates a real domain abstraction or isolates
  a volatile dependency.
- Comments and docstrings should explain contracts, assumptions, shapes/dtypes,
  side effects, or why something is non-obvious; do not restate implementation.

## Validation

- During iterative edits, prefer inspection and narrow syntax checks. Do not run
  Pylint, Pyright, or pytest after every edit.
- Before handoff or commit of non-doc Python changes, run the narrowest relevant
  checks, including `python -m pylint <changed paths>` where practical.
- CPU tests are fine for CPU-safe code paths.
- End-to-end, Habitat, VGGT, training, profiling, and GPU-marked checks must run
  under `srun`/sbatch, never directly on a login node.
- Do not call checks “tests” unless they execute a test suite.
- Do not hide validation output with `tail` or truncating pipes; filtering discovery
  output is okay.

## Context and tracking

- Read `CONTEXT.md` and relevant `docs/adr/` entries for architecture, data-flow,
  experiment-contract, or cross-module work.
- Linear project `3D-WM-ObjectNAV` exists but is not always active. Read or update
  Linear only when the user gives a key or asks for issue-tracker work.
- For PRs/reviews, follow `docs/agents/pr-workflow.md`.

## Thesis writing

The thesis is the separate git repo `../writing/`
(`/pfs/data6/home/ul/ul_student/ul_hfj15/writing/`). Commit there separately.
Reportable experiment prose and figures belong there, not only in `docs/notes/`.

Key files:
- `../writing/document.tex`
- `../writing/content/{Introduction,Related work,Method,Experiments,Discussion,Appendix}.tex`
- `../writing/img/`

Follow existing section, label, `natbib`/`splncs04`, and figure conventions.

## Worktrees and naming

- Do not create or add worktrees unless the user explicitly asks.
- Worktree names: `worktrees/<linear-key>-<slug>/` or
  `worktrees/<intent-prefix>-<slug>/`.
- In fresh worktrees, run `./scripts/setup_worktree.sh` before training/eval.
- Branch/worktree with Linear key: `<linear-key>-<slug>`.
- Without Linear key: `fix-<slug>`, `add-<slug>`, `remove-<slug>`, or
  `chore-<slug>`.
- Commit/PR title: `<Linear issue key>: <summary>` when tied to Linear;
  otherwise `<type>: <summary>`.
