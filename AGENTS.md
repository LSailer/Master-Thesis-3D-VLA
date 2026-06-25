# AGENTS.md

Repo-wide instructions for coding agents on bwUniCluster. Scoped `AGENTS.md`
files may add stricter rules.

## Style

- Be brief, direct, and technical. No fluff or emojis.
- Answer questions before running implementation commands or editing.
- When responding to feedback/analysis, say whether you agree or disagree first.
- Give me step-by-step reasoning along with the answer.

## Code changes

- Read relevant files in full before broad edits; do not rely only on search snippets.
- Avoid `any` and untyped code unless no practical typed alternative exists.
- Ask before removing intentional-looking functionality.
- Fix lint/type issues instead of silencing them; ask before adding scoped disables.
- Do not preserve backward compatibility unless explicitly requested.
- Keep keybindings configurable; add defaults instead of hardcoded key checks.

## Coding standards

Apply these standards to production and scratchpad code.

- Clarity: use meaningful names for functions, variables, classes, and modules.
  Code should be understandable without deep domain context.
- Comments and docstrings should explain why, assumptions, contracts, inputs,
  outputs, shapes/dtypes, or side effects. If a comment explains what a block
  does, prefer clearer code or a well-named extracted function.
- Simplicity: follow KISS. Solve the current problem with straightforward code;
  avoid clever patterns, premature generalization, and YAGNI features.
- Consistency: follow local project style, naming, typing, and structure over
  personal preference. Match nearby idioms before introducing a new pattern.
- Refactorability: keep modules cohesive and loosely coupled. Each component
  should have one clear responsibility. Avoid changes that require touching many
  unrelated files.
- Design principles: use SOLID as guidelines where useful, but prioritize
  understandable, modifiable code over pattern compliance.
- Dead code: remove dead code and resolve TODOs when in scope. If a TODO cannot
  be fixed now, make it explicit, actionable, and justified.
- Language idioms: write idiomatic code for the language in use, e.g. Pythonic
  Python rather than Java-style Python.
- Complex conditionals: extract named predicates or functions. If logic remains
  hard to follow, question and simplify the underlying state/domain model rather
  than only rearranging the condition.
- AI-generated code must satisfy the same standards. Do not add verbose comments
  or documentation that merely restates the implementation.

## Scratchpad workflow

Default mode is generation-then-comprehension: agents generate, verify, and explain
inside `scratchpad/`; the user manually copies accepted code into production.

- Create or edit files only under `scratchpad/` unless the user explicitly
  enables repo-edit mode for the current task.
- In pi, use `/repo-edit` to permit direct repo edits, and `/scratch-only` to
  return to scratch-only mode. Outside pi, require an explicit message such as
  `Repo edit mode: ON. You may edit <paths>`.
- In scratch-only mode, production changes must be provided as a patch, exact
  target file list, or copy/paste-ready snippet; do not apply them directly.
- Scratchpad code must import existing project functions, classes, constants,
  and types instead of reimplementing them. Inspect signatures and use current
  variable names from the surrounding code.
- When using web, code, or documentation search for project work, save an
  evidence note as Markdown under `scratchpad/research/` before relying on the
  result. Include sources, relevant API facts, and open uncertainties.
- For generated Python code, add docstrings that describe input parameters,
  return values, shapes/dtypes when relevant, and side effects.
- Record assumptions and decisions as short `# Assumption:` or `# Decision:`
  comments only where they explain why a non-obvious choice was made.
- Keep generated code clear, simple, consistent with nearby idioms, modular, and
  easy to refactor. Avoid cleverness, premature generalization, dead code, and
  unresolved TODOs.
- Validate generated Python with `python -m pylint <changed scratchpad paths>`
  and fix all reported errors, warnings, and infos unless the user accepts a
  documented blocker.
- For GPU-related generated code, run a narrow GPU smoke check from the repo root
  with `PYTHONPATH=src` and save the command plus result under
  `scratchpad/checks/`. Do not call it a test unless it executes a test suite.

## Commands

- Run commands directly; do not hide output with `tail`/pipes.
- Do not run Pylint/Pyright after every edit.
- Before committing non-doc code changes, run the narrowest relevant checks:
  - `python -m pylint <changed paths>`
  - configured Pylance/Pyright diagnostics when available
- Do not call checks “tests” unless they actually run tests.
- Fix all reported errors, warnings, and infos before committing.

## Context routing

- Read the nearest scoped `AGENTS.md` before editing, especially:
  - `src/r2dreamer/AGENTS.md`
  - `src/vggt/AGENTS.md`
  - `scripts/r2dreamer/AGENTS.md`
  - `tests/r2dreamer/AGENTS.md`
- Load `CONTEXT.md` and relevant `docs/adr/` entries for architecture,
  data-flow, experiment-contract, or cross-module work.
- Linear issues live in `3D-WM-ObjectNAV`; read the issue/spec before editing.
- Treat `src/prototyp/` as protected prototype output; edit it only when the user
  explicitly names it.

## Thesis writing

The thesis is the separate git repo `../writing/`
(`/pfs/data6/home/ul/ul_student/ul_hfj15/writing/`). Commit there separately.
Reportable experiment prose and figures belong there, not only in `docs/notes/`.

Key files:
- `../writing/document.tex`
- `../writing/content/{Introduction,Related work,Method,Experiments,Discussion,Appendix}.tex`
- `../writing/img/`

Follow existing section, label, `natbib`/`splncs04`, and figure conventions.

## Worktrees

- Main checkout is for orchestration: status, SLURM/W&B/PR/Linear inspection,
  branch review, and worktree management.
- Do not create/add worktrees unless the user explicitly asks.
- Worktree names: `worktrees/<linear-key>-<slug>/` or
  `worktrees/<intent-prefix>-<slug>/`.
- Before editing, run:
  - `git rev-parse --show-toplevel`
  - `git status --short --branch`
  - `git worktree list`
- In fresh worktrees, run `./scripts/setup_worktree.sh` before training/eval.

## Naming and review

- Branch/worktree with Linear key: `<linear-key>-<slug>`.
- Without Linear key: `fix-<slug>`, `add-<slug>`, `remove-<slug>`, or
  `chore-<slug>`.
- Commit/PR title: `<Linear issue key>: <summary>` when tied to Linear;
  otherwise `<type>: <summary>`.
- Do not include agent/tool/model names in branch, worktree, commit, or PR titles.
- For PRs/reviews, follow `docs/agents/pr-workflow.md`.
