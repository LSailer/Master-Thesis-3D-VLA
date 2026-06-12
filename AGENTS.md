# AGENTS.md

This repo is worked on by coding agents on bwUniCluster. Follow these rules for Linear issue implementation and review.

`AGENTS.md` is the tooling-agnostic project contract read by every major agentic CLI (Claude Code, Codex, Cursor, Copilot, Aider, …). The pattern and conventions here follow [matthewsinclair/intent](https://github.com/matthewsinclair/intent). The nearest-ancestor file wins, so large subfolders carry their own scoped module contract — read the one closest to the files you are editing:

- [`src/r2dreamer/AGENTS.md`](src/r2dreamer/AGENTS.md) — R2Dreamer JAX/Flax agent (RSSM, behavior, representation, launch)
- [`src/vggt/AGENTS.md`](src/vggt/AGENTS.md) — VGGT 3D encoder (PyTorch reference + JAX production port)
- [`scripts/r2dreamer/AGENTS.md`](scripts/r2dreamer/AGENTS.md) — experiment drivers + SLURM wrappers
- [`tests/r2dreamer/AGENTS.md`](tests/r2dreamer/AGENTS.md) — R2Dreamer test suite

## Thesis Writing

The written thesis is a **separate LaTeX repo** at [`../writing/`](../writing/) (sibling of this checkout; absolute path `/pfs/data6/home/ul/ul_student/ul_hfj15/writing/`). It is version-controlled independently — commit there separately, never from this repo. When an experiment here produces a reportable result, the prose and figures belong there, not only in `docs/`:

- [`../writing/document.tex`](../writing/document.tex) — main file; `\input{content/...}` in reading order
- [`../writing/content/Experiments.tex`](../writing/content/Experiments.tex) — results, per-level SR/SPL, encoder comparisons
- `../writing/content/{Introduction,Related work,Method,Discussion,Appendix}.tex`
- `../writing/img/` — figures, referenced by **bare filename** in `\includegraphics`; drop generated PNGs here

Match the existing `\section`/`\subsection`, `\label{sec:|fig:|tab:}`, and `natbib` (`splncs04`) conventions already in those files. The repo-side `docs/notes/` writeups are staging drafts; the thesis is the destination.

## HTML Report Handoff

When creating or updating local HTML reports under `docs/`, `docs/notes/`, or worktree-local `docs/` folders, always hand off a browser-openable localhost URL in the final response, not only a filesystem path. Prefer the already active in-app browser/server origin when visible (for example `http://localhost:<port>/...`), preserving the path relative to the repository root. If no active origin is visible but a Python HTTP server is known to be running from the repo root, use `http://localhost:8000/<relative-path>`. If no server is running or the serving root is unclear, state the exact relative path and the command/server root needed to serve it.

Do not hard-code localhost URLs into portable HTML files. Keep links inside HTML documents relative, so the same file works from `localhost`, the Codex in-app browser port, GitHub Pages, or a static file preview. If a report should be discoverable from the docs landing page, add a relative link from the appropriate index page separately.

## Multi-Agent Worktree Policy

The main checkout is the orchestrator/control checkout. Use it for status checks, SLURM/W&B/PR inspection, branch review, worktree creation/removal, and coordination.

Do not implement feature changes in the main checkout unless the user explicitly asks for that. Coding agents must work in a dedicated Git worktree on a dedicated branch.

Use `worktrees/<linear-key>-<short-task-slug>/` as the canonical workspace layout for agent implementation work. The parent checkout ignores `worktrees/`, so nested worktrees do not make the orchestrator checkout dirty.

Before editing, agents must check where they are:

```
git rev-parse --show-toplevel
git rev-parse --git-dir
git rev-parse --git-common-dir
git status --short --branch
git worktree list
```

If `git rev-parse --git-dir` and `git rev-parse --git-common-dir` both resolve to the main checkout's `.git`, the agent is in the orchestrator checkout and must create or switch to a task worktree before editing code.

Each agent owns only its assigned worktree and branch. Do not edit another agent's worktree, and do not remove or clean another worktree unless explicitly asked.

## Worktree Setup

When working in a `git worktree` (any directory under `worktrees/`), run once per fresh worktree before any training or eval command:

```
./scripts/setup_worktree.sh
```

This symlinks the shared `data/` and `.venv/` from the main checkout into the worktree. Habitat and other scripts resolve dataset paths relative to CWD, so a worktree without these links fails fast with `FileNotFoundError: data/datasets/...`. The script is idempotent and is a no-op in the main checkout.

## Naming Conventions

Use names that describe the Linear issue or task, not the tool or agent doing the work.

- Branches and worktrees: `<linear-key>-<short-task-slug>`; example: `3d-47-skip-heldout-eval`
- Commit subjects and PR titles: `<Linear issue key>: <short change summary>`; example: `3D-47: Skip heldout eval for offline ablations`
- Do not include tool or agent names such as `codex`, `ai`, `agent`, `bot`, or model names in branch names, worktree names, commit subjects, or PR titles.
- Mention agent/tool involvement only in the PR body `Agent involvement` section.

## Default Workflow

Before editing:
- Read the Linear issue, linked spec, and relevant existing files.
- Identify the acceptance criteria and non-goals.
- Check current implementation patterns before adding new ones.
- Inspect current git status so unrelated work is not disturbed.
- If Linear write tools are available, add a short issue comment that work has started and note the intended verification. If Linear write tools are not available, say so in the final response.

While editing:
- Implement only the stated acceptance criteria.
- Do not change unrelated files.
- Do not refactor opportunistically.
- Preserve existing behavior unless the issue explicitly changes it.
- Follow existing code style, architecture, naming, and UI conventions.
- Add or update tests when the change affects logic, data flow, permissions, integrations, or user-visible behavior.

Before opening a PR:
- Run the relevant checks for the files touched.
- Run `scripts/check_policy.sh` from the repository root before pushing, publishing, opening, or updating a PR. The policy currently runs Semgrep Code through `semgrep ci --code` only on `src/**` and `scripts/**`; agents must not push or create/update a PR if this check fails.
- Review the diff for unrelated changes.
- Confirm the PR description follows `.github/pull_request_template.md` when that template exists.
- Add a Linear issue comment summarizing what changed, acceptance criteria checked, verification run, known blockers, and anything intentionally not done. If the implementation cannot satisfy an acceptance criterion because of local hardware, credentials, or missing dependencies, state that explicitly in the comment.

## PR Standard

Every PR should explain:
- What changed
- Why
- Linear issue
- Acceptance criteria checked
- Screenshots, Loom, or preview URL when relevant
- Risk
- How to test
- What was intentionally not done
- Agent involvement
- Follow-up issues created

## PR Review Standard

Review against the linked Linear issue only.

Look for:
- Acceptance criteria gaps
- Bugs
- Broken data flow
- Unnecessary scope expansion
- Security issues
- Bad abstractions
- Missing loading/error states
- Code that will be hard for future agents to modify

Do not suggest unrelated improvements unless they are severe.

Return review feedback in three groups:
1. Must fix before merge
2. Should fix soon
3. Safe to merge

## Verification

Use the narrowest useful verification command for the task.

If a broad check is already known to have unrelated failures, say that plainly in the PR and include the targeted checks that passed.
