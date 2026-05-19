# AGENTS.md

This repo is worked on by coding agents on bwUniCluster. Follow these rules for Linear issue implementation and review.

## Worktree Setup

When working in a `git worktree` (any directory under `worktrees/`), run once per fresh worktree before any training or eval command:

```
./scripts/setup_worktree.sh
```

This symlinks the shared `data/` and `.venv/` from the main checkout into the worktree. Habitat and other scripts resolve dataset paths relative to CWD, so a worktree without these links fails fast with `FileNotFoundError: data/datasets/...`. The script is idempotent and is a no-op in the main checkout.

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
