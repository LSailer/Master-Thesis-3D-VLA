# PR Workflow

Use this for GitHub PRs linked to implementation issues in the local tracker
(`.scratch/<feature>/issues/`, see `issue-tracker.md`).

## Before Opening Or Updating A PR

- Run the narrowest useful verification for the files changed.
- Review the diff for unrelated changes.
- Follow `.github/pull_request_template.md` when it exists.
- Update the linked issue file with what changed, acceptance criteria checked, verification run, known blockers, and intentional omissions (append under `## Comments`).
- If local hardware, credentials, or missing dependencies block an acceptance criterion, state that explicitly.

## PR Body

Every PR should explain:

- What changed
- Why
- Linked issue file path (e.g. `.scratch/<feature>/issues/02-<slug>.md`)
- Acceptance criteria checked
- Screenshots, Loom, or preview URL when relevant
- Risk
- How to test
- What was intentionally not done
- Agent involvement
- Follow-up issues created

## Review Standard

Review against the linked issue file only.

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
