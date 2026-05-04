You are implementing a single GitHub issue end-to-end on the
Master-Thesis-3D-VLA repository.

## Issue

- Number: #${ISSUE_NUM}
- Title: ${ISSUE_TITLE}
- Branch: you are already checked out on `${BRANCH}` (off latest `origin/main`).

## Body

${ISSUE_BODY}

## Your task

1. Read `CLAUDE.md` and any `docs/agents/*.md` relevant to the area you'll
   touch before writing code. Honour the project's "no overengineering" and
   "no unnecessary comments" rules from `CLAUDE.md`.
2. Implement the change in the smallest correct way that satisfies the
   acceptance criteria.
3. Run any tests / lints / type-checks that apply to the files you touched.
4. Commit your work with a Conventional-Commits style message; reference the
   issue in the body (`Refs #${ISSUE_NUM}`).
5. **Do NOT** push, open a PR, close the issue, or change labels — the outer
   loop owns those.
6. If a `gh` or `git` command fails with what looks like a transient error
   (network, rate-limit, lock contention), retry exactly once after a 5-second
   wait before treating it as a real failure.
7. If you genuinely cannot complete the issue (acceptance criteria
   underspecified, missing dependency, blocked by another open issue), do
   **not** create a partial commit. Leave the working tree clean and exit;
   the outer loop will detect the missing commit and halt with the issue
   labelled `in-progress` for human triage.

When you have committed your work, you are done. Exit normally.
