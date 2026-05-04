You are the review pass for issue #${ISSUE_NUM} on branch `${BRANCH}`.
The implement pass has already committed work to this branch.

## Your task

1. Run the project's review skill on the diff between this branch and
   `origin/main`. Prefer:
       /pr-review-toolkit:review-pr
   If that skill is unavailable, perform an equivalent review using the
   `pr-review-toolkit:code-reviewer` and `pr-review-toolkit:silent-failure-hunter`
   subagents on the same diff.
2. If the review surfaces *blocker-level* issues (correctness bugs, broken
   tests, security problems, violations of the rules in `CLAUDE.md`), **fix
   them and commit the fix to this branch**. Re-run the review on the new
   diff. Iterate until no blocker-level issues remain.
3. Non-blocker style nits should be ignored — the implement pass already
   honours `CLAUDE.md`'s "no overengineering" rules; do not relitigate.

## Verdict file

When review is complete, write a single line to `${VERDICT_FILE}`:

- `APPROVED` — the diff is ready for human review.
- `BLOCKED: <one-line reason>` — the review found something you could not fix
  (e.g. requires architectural decision, missing context, intentional design
  question for a human).

Do NOT push, open a PR, or close the issue. The outer loop owns those steps.

When the verdict file is written, exit normally.
