# QA Review

Human-in-the-loop review workflow for PRs created by ralph.

## Trigger

User says `/qa-review` to start reviewing pending PRs.

## Process

### 1. List PRs awaiting review

```bash
gh pr list --label in-review --json number,title,url,headRefName --template '{{range .}}#{{.number}} {{.title}} ({{.headRefName}}){{"\n"}}{{end}}'
```

If no PRs: print "No PRs awaiting review." and stop.

### 2. For each PR, present a review summary

```bash
# Diff stats
gh pr diff <number> --stat

# Full diff (condensed)
gh pr diff <number>

# Check results
gh pr checks <number>

# Linked issue
gh pr view <number> --json body -q .body
```

Present to user:
- **PR title and number**
- **Linked issue**: extracted from "Closes #N" in body
- **Diff summary**: files changed, insertions, deletions
- **Key changes**: summarize what the diff does (read the diff)
- **Test status**: pass/fail from checks
- **Concerns**: flag anything suspicious (large files, missing tests, style issues)

### 3. User decides

Prompt user with options:

**[A] Approve and merge**
```bash
gh pr merge <number> --squash --delete-branch
gh issue close <linked_issue>
```
Move issue to Done column if project board configured.

**[C] Request changes**
```bash
gh pr review <number> --request-changes --body "<user's feedback>"
```
Remove `in-review` label, add `in-progress`:
```bash
gh pr edit <number> --remove-label in-review --add-label in-progress
gh issue edit <linked_issue> --remove-label in-review --add-label in-progress
```
Task goes back into ralph's queue on next loop.

**[R] Reject**
```bash
gh pr close <number>
gh issue edit <linked_issue> --remove-label in-review --add-label rejected
gh issue comment <linked_issue> --body "<user's rejection reason>"
```
Optionally create follow-up issues if the approach needs rethinking.

**[S] Skip** — move to next PR without action.

### 4. Continue

After each decision, move to the next PR in the list. Continue until all reviewed or user says stop.

### 5. Summary

```
QA Review Summary
═════════════════
Reviewed:  <N> PRs
Approved:  <list>
Changes:   <list>
Rejected:  <list>
Skipped:   <list>
```

## Rules

- Always show the diff before asking for a decision
- Always show test/check results
- Never auto-approve — this is a human decision point
- Squash merge to keep main history clean
- Delete branch after merge
- Update issue labels to reflect current state
- If checks are failing, highlight this prominently before asking for decision
