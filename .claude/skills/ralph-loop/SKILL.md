# Ralph Loop

Core autonomous execution loop: scan for ready tasks, implement via TDD in worktrees, create PRs.

## Trigger

User says `/ralph-loop` or ralph agent starts execution.

## Inputs

- `max_tasks`: maximum tasks to complete before stopping (default: 5)
- `time_limit`: maximum minutes to run (default: 120)

## Process

Record start time. Initialize `tasks_completed = 0`.

### 1. SCAN — Find available tasks

```bash
gh issue list --label AFK --label ready --state open --json number,title,body,labels -q '.[] | {number, title, labels: [.labels[].name]}'
```

Also check in-progress tasks that might have been interrupted:
```bash
gh issue list --label AFK --label in-progress --state open --json number,title,body -q '.[] | {number, title}'
```

Prefer in-progress tasks (resume interrupted work) over ready tasks.

### 2. FILTER — Skip blocked tasks

For each candidate, check the issue body for "Blocked By" section. Extract referenced issue numbers. Check if those issues are still open:

```bash
gh issue view <dep_number> --json state -q .state
```

Skip any task where a blocking issue is still `OPEN`.

### 3. PICK — Select lowest issue number

From unblocked candidates, pick the one with the lowest issue number (most foundational).

If no tasks available: print summary and stop.

### 4. CLAIM — Mark as in-progress

```bash
gh issue edit <N> --remove-label ready --add-label in-progress
gh issue comment <N> --body "Ralph picking up this task."
```

### 5. SETUP — Create worktree

```bash
git fetch origin main
git worktree add .claude/worktrees/issue-<N> -b issue-<N> origin/main
cd .claude/worktrees/issue-<N>
```

If worktree already exists (resumed task), just cd into it.

### 6. EXECUTE — Run TDD

Invoke the `/tdd` skill with the issue number:
- Read acceptance criteria from the issue
- Execute red-green-refactor loop
- Each green test gets a commit: `green: <test_name> (#<N>)`

### 7. PR — Create pull request

After all acceptance criteria are green (or max blocked):

```bash
git push -u origin issue-<N>
gh pr create \
  --title "<issue title>" \
  --body "$(cat <<'EOF'
Closes #<N>

## Changes

<bullet list of what was implemented>

## Test Results

<paste test output summary>
EOF
)"
```

### 8. UPDATE — Move on board

```bash
gh issue edit <N> --remove-label in-progress --add-label in-review
gh pr edit <pr_number> --add-label in-review
```

### 9. CLEANUP — Remove worktree

```bash
cd /path/to/repo
git worktree remove .claude/worktrees/issue-<N>
```

### 10. LOOP — Check stop conditions

```
tasks_completed += 1
```

**Stop if ANY**:
- `tasks_completed >= max_tasks`
- elapsed time >= `time_limit` minutes
- no more AFK tasks available

Otherwise: go to step 1.

## Error Handling

### Test failures (from TDD skill)
- 3 consecutive failures on same test → TDD skill adds `blocked` label
- Ralph skips to next task (does NOT stop the loop)

### PR creation failure
- Retry once after 10 seconds
- If second attempt fails: log error, skip cleanup, continue loop

### Worktree conflicts
- If branch already exists: `git checkout issue-<N>` instead of creating new
- If worktree path exists: `git worktree remove` first, then recreate

### Git push failures
- `git pull --rebase origin issue-<N>` then retry push
- If rebase conflicts: abort rebase, add `blocked` label, skip task

## Output

After loop ends, print summary:

```
Ralph Loop Summary
══════════════════
Tasks completed: <N>/<max>
Time elapsed:    <M> minutes
PRs created:     <list of PR numbers>
Blocked:         <list of blocked issue numbers>
Remaining AFK:   <count>
```

## Rules

- NEVER push to main directly
- NEVER force push
- ALWAYS work in worktrees, never in the main working tree
- ALWAYS create PRs for human review
- ALWAYS commit after each green test
- ONE task at a time, sequential execution
- Clean up worktrees after PR creation
- Respect time limits — check before starting each new task
