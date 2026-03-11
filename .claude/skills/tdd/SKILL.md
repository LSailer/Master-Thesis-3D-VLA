# TDD Red-Green-Refactor

Strict test-driven development: one test at a time, vertical slices, never batch.

## Trigger

User says `/tdd` or agent invokes TDD skill with an issue reference.

## Inputs

- `issue`: GitHub issue number or acceptance criteria text
- `file`: (optional) target test file path

## Process

### 1. Read acceptance criteria

```bash
gh issue view <issue> --json body -q .body
```

Extract testable requirements. Order them by dependency (foundational first).

### 2. RED — Write ONE failing test

- Create or append to `tests/test_<module>.py`
- Write exactly ONE test function targeting ONE acceptance criterion
- Name it descriptively: `test_<module>_<behavior>`
- Run the test and confirm it FAILS:

**CPU test** (default):
```bash
uv run pytest tests/<file> -x -q -k "<test>"
```

**GPU test** (detected by `@pytest.mark.gpu`, or imports of `jax`, `habitat_sim`, `torch.cuda`):
```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00 uv run pytest tests/<file> -x -q -k "<test>"
```

- If the test passes unexpectedly: the requirement is already met. Delete the test or mark it and move to the next criterion.
- If the test errors for reasons unrelated to the feature (import errors, missing fixtures): fix the test setup first, do NOT count this as a failure.

### 3. GREEN — Write MINIMUM code to pass

- Edit only `src/` files
- Write the absolute minimum code to make the failing test pass
- No premature abstractions, no extra features, no "while I'm here" changes
- Run the test again and confirm it PASSES
- If it fails: fix the implementation (not the test). Increment failure counter.

### 4. REFACTOR — Clean up

- Remove duplication, improve names, extract functions if needed
- Run ALL tests for the module to confirm no regressions:
```bash
uv run pytest tests/<file> -x -q
```
- If any test breaks during refactor: undo the refactor, keep the green state

### 5. COMMIT

```bash
git add -A
git commit -m "green: <test_name> (#<issue>)"
```

### 6. LOOP

Go back to step 2 with the next acceptance criterion. Continue until all criteria are covered.

## Error Budget

Track consecutive failures on the SAME test (not across different tests).

- **1st failure**: re-read the error, fix implementation
- **2nd failure**: re-read the test, check assumptions, read related source
- **3rd failure**: STOP. Add `blocked` label to the issue:
  ```bash
  gh issue edit <issue> --add-label blocked
  gh issue comment <issue> --body "Blocked: 3 consecutive failures on \`<test_name>\`. Error: <last_error_summary>"
  ```
  Move to the next acceptance criterion or stop if none remain.

## Rules

- ONE test at a time. Never write multiple tests then implement.
- Vertical slices: each test+implementation pair delivers a thin end-to-end slice.
- Never skip a failing test. Fix it or escalate after 3 failures.
- Never modify tests to make them pass (except genuine test bugs).
- Commit after every green. Small commits are good commits.
- GPU detection heuristic: scan the test file and target source for `jax`, `habitat_sim`, `torch.cuda`, or `@pytest.mark.gpu`. If any match, use `srun`.

## Output

After all criteria are done (or blocked), print a summary:

```
TDD Summary (#<issue>)
  Green:   <N> tests passing
  Blocked: <M> tests (if any)
  Commits: <K>
```
