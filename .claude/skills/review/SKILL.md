---
name: review
description: Review code for bugs, logic errors, security issues, and convention violations. Use after each engineer phase or standalone after any code change.
---

Review the recent code changes. You are the reviewer — find problems, not style nits.

## When invoked

Examine the changed files (use `git diff` to identify them) for:

- **Logic errors** — wrong conditions, off-by-one, missing edge cases
- **Bugs** — null/None dereference, uninitialized variables, wrong return types
- **Security** — injection, unsafe deserialization, hardcoded credentials
- **JAX-specific** — in-place mutations (illegal in JAX), missing `jax.jit` annotations, PRNG misuse, shape mismatches
- **Convention violations** — deviations from codebase patterns in `modules/`

For each issue found, report:
1. File and line number
2. What's wrong
3. Severity: **blocker** / **warning** / **nit**
4. Suggested fix

Fix all **blockers** immediately. Present **warnings** and **nits** to the user for decision.

## After review

Run tests to confirm nothing broke:
```bash
uv run pytest
```

Print a summary:
- Issues found and fixed
- Test results
- Ready for next phase (if called from `/engineer`) or done
