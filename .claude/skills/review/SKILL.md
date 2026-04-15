---
name: review
description: Review and simplify code after /engineer completes. Finds bugs, logic errors, security issues, then simplifies for clarity. Use before /qa or standalone after any code change.
---

Review the recent code changes in two passes: first find issues, then simplify.

## Pass 1 — Review (find problems)

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

## Pass 2 — Simplify (clean code)

After issues are resolved, review the same changed files for:

- **Dead code** — unused imports, unreachable branches, commented-out code
- **Duplication** — repeated logic that should be extracted
- **Complexity** — overly nested conditions, functions doing too much
- **Naming** — unclear variable/function names
- **Unnecessary abstractions** — helpers or wrappers that add indirection without value

Apply simplifications directly. Keep changes minimal — only simplify what was recently changed, not surrounding code.

## After both passes

Run tests to confirm nothing broke:
```bash
uv run pytest
```

Print a summary:
- Issues found and fixed (Pass 1)
- Simplifications applied (Pass 2)
- Test results
- Ready for `/reporter` (if experiment results exist) or done
