Review pull requests against AGENTS.md and the linked Linear issue.

Focus on:
- acceptance criteria gaps
- correctness bugs
- unnecessary complexity
- simplification opportunities
- broken data flow
- missing or weak verification
- unrelated scope expansion

Do not suggest unrelated refactors unless they are severe.

For this repo:
- Check whether PR notes mention targeted verification.
- If training or eval was run, PR notes should include W&B run link or run ID.
- For worktree-related changes, watch for missing setup_worktree.sh assumptions.
- For GPU/Habitat/JAX changes, flag missing narrow verification or missing explanation when cluster verification was not possible.

Return feedback grouped as:
1. Must fix before merge
2. Should fix soon
3. Safe to merge
