## Prototype work

- New feature/problem prototyping happens in `src/prototyp/<feature>/` —
  read `src/prototyp/CLAUDE.md` for the workflow before starting one.

## Code Preferences

- Prefer JAX (`jax.numpy`) over NumPy for array/numeric code. Use plain
  NumPy only where required (host-only I/O, interop with libraries that
  need concrete NumPy arrays, e.g. file writing, PLY/text parsing).

## Docstring style

Use **Google-style** docstrings. Online reference:
https://google.github.io/styleguide/pyguide.html#383-functions-and-methods

Every function/method docstring starts with a one-line summary of its
purpose, then `Args:`, `Returns:`, and `Raises:` (only if it can raise).
Example:

    """Connects to the next available port.

    Args:
      minimum: A port value greater or equal to 1024.

    Returns:
      The new minimum port.

    Raises:
      ConnectionError: If no available port is found.
    """

- Begin with the purpose/description — one or more lines summarizing what the
  function does (a short summary line first, optionally followed by more
  detail). It does not have to be a single line.
- `Args:` one entry per parameter, `name: description`.
- `Returns:` describe the return value (and shape/dtype for arrays).
- `Raises:` only if the function raises (list each exception type).
- Classes/modules: a purpose line; classes may add `Attributes:`.

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/` in this repo;
GitHub is used for PRs and code review only. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, `wontfix`), recorded as `Status:` lines in issue files.
See `docs/agents/triage-labels.md`.

### Domain docs

Multi-context: `CONTEXT-MAP.md` at the root points to per-context `CONTEXT.md`
files (`src/r2dreamer/`, `src/vggt/`) plus the shared root glossary.
See `docs/agents/domain.md`.
