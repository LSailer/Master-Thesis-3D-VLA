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

- Begin with the purpose/description — one lines summarizing what the
  function does (a short summary line first, optionally followed by more
  detail). It does not have to be a single line.
- `Args:` one entry per parameter, `name: description`.
- `Returns:` describe the return value (and shape/dtype for arrays).
- `Raises:` only if the function raises (list each exception type).
- Classes/modules: a purpose line; classes may add `Attributes:`.

