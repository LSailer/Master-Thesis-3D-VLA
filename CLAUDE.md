## Prototype work

- New feature/problem prototyping happens in `src/prototyp/<feature>/` —
  read `src/prototyp/CLAUDE.md` for the workflow before starting one.

## Code Preferences

- Prefer JAX (`jax.numpy`) over NumPy for array/numeric code. Use plain
  NumPy only where required (host-only I/O, interop with libraries that
  need concrete NumPy arrays, e.g. file writing, PLY/text parsing).
