# Prototype workspace — feature-folder workflow

One folder per feature/problem: `prototyp/<feature>/`.

## Starting a new feature/problem

Create `prototyp/<feature>/` containing:

- `GOAL.md` — goal, hypothesis, planned approach
- `PROBLEMS.md` — open problems, dead ends, research notes (append as they come up)
- `HANDOFF.md` — session-to-session handoff state
- experiment scripts and their helpers, all *inside* the folder

No shared helper module across features — duplicate small utilities into each
folder instead of coupling features together.

## Lifecycle: folders are throwaway

When a problem is **solved**, do not delete the folder yourself. Instead:

1. Propose what should be distilled into `docs/notes/<topic>.md`
   (results, decisions, reusable insights, numbers worth citing in the thesis).
2. Let the user confirm the extraction; the user deletes the folder afterwards.
3. Code that graduates to "real" moves to `src/r2dreamer/` (or the proper
   package) — it never stays here.

## Rules

- Treat other features' folders as read-only unless asked.
- No generated outputs here; write to `outputs/prototype/<feature>/`.
- Tests for prototype code go in `tests/prototyp/<feature>/`.
