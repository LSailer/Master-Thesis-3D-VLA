# ADR 0001: Use `ready-for-agent` (not `AFK`+`ready`) as the agent-loop picker label

**Status**: Accepted  •  **Date**: 2026-05-04

## Context

The repo has two overlapping vocabularies for "this issue is ready for an
autonomous agent":

1. **`AFK` + `ready`** — used in practice (e.g. issue #116) and matched by
   the prior `scripts/run_ralph.sh` dispatcher.
2. **`ready-for-agent`** — the canonical mattpocock triage label documented
   in `docs/agents/triage-labels.md`.

When introducing `scripts/agent_loop.sh`, we had to pick one as the
single source of truth for which issues the loop will pop.

## Decision

`agent_loop.sh` recognises **only `ready-for-agent`**. The `AFK` label is
demoted to its documented role: an orthogonal *execution-mode* tag (autonomous
vs human-in-the-loop), not a picker signal.

A one-time migration relabels existing `AFK`+`ready` issues to
`ready-for-agent`. Project-specific guidance for `to-issues` (apply
`ready-for-agent` directly to AFK slices, skipping `needs-triage`) lives
in `docs/agents/agent-loop.md` under "Feeding the loop", not in
`docs/agents/issue-tracker.md` — the latter is reserved for canonical
mattpocock conventions.

## Consequences

**Positive**

- The picker query is one label, not a conjunction — fewer ways to
  mis-tag and silently skip an issue.
- Documentation and reality agree. `docs/agents/triage-labels.md` already
  said `AFK`/`HITL` are orthogonal to triage state; we now enforce that.
- A single tracker query (`gh issue list --label ready-for-agent`) tells
  any operator exactly what the loop will pick up next.

**Negative**

- Existing issues with `AFK`+`ready` need a one-time relabel.
- Any external automation or muscle memory referring to `AFK`+`ready` as
  the trigger pair is now wrong.
- The choice is hard to reverse cheaply: re-introducing `AFK`+`ready` later
  would require migrating both labels back across all open issues.

## Alternatives considered

- **Accept either** (`AFK`+`ready` *or* `ready-for-agent`). Forgiving during
  transition, but the picker becomes the source of truth and the docs drift
  again. Rejected for the same reason it appealed: implicit dual vocabulary.
- **Keep `AFK`+`ready`, retire `ready-for-agent` from the docs.** Simpler
  for #116, but loses alignment with the upstream mattpocock skill set we
  installed and would put `AFK` back in the triage axis it doesn't belong on.
