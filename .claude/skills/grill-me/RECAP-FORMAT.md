# Recap Format

A recap captures the *journey* of a grill session — what was decided, why, and what was rejected. It complements (does not duplicate) the destillated end-products in `docs/wiki/methods/` and `docs/wiki/experiments/`.

## Location

`docs/wiki/recaps/<YYYY-MM-DD>-<topic-slug>.md`

One file per session. If the same topic returns later, write a new dated recap rather than editing the old one — recaps are timestamped audit trails, not living documents.

## Structure

```md
# Recap: {Title}

**Date:** YYYY-MM-DD
**Source:** /grill-me session
**Topic:** {one-line topic}
**Outcome:** {one-line headline result, e.g. "Two-skill design + bash glue, no new orchestrator"}

## Kontext

{2-4 sentences: what triggered the session, what constraints from existing assets shaped the discussion. Mention pre-existing files, prior decisions, or external triggers — anything a reader needs to make sense of the decisions below.}

## Design Decisions

### D1 — {Decision title in imperative form}
**Decision:** {what was chosen, in one sentence}
**Why:** {the reason that made this win — usually a constraint or a trade-off}
**Rejected alternatives:** {only if non-obvious; skip if there were no real alternatives}
**How to apply:** {one line on how the decision shapes downstream work — file paths, commands, or behaviour}

### D2 — ...

(One subsection per decision. Number them so the Eval-Pass table can reference them.)

## Eval-Pass

| Decision | Eval mode | Concrete check |
|---|---|---|
| D1 | auto / manual / N/A | {what file / metric / human inspection} |
| D2 | ... | ... |

If a row reads "N/A", that's a deliberate acknowledgement that the decision is architectural / naming / non-measurable. Don't force a check.

## Open Questions / Parked

- **OQ1:** {question that was raised but deferred} — {when it should be revisited}
- **OQ2:** ...

## Deliverables (next steps)

1. {Concrete file or skill to create / modify}
2. ...
```

## Rules

- **Be terse.** Each decision is one paragraph, not a page. The reader should be able to skim D1–DN in under two minutes.
- **Lead with the decision, not the discussion.** "**Decision:** Sonnet 4.6 for verify and report" — not "We talked about which model to use and considered Opus..."
- **Capture the *why*, not the deliberation.** A future re-grill needs to know why the choice was made, not which questions were asked along the way.
- **Reject mode is for non-obvious alternatives only.** If the rejected option is what most readers would otherwise assume ("why not just use Python?"), record it. If there was no real alternative, skip the line.
- **Eval-Pass is a table, not prose.** It exists to be machine-skimmable for re-grills and for downstream pipeline runs.
- **Open Questions are explicit parking lots.** Anything that came up and was deferred goes here, not into Decisions. This is the bridge to the next grill session.

## Numbering

Decisions: `D1, D2, ...` — sequential, never renumbered if a decision is later removed (cross out instead, or add an OQ noting the reversal).

Open Questions: `OQ1, OQ2, ...` — same convention.

## When NOT to write a recap

- A grill session that resolved a single yes/no question — chat-summary is enough.
- A grill that ended with "let's think more, no decisions made" — there's nothing to capture.
- A grill on something that's already covered by an existing recap or methods page — update the existing artefact instead.

When in doubt, ask the user before writing.
