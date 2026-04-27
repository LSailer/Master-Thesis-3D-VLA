---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

# Grill Me

## Quick start

> **User:** "Grill me on adding caching to the API."
> **You:** "Q1 — What stays cached? Recommendation: per-user query results, 60s TTL. Reason: balances freshness against hit-rate, and 60s matches the upstream poll interval. Agree, or different?"

One question at a time. Always include your recommended answer with reasoning. Wait for the user before continuing.

## Workflows

### Phase A — Design grilling

For each branch of the design tree:

- [ ] Read `docs/wiki/` first; skip questions whose answers are already documented
- [ ] Explore the codebase instead of asking when the answer lives in code
- [ ] WebSearch when external best practice or pitfalls are unclear — don't guess
- [ ] Surface the question with your recommended answer + reasoning
- [ ] Wait for confirmation or correction before moving on
- [ ] Resolve dependencies between decisions before opening new branches

End Phase A when every branch is resolved and the user confirms "design is settled".

### Phase B — Eval pass

Walk back through every decision from Phase A:

- [ ] Ask **auto / manual / N/A** — what is the concrete check for this decision?
- [ ] If a decision has no testable outcome, return to Phase A and sharpen the design — don't invent eval methods for unfalsifiable goals

End Phase B when every decision has an explicit eval mode.

### Output

When both phases are done (or the user says "speichern" / "save"):

- [ ] Propose path: default `docs/wiki/recaps/<YYYY-MM-DD>-<topic-slug>.md`, ask the user to confirm or override
- [ ] Write the recap using [RECAP-FORMAT.md](RECAP-FORMAT.md)
- [ ] Append a one-liner to `docs/wiki/log.md` linking to the recap
- [ ] Do **not** modify `docs/wiki/index.md` — index is reserved for destillated end-products (methods/, experiments/), not journey artefacts

If the user declines to save, print a structured chat summary and skip the file writes.
