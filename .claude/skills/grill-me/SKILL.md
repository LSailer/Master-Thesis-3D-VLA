---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

Interview the user relentlessly about every aspect of the plan until you reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one.

## Rules

- **Ask one question at a time.** Wait for the answer before continuing.
- **For each question, provide your recommended answer** with reasoning so the user can simply agree or correct you.
- **If a question can be answered by exploring the codebase, explore instead of asking.**
- **If you're unsure or feel a knowledge gap, look it up before asking** — read `docs/wiki/`, grep the codebase, run `WebSearch` for external best-practice / pitfalls / prior art. Do not guess silently.
- **Read `docs/wiki/` first** for what's already known. Skip questions whose answers are already in `methods/`, `experiments/`, `meetings/`, `recaps/`. Surface what you found instead of re-grilling.

## Two-phase structure

Every grill session has two phases. Do not collapse them.

### Phase A — Design grilling

Walk the design tree. For each branch:
- State the question and your recommendation.
- Wait for confirmation or correction.
- Resolve dependencies between decisions before moving on.

End Phase A when all design branches are resolved AND the user confirms "design is settled".

### Phase B — Eval pass

Now walk back through every decision made in Phase A and ask, per decision:

> "How will we evaluate that this works — automatically, manually together, or not applicable?"

For each:
- **Auto:** define the concrete check (file path, metric threshold, test command).
- **Manual:** state what the human will look at and what counts as pass.
- **N/A:** acknowledge architectural/naming decisions that aren't measurable; don't force a check.

If the eval pass surfaces an untestable success criterion, that is a signal to **return to Phase A** and sharpen the design — not to invent an eval method for an unfalsifiable goal.

End Phase B when every decision has an explicit eval mode.

## Output: write a recap

When both phases are done (or when the user explicitly says "speichern" / "save"):

1. **Propose a path** for the recap file. Default convention: `docs/wiki/recaps/<YYYY-MM-DD>-<topic-slug>.md`. Ask the user to confirm or override the path.
2. **Write the recap** with these sections:
   - `# Recap: <Title>` + frontmatter-style header (Date, Source, Topic, Outcome)
   - `## Kontext` — what triggered this session, constraints from existing assets
   - `## Design Decisions` — one numbered subsection per decision with **Decision**, **Why**, **How to apply**, and rejected alternatives where relevant
   - `## Eval-Pass` — table mapping each decision to auto/manual/N/A with the concrete check
   - `## Open Questions / Parked` — anything explicitly deferred
   - `## Deliverables (next steps)` — concrete files/skills/issues to create
3. **Append a one-liner to `docs/wiki/log.md`** at the top:
   ```
   ## [YYYY-MM-DD] grill | <Topic> | source: /grill-me → recaps/<file>.md
   <2-3 sentences summarizing key decisions>
   ```
4. **Do NOT modify `docs/wiki/index.md`.** That file indexes destillated end-products (methods/, experiments/) — recaps are journey, not destination. Index gets updated when the destillated artifact is created downstream.

If the user declines to save (says "no, just chat-summary"), print a structured summary in the chat instead and skip steps 1–4.
