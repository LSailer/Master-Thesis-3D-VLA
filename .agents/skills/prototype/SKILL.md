---
name: prototype
description: "Build a scratchpad-only throwaway prototype to answer a design question: a runnable terminal app for state/business-logic questions, or several radically different UI variations."
---

# Prototype

A prototype is **throwaway scratchpad code that answers a question**. The question decides the shape.

## Pick a branch

Identify which question is being answered — from the user's prompt, the surrounding code, or by asking if the user is around:

- **"Does this logic / state model feel right?"** → [LOGIC.md](LOGIC.md). Build a tiny interactive terminal app that pushes the state machine through cases that are hard to reason about on paper.
- **"What should this look like?"** → [UI.md](UI.md). Generate several radically different UI variations, switchable from one scratchpad prototype entrypoint.

The two branches produce very different artifacts — getting this wrong wastes the whole prototype. If the question is genuinely ambiguous and the user isn't reachable, default to whichever branch better matches the surrounding code (a backend module → logic; a page or component → UI) and state the assumption at the top of the prototype.

## Rules that apply to both

1. **Scratchpad only.** Create or edit prototype files only under `scratchpad/prototypes/<prototype-slug>/` or an active `scratchpad/experiments/<experiment-slug>/prototype/`. Do not create or edit production files for prototype work.
2. **Production integration is explicit.** If the prototype answer should enter production, provide a patch/snippet or ask the user to approve a separate production edit. Do not quietly promote prototype code.
3. **One command to run.** Add a scratchpad-local `README.md`, `run.sh`, or equivalent command note. Do not edit the project task runner just to run a prototype.
4. **No persistence by default.** State lives in memory. Persistence is the thing the prototype is _checking_, not something it should depend on. If the question explicitly involves storage, use a scratchpad-local file clearly named `PROTOTYPE-*`.
5. **Skip the polish.** No tests, no error handling beyond what makes the prototype _runnable_, no abstractions. The point is to learn something fast and then delete it.
6. **Surface the state.** After every action (logic) or on every variant switch (UI), print or render the full relevant state so the user can see what changed.
7. **Delete or absorb when done.** When the prototype has answered its question, either delete the scratchpad prototype or fold the validated decision into real code in a separate, approved production change.

## When done

The _answer_ is the only thing worth keeping from a prototype. Capture it somewhere durable (commit message, ADR, issue, or a `NOTES.md` next to the prototype) along with the question it was answering. If the user is around, that capture is a quick conversation; if not, leave the placeholder so they (or you, on the next pass) can fill in the verdict before deleting the prototype.
