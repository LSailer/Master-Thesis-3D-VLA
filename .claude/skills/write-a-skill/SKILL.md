---
name: write-a-skill
description: Create new agent skills with proper structure, progressive disclosure, and bundled resources. Use when user wants to create, write, or build a new skill.
---

# Writing Skills

Adapted from [mattpocock/skills](https://github.com/mattpocock/skills/blob/main/write-a-skill/SKILL.md). Format details in [REFERENCE.md](REFERENCE.md).

## Quick start

> **User:** "Help me write a skill for summarising weekly progress."
> **You:** "OK. Three questions to scope it: (1) input — wandb runs, git log, both? (2) output format — markdown to a file, chat summary, or both? (3) trigger — manual `/weekly-summary`, scheduled, or both? My recommendation: git log + wandb of last 7 days, markdown to `docs/wiki/weekly/<date>.md`, manual trigger. Agree, or different?"

The skill itself is a `/grill-me`-style interview, then a draft, then a review.

## Workflow

### Phase 1 — Gather requirements

- [ ] What task or domain does the skill cover?
- [ ] What specific use cases should it handle?
- [ ] Does it need executable scripts, or instructions only?
- [ ] Any reference materials (docs, prior art, existing skill to mirror)?

### Phase 2 — Draft

- [ ] Create `.claude/skills/<name>/SKILL.md` using the template in [REFERENCE.md](REFERENCE.md)
- [ ] Split content >100 lines into sibling files (`REFERENCE.md`, `EXAMPLES.md`, format files)
- [ ] Add scripts only if the operation is deterministic and would be generated repeatedly

### Phase 3 — Review with user

- [ ] Does this cover the use cases?
- [ ] Anything missing or unclear?
- [ ] Should any section be more or less detailed?

## Review Checklist

After drafting, verify:

- [ ] Description ≤1024 chars, third person, first sentence = what, second = "Use when…"
- [ ] Description includes specific triggers (keywords, contexts, file types)
- [ ] SKILL.md under 100 lines
- [ ] Quick start has a concrete example (dialogue or command)
- [ ] Workflows expressed as checklists, not paragraphs, when steps are sequential
- [ ] References one level deep (SKILL.md → REFERENCE.md, not deeper)
- [ ] No time-sensitive info (dates, version numbers, "currently…")
- [ ] Consistent terminology with the rest of the project

## Existing skills in this repo to mirror

- `domain-model/` — short SKILL.md + `CONTEXT-FORMAT.md` + `ADR-FORMAT.md`
- `grill-me/` — short SKILL.md + `RECAP-FORMAT.md`
- `engineer-team/` — short SKILL.md + `ARGS-FORMAT.md`
