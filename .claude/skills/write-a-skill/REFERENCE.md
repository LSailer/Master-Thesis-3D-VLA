# Skill Authoring Reference

Detailed format and decision rules. Loaded only when writing or editing a skill.

## Skill folder structure

```
.claude/skills/<name>/
├── SKILL.md           # Main instructions (required)
├── REFERENCE.md       # Detailed docs (optional)
├── EXAMPLES.md        # Usage examples (optional)
├── <FORMAT>.md        # Output-format specs (optional, e.g. RECAP-FORMAT.md)
└── scripts/           # Utility scripts (optional)
```

Reference one level deep — `SKILL.md` links to siblings, but a sibling should not link to a third file. Flat beats nested.

## SKILL.md template

```md
---
name: skill-name
description: {Brief capability statement}. Use when {specific triggers}.
---

# Skill Name

## Quick start

{Minimal working example — show one concrete invocation + ideal first response.}

## Workflow

### Phase 1 — {name}
- [ ] {step}
- [ ] {step}

### Phase 2 — {name}
- [ ] {step}

## Advanced features

{Link to siblings: See [REFERENCE.md](REFERENCE.md).}
```

## Description requirements

The description is **the only thing the agent sees** when deciding which skill to load. It is surfaced in the system prompt alongside all other installed skills.

The agent reads these descriptions and picks the relevant skill based on the user's request.

**Goal**: give the agent just enough info to know:
1. What capability this skill provides
2. When/why to trigger it (specific keywords, contexts, file types)

**Format**:
- Max 1024 chars
- Write in third person
- First sentence: what it does
- Second sentence: `Use when {specific triggers}`

**Good**:
```
Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when user mentions PDFs, forms, or document extraction.
```

**Bad**:
```
Helps with documents.
```

The bad example gives the agent no way to distinguish this from other document skills.

## When to add scripts

Add utility scripts when:
- The operation is deterministic (validation, formatting, parsing)
- The same code would be generated repeatedly
- Errors need explicit handling

Scripts save tokens and improve reliability vs generated code.

## When to split files

Split into separate files when:
- SKILL.md exceeds 100 lines
- Content has distinct domains (one format-spec per output type)
- Advanced features are rarely needed (lazy-load by reference)

If you split, name siblings descriptively — `RECAP-FORMAT.md`, `ARGS-FORMAT.md`, not `MORE.md`.

## Frontmatter options

- `name:` (required) — folder-name slug
- `description:` (required) — see Description Requirements above
- `disable-model-invocation: true` (optional) — skill is only triggered when the user types `/<name>` explicitly. Use for skills that should never auto-fire on keyword match (e.g. heavy or destructive operations).
