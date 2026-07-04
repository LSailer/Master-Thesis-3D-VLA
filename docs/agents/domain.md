# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout

This is a multi-context repo. `CONTEXT-MAP.md` at the repo root lists the contexts and points at their `CONTEXT.md` files.

```
/
├── CONTEXT-MAP.md
├── CONTEXT.md                 ← shared cross-context glossary
├── docs/adr/                  ← system-wide decisions
└── src/
    ├── r2dreamer/
    │   ├── CONTEXT.md         ← agent/world-model language (created lazily)
    │   └── docs/adr/          ← context-scoped decisions
    └── vggt/
        ├── CONTEXT.md         ← feature-extractor language (created lazily)
        └── docs/adr/
```

- Read `CONTEXT-MAP.md` first, then each `CONTEXT.md` relevant to the topic. The shared root glossary applies everywhere.
- Read `docs/adr/` for system-wide decisions, and `src/<context>/docs/adr/` for context-scoped ones, that touch the area you are about to work in.
- If any of these files do not exist, proceed silently. Do not suggest creating them upfront.

The producer skills create these files lazily when project terms or decisions have actually been resolved.

## Use the Glossary's Vocabulary

When your output names a domain concept in an issue title, refactor proposal, hypothesis, or test name, use the term as defined in the relevant `CONTEXT.md`.
Do not drift to synonyms the glossary explicitly avoids.

If the concept you need is not in the glossary yet, either reconsider whether you are inventing language the project does not use, or note the gap for a future domain-doc pass.

## Flag ADR Conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding the decision.
