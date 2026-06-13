# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout

This is a single-context repo.

- Read `CONTEXT.md` at the repo root when it exists.
- Read `docs/adr/` for architectural decisions that touch the area you are about to work in.
- If these files do not exist, proceed silently. Do not suggest creating them upfront.

The producer skills create these files lazily when project terms or decisions have actually been resolved.

## Use the Glossary's Vocabulary

When your output names a domain concept in an issue title, refactor proposal, hypothesis, or test name, use the term as defined in `CONTEXT.md`.
Do not drift to synonyms the glossary explicitly avoids.

If the concept you need is not in the glossary yet, either reconsider whether you are inventing language the project does not use, or note the gap for a future domain-doc pass.

## Flag ADR Conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding the decision.
