# Context Map

This repo has multiple domain contexts. Read the shared glossary first, then the
context(s) relevant to your task. Consumer rules live in `docs/agents/domain.md`.

| Context   | Glossary                   | Scope                                                                                  |
| --------- | -------------------------- | -------------------------------------------------------------------------------------- |
| Shared    | `CONTEXT.md`               | Cross-context ObjectNav project language (Observation Preparation, Encoder Module, …)   |
| r2dreamer | `src/r2dreamer/CONTEXT.md` | Agent/world-model language: Dreamer, RSSM, replay, training loop (created lazily)       |
| vggt      | `src/vggt/CONTEXT.md`      | Feature Extractor language: VGGT, KV cache, house-points context (created lazily)       |

System-wide ADRs live in `docs/adr/`; context-scoped ADRs in `src/<context>/docs/adr/`.
