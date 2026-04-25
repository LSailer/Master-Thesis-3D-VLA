# Wiki Index

## Experiments

- [L1 Rerun — Buffer Fix + Step Penalty](experiments/l1-rerun-buffix.md) — L1 with buffer fix, 75% SR (+8pp over original), step penalty improves SPL
- [L2 — 1 House, 6 Goals](experiments/l2-1house-6goals.md) — 6 goal categories, 36% avg SR, goal difficulty hierarchy driven by navigation complexity
- [L3 — 10 Houses, Chair Only](experiments/l3-10houses-chair.md) — 10-house generalization, 32% SR, still 8x above random
- [L1 Baseline — 2.4M Steps](experiments/l1-baseline-2.4m.md) — Original L1 (1 house, chair), 67% SR, world model learns but overfits
- [Random Baseline — L1 Curriculum](experiments/random-baseline-l1.md) — Uniform-random agent on L1 (1 house, chair), 3.84% SR, performance floor
- [Baseline 2.4M — All Scenes](experiments/baseline-2.4m-all-scenes.md) — No goal conditioning, 145 scenes, 2.36% SR (random-level), no learning

## Methods

- [Claude Code Workflow](methods/workflow.md) — Full pipeline, skill inventory, TDD rules, wiki conventions
- [World Model Training Loop](methods/world-model-training-loop.md) — How 64-step windows, 15-step imagination, and full episodes fit together
- [Training Orchestration](methods/training-orchestration.md) — Unified buffer, Trainer module, ObsAdapter pattern (RFC #68)
- [L4 Pipeline Profiling](methods/l4-profiling.md) — VGGT vs CNN per-phase timing on H100; vggt_forward = 168ms p50 dominates; PyTorch↔JAX boundary is 0.4ms (not the bottleneck)
- [VGGT JAX Streaming](methods/vggt-jax-streaming.md) — Padded 3-tuple KV-cache + jit pattern; closes JAX vs PT gap from 24× to 1.9×; #80 (aggregator) + #81 (camera_head)
- [Launcher Refactor](methods/launcher-refactor.md) — Encoder ABC + per-level shims + L1-L4 test pyramid; 3-phase migration absorbing #85 + #52, archiving dormant PyTorch scripts
- [Phased Orchestration Pattern](methods/phase-orchestration.md) — Opus-orchestrator + 1-Sonnet-subagent-per-phase recipe with strict file whitelists, trust-but-verify gates, and end-to-end smoke as required exit; codified from the #85 launcher refactor session
- [VGGT → R2Dreamer Call Chain](methods/vggt-r2dreamer-callchain.md) — End-to-end data flow for one env step when `encoder_type="vggt"`; 5 high-leverage breakpoints for debugging

## Meetings

- [2026-03-03 Braun](meetings/2026-03-03-braun.md) — Pivot from diffusion policy to world models, research question defined

## Research

*No research pages yet.*
