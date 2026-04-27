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
- [Cross-Correlation Matrix](methods/cross-correlation-matrix.md) — Barlow Twins–form regularizer used here to align RSSM-projected feature with frozen VGGT embedding (not contrastive learning)
- [OpenClaw Verdict — shelved](methods/openclaw-verdict.md) — End-state 2026-04-27: install completes, plugin-bug fixed in 4.25, but `claude -p` cold-spawn = 20–30 s/message. Structural, unfit for Slack-from-phone use case. Companion to install playbook + skill audit
- [OpenClaw Install Playbook](methods/openclaw-install.md) — BWUniCluster Phase 3 setup: nvm + claude-cli backend (Claude Max, no API charges) + Slack Socket Mode bot `clusterbot`, persisted via tmux with login-node-roulette workaround and a daemon-stability gate
- [OpenClaw Skill Audit](methods/openclaw-skill-audit.md) — Classification of every active Claude Code skill (repo + plugin) against OpenClaw's chat-only constraints: redundant / partial / synergy / unchanged, with action items for safe disables
- [`loss/dyn` ≡ `loss/rep` is cosmetic — JAX proof](methods/dyn-rep-loss-cosmetic-proof.md) — Synthetic JAX verification that forward bit-equality of the three logged KL metrics is `stop_gradient` symmetry, not a logging bug; backward gradients route correctly to disjoint params with configured scales
- [kl_free per-group — investigated, rejected](methods/kl-free-per-group-fix.md) — Patched per-group floor (matches DreamerV3) prevents latent collapse in mini-smoke, but canonical R2-Dreamer (`external/r2dreamer/rssm.py:222-230`) also sums-then-floors → pre-patch JAX is port-faithful. Decision: keep R2-Dreamer-faithful on main, leave patch unmerged in worktree for future ablation
- [R2Dreamer encoder-drift visualization (L1 VGGT)](methods/r2dreamer-encoder-drift-viz.md) — Linear-probe + temporal-similarity comparison of latents vs VGGT world_points on a matched success (ep7) / near-miss (ep1) pair. Three coherent drift signals show ep1 has higher absolute probe RMSE *and* collapsed similarity range while preserving relational rank — representation-collapse symptom, not topology loss. Surfaces #101 (TF32 non-determinism) + #102 (standalone eval was CNN-only)

## Meetings

- [2026-03-03 Braun](meetings/2026-03-03-braun.md) — Pivot from diffusion policy to world models, research question defined

## Research

*No research pages yet.*

## Recaps

- [2026-04-26 Output Restructure](recaps/2026-04-26-output-restructure.md) — 13 decisions locked: 3-bucket layout, slug-jobid naming, MANIFEST.json contract, Big-Bang migration
