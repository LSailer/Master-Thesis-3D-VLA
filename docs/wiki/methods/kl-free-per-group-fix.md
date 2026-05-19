---
title: kl_free per-group — investigated, rejected as recipe-drift
tags: [r2dreamer, kl-loss, code-audit, dreamerv3, decision]
date: 2026-04-27
status: investigated-rejected
---

# `kl_free` per-group — investigated, rejected as recipe-drift

**TL;DR:** Patched per-group floor was technically valid (matches DreamerV3, prevents latent collapse) but was a deviation from the canonical R2-Dreamer source `external/r2dreamer/`. Decision: **keep R2-Dreamer-faithful behavior on `main`** to preserve comparability with published R2-Dreamer baselines for the thesis. Patch left unmerged in worktree as a reference for future ablation.

## Bug

The R2-Dreamer JAX agent's `_kl_loss` was applying the `kl_free=1.0` floor to the **summed** KL across all 32 categorical stoch groups, rather than per-group. With 32 groups, the effective per-group floor was `1.0 / 32 ≈ 0.031` nats — three orders of magnitude below the intended 1-nat-per-group floor.

In the L1 CNN smoke run (100k steps), this caused `latent/posterior_entropy` to decay from 2.65 nats (step 250) to **0.20 nats** (step 95k), with the floor never activating because the summed KL stayed comfortably above 1.0 (range 2.85–7.1 nats). At 2.4M steps with that decay rate, latents projected to <0.05 nats — a known DreamerV3 latent-collapse failure mode.

## Patch

`src/r2dreamer/agent.py:645-683` — remove the per-group sum *before* the clamp:

```diff
-    kl_dyn = jnp.sum(_kl(sg_post_probs, sg_post_log, prior_log), axis=-1)
+    kl_dyn = _kl(sg_post_probs, sg_post_log, prior_log)  # (N, C) per-group
     dyn_loss = jnp.maximum(kl_dyn, kl_free)
```
(symmetric for `kl_rep` / `rep_loss`).

Shape change: `dyn_loss` and `rep_loss` go from `(N,)` to `(N, C)`. Downstream `jnp.mean(dyn_loss)` (`agent.py:408-409`, `:631`) reduces over all axes and is shape-transparent — no consumer-side fix needed.

The patch was applied in worktree `worktree-agent-a7b6ae2626d4d3d43` and verified before merge to `main`.

## Upstream comparison

Cross-checked all 9 KL-related code dimensions against `danijar/dreamerv3` (`rssm.py:100-113`, `agent.py:187-189`, `configs.yaml`):

| # | Dimension | R2-Dreamer (patched) | Upstream | Status |
|---|---|---|---|---|
| 1 | KL formula | `sum(p*(logp-logq), -1)` (`agent.py:665`) | same via `OneHot.kl` (`embodied/jax/outs.py`) | **MATCH** |
| 2 | Stop-gradient placement | `KL(sg(post)‖prior)` for dyn, `KL(post‖sg(prior))` for rep | same (`rssm.py:105-106`) | **MATCH** |
| 3 | Free-bits style | clamp `jnp.maximum(KL, free)` | clamp (`rssm.py:108-109`) | **MATCH** |
| 4 | Free-bits axis | per-group, `(N, C)` at clamp time | per-group, `(B, T, stoch)` at clamp time | **MATCH** |
| 5 | Loss reduction | `jnp.mean(dyn_loss)` over `(N, C)` | `v.mean()` over full tensor | **MATCH** |
| 6 | Scale multiplication | applied in `total_loss` only (`agent.py:609-611`) | applied in `total_loss` only (`agent.py:189`) | **MATCH** |
| 7 | Logged metric semantics | `latent/kl_divergence` = per-group post-clamp mean | upstream logs `dyn_ent`/`rep_ent` post-clamp | **MATCH** |
| 8 | `kl_free` value | `1.0` (`config.py:71`) | `1.0` (`configs.yaml`) | **MATCH** |
| 9 | Latent shape | `(N, C=32, K=16)` | `(stoch=32, classes=32)` | **DIVERGE** (issue #94, orthogonal) |

The only divergence is `stoch_discrete=16` vs upstream `32`, tracked separately as issue #94 — independent of this fix.

## Canonical R2-Dreamer comparison

The JAX agent `src/r2dreamer/` was ported from `external/r2dreamer/` (Bansal et al., the R2-Dreamer paper source). Re-checked the 6 KL dimensions against that source — the picture is **not the same as DreamerV3**.

| # | Dimension | R2-Dreamer JAX (patched) | Canonical R2-Dreamer (PyTorch) | Status |
|---|---|---|---|---|
| 1 | Free-bits axis | per-group `(N, C)` (`agent.py:675-681`) | **per-(B,T) total** — `kld(...).sum(-1)` collapses group axis *before* `torch.clip(min=free)` (`rssm.py:224-228`) | **DIVERGE** |
| 2 | Free-bits style | clamp `jnp.maximum(KL, free)` (`agent.py:676,681`) | clamp `torch.clip(x, min=free)` (`rssm.py:227-228`) | **MATCH** |
| 3 | `kl_free` value | `1.0` (`config.py:71`) | `1.0` (`configs/model/_base_.yaml:2`) | **MATCH** |
| 4 | `scale_dyn` / `scale_rep` | `1.0` / `0.1` (`config.py:62-63`) | `1.0` / `0.1` (`configs/model/_base_.yaml:27-28`) | **MATCH** |
| 5 | Stop-gradient placement | `KL(sg(post)‖prior)` for dyn, `KL(post‖sg(prior))` for rep (`agent.py:673-680`) | `kld(post, prior.detach())` for rep, `kld(post.detach(), prior)` for dyn (`rssm.py:224-225`) | **MATCH** |
| 6 | Reduction order | clamp `(N, C)` → `jnp.mean` over all axes (`agent.py:408-409`) | `.sum(-1)` over groups → `torch.clip` → `torch.mean` over `(B, T)` (`dreamer.py:373-374`) | **DIVERGE** (axis order swap) |

**Synthesis.** The patched JAX `_kl_loss` matches `danijar/dreamerv3` (vanilla DreamerV3, 8/8) but **diverges from the canonical R2-Dreamer source it was ported from** in the free-bits axis: canonical R2-Dreamer applies the 1-nat floor to the **sum across all 32 stoch groups** (per-(B,T) total), so the effective per-group floor is `1/32 ≈ 0.031` nats — exactly the original (pre-patch) JAX behavior. The pre-patch JAX code was therefore *faithfully porting* the canonical R2-Dreamer choice; what looks like a bug against DreamerV3 is the canonical R2-Dreamer paper's actual recipe.

**Severity classification.** Per the brief: *"canonical R2-Dreamer is also different from DreamerV3 paper but the JAX port chose DreamerV3 alignment — fix is consistent with that choice"* — this is the case here. The patch deliberately moves the JAX port from R2-Dreamer-faithful to DreamerV3-faithful. This is defensible given (a) the canonical R2-Dreamer's diluted floor explains the latent collapse we observed (1/32 ≈ 0.03 nat floor is empirically inert), and (b) DreamerV3's per-group floor is well-established as the correct latent-regularization recipe in subsequent literature. But it should be **disclosed in the thesis as an intentional deviation from the ported-from base**, not framed as a pure bug-fix.

## Empirical validation

Mini-smoke (Crafter, 30k steps, applied patch, detached PID 2515474):

| Step | `posterior_entropy` (patched) | `kl_divergence` (patched) | `posterior_entropy` (buggy run) |
|---|---|---|---|
| 250 | 2.55 | **1.000** | 2.65 |
| 1250 | 2.14 | **1.000** | — |
| 2250 | 2.13 | **1.000** | — |
| 3250 | 2.08 | **1.000** | — |
| 4250 | **2.00** | **1.001** | (would be ≈1.40 at 5k) |

Two qualitative changes confirm the fix is active:

1. **`latent/kl_divergence` pinned at 1.0** — the per-group floor clamps the post-clamp mean at the configured 1-nat threshold. Pre-patch this metric ranged 2.85–7.1 freely, the floor was inert.
2. **`posterior_entropy` decay is suppressed** — at step 4.25k the patched run holds 2.00 nats vs the buggy run's 1.40 at step 5k (and 0.20 by 95k). Decay rate fell from ~0.10 nats/k-step to ~0.013 nats/k-step.

## Decision: REVERT — keep R2-Dreamer-faithful behavior (2026-04-27)

**Verdict: Patch NOT merged.** Worktree `worktree-agent-a7b6ae2626d4d3d43` left unmerged; `main` retains the canonical R2-Dreamer per-(B,T)-total floor.

**Rationale.** The patch was technically defensible (matches DreamerV3 8/8, empirically prevents latent collapse) but is an *intentional deviation* from the ported-from R2-Dreamer source. The thesis question is "R2-Dreamer + 3D features (VGGT/UNITE) vs 2D features" — comparability with the published R2-Dreamer recipe outweighs latent-stability gains. A floor-recipe shift would introduce a confound alongside the encoder-comparison variable.

**What stays in `main`:**
- Original `_kl_loss` with `kl_free` floor on summed-KL (R2-Dreamer convention, effective per-group ~0.031 nats).
- The `act_entropy=3e-4` fix from PR #96 (this *is* a real bug-fix, deviated from R2-Dreamer's `3e-4` default — see PR #96 body).

**What was empirically verified (still useful as diagnostic baseline):**
- 30k Crafter mini-smoke confirmed per-group floor at 1.0 nat would prevent posterior collapse if adopted.
- All 9 dimensions match `danijar/dreamerv3`; 6 dimensions vs canonical R2-Dreamer (2 DIVERGE: free-bits axis + reduction order).

**Implications for prior runs.** `krokhgwi` (67% SR), `y5a0upzd` (75%), `flky9ybh` (36%), `rsopsua1` (32%), and the 3 active VGGT actent reruns (`4109486-88`) ran with R2-Dreamer-faithful behavior. The "silent latent collapse" they exhibit is the canonical recipe's actual property, not a JAX-port bug. **Comparable to PyTorch R2-Dreamer baselines.**

**If latent stability becomes a thesis blocker** — e.g., 3D-feature variants don't generalize because the latent is too compressed — revisit this decision and re-run the patch under flag `R2DreamerConfig.kl_free_per_group=True` (not yet implemented; would be a config-toggle for future ablation).

## Related

- [`loss/dyn` ≡ `loss/rep` cosmetic proof](dyn-rep-loss-cosmetic-proof.md) — same KL site, different bug-or-not question, both verified empirically.
- L1 CNN smoke audit (no separate page yet; data at `output/runs/r2dreamer-curriculum-l1-smoke-actent/smoke-20260427-104258/`).
- PR #96 (act_entropy 3e-2 → 3e-4) — the smoke that surfaced this bug was originally for verifying the act_entropy fix.
- Issue #94 — `stoch_discrete=16` vs paper `32` (separate, pre-existing).

## Files

- Patch site: `src/r2dreamer/agent.py:645-683` (worktree `worktree-agent-a7b6ae2626d4d3d43`)
- Config: `src/r2dreamer/config.py:62-71` (`kl_free=1.0`, `scale_dyn=1.0`, `scale_rep=0.1`)
- Mini-smoke data: `/tmp/posterior_fix_minismoke/metrics.csv`
- Upstream reference: `danijar/dreamerv3` `dreamerv3/rssm.py:100-113`, `dreamerv3/agent.py:187-189`
- Canonical R2-Dreamer (ported-from) reference: `external/r2dreamer/rssm.py:222-230` (KL site), `external/r2dreamer/dreamer.py:372-374` (loss site), `external/r2dreamer/configs/model/_base_.yaml:2,21-31` (defaults)
