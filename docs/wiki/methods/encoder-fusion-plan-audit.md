# Audit of Encoder Fusion Plan (PR #110)

**PR:** https://github.com/LSailer/Master-Thesis-3D-VLA/pull/110
**Branch:** `feat/encoder-fusion-ablation`
**Plan audited:** [encoder-fusion-plan.md](encoder-fusion-plan.md)

## Context

PR #110 is a *spec PR*: it adds the encoder-fusion plan and the `l1-act-entropy-3e-2` baseline page; no implementation. The plan addresses the **3 confounders** documented in issues #87/#88/#89 (geometry destroyed, pose drowned at 0.22% of dims, no multiplicative pose×geometry interactions) by proposing 4 encoder variants on a shared ConvEncoder backbone. Until those confounders are removed, the thesis's 3D-vs-2D comparison is uninterpretable.

This audit was run by 3 parallel Explore agents in two rounds (pre-checkout via `git show`, then live-tree). Both rounds converged on the same blockers. Angles: **code-truth** of cited paths/symbols, **design-risk** in the proposed mechanisms, **experiment-rigor** for thesis-defense readiness.

---

## Blockers — must resolve before Phase 1 starts

### B1. Pose-gradient probe is not implementable as written
- `_loss_fn` lives in [agent.py:299](../../../src/r2dreamer/agent.py#L299), **not** `trainer.py`. Signature passes `batch` (a flat dict); pose is **bundled inside `batch["obs"]`** by the time JAX sees it.
- `jax.grad(lambda p: stoch_logits(... pose=p ...).sum())(pose)` cannot run as written — there is no `pose=...` kwarg path through the encoder/RSSM.
- For Tile/Plücker the gradient *is* well-defined (pose enters as concat'd channels), but the meaning differs across variants — image-space gradient flux for Tile, modulation-weight sensitivity for FiLM. **The diagnostic is not directly comparable across encoder types.**
- **Fix:** restructure encoder call signatures to accept `pose` as a separable JAX leaf, or accept the cross-variant interpretation gap and document it.

### B2. Habitat extrinsics + intrinsics are NOT in the obs dict
- The plan claims `obs_dict["sensor_pose"]` will source Plücker rays. The Habitat R2Dreamer wrapper returns only `{image, reward, is_first, is_last, is_terminal, success, spl}`. **No agent_state, no sensor_pose at all.**
- **Fix:** extend [habitat_r2dreamer.py](../../../src/environments/habitat_r2dreamer.py) to surface `agent_state` (4×4 extrinsics + intrinsics) AND extend the buffer schema in the same change. Plan's Phase 1 must include this, not defer it to Phase 2.
- Validate Habitat's coordinate frame matches VGGT's `world_points` frame visually before training.

### B3. ReplayBuffer is memory-resident — 22× growth lands in RAM
- [replay_buffer.py:42](../../../src/buffer/replay_buffer.py#L42): `self.obs = np.zeros((cap, *config.obs_shape), dtype=np_dtype)`. No disk backing.
- Current footprint: ~16.5 GB (4116 floats × 1M cap × 4 bytes). New schema (91732 floats): **~367 GB**.
- The "halve patch tokens to 32 channels" mitigation only saves ~84 GB → still ~283 GB.
- **Fix:** before Phase 1, choose one of: (a) cap buffer at 250k, (b) disk-backed wrapper (HDF5 / memmap), (c) compress patch tokens harder than 64 dims. Measure first.

### B4. `aggregated_tokens` is not returned by `extract()` and shape is unverified
- [feature_extractor.py:129](../../../src/vggt/feature_extractor.py#L129) computes `aggregated_tokens` but only uses it internally for `camera_head` and `point_head`. `extract()` returns only `world_points` and `camera_pose`.
- The plan's "37×37×1024" assumption for patch tokens is unverified — the variable shape at line 129 may be `(B, T_agg, C_agg)`, not the spatial 37×37 grid the plan needs.
- **Fix:** Phase 1 step 1 = print/assert the shape before designing the projection.

### B5. Cited paths are wrong in 4 places — fix the plan first
| Plan cites | Actual location |
|---|---|
| `train.py:67` (encoder registry) | [registries.py:15](../../../src/r2dreamer/launch/registries.py#L15) |
| `trainer.py` (`_loss_fn`) | [agent.py:299](../../../src/r2dreamer/agent.py#L299) |
| `src/r2dreamer/eval/` | does not exist; entry is [evaluate.py:72](../../../src/r2dreamer/launch/evaluate.py#L72) |
| `slurm/l1/` | does not exist; precedent is [train_curriculum_l1.sbatch](../../../scripts/r2dreamer/slurm/train_curriculum_l1.sbatch) |

---

## Major risks — should be addressed but not blocking

### M1. Fixed Johnson-Lindenstrauss projection 1024 → 64 is too aggressive
- 16× compression. JL bound for ε=0.1, N=1369 patches requires k > ~6000 — the plan's 64 is two orders of magnitude under.
- ~94% information loss before the JAX encoder sees the tokens.
- The fixed projection matrix must be **byte-identical across SLURM jobs** — the plan doesn't seed/serialize it. If `torch.randn` runs at module construction without a fixed seed, every job gets a different projection → silent feature-distribution shift across the matrix.
- **Fix:** start at 512 dims (15× less compression), or commit to learned-PCA upfront (the plan mentions it as fallback only). Seed and serialize the projection matrix into the model checkpoint.

### M2. Statistical power: no seed-variance baseline exists
- Wiki has zero `seed=X,Y,Z: SR ± σ` measurements. The 75% SR baseline is one seed; the act_entropy=3e-2 rerun ([l1-act-entropy-3e-2.md](../experiments/l1-act-entropy-3e-2.md)) is still in progress.
- 3 seeds × 4 encoders cannot discriminate a 5pp effect if the seed-to-seed σ is itself 3-5pp.
- **Fix:** Phase 0 = 3 seeds of baseline `vggt` first to establish σ. If σ > 3pp, expand to 5 seeds per variant or focus matrix on top-2 variants.

### M3. Issue #87's "Trivial" fix (scale pose × 100) is not tested
- The [vggt-r2dreamer-callchain](vggt-r2dreamer-callchain.md) doc identifies pose-magnitude collapse as one of the 3 confounders; issue #87 explicitly proposes the trivial fix: scale pose ×100 through the existing Dense projection.
- The 4 variants all change the *encoder*. None tests whether **just rescaling pose** would close the gap.
- Without this control, a Tile/FiLM/Plücker win is ambiguous — geometry preservation, or just fixing the magnitude gap?
- **Fix:** add `vggt_pose_scaled` (pose × 100, existing Dense) as a 5th variant or 2-run diagnostic. Cheap; large interpretability gain.

### M4. FiLM "small diff" claim understates the refactor
- ConvEncoder's conv loop is tight ([networks.py:377-383](../../../src/r2dreamer/networks.py#L377-L383)): Conv → max_pool → RMSNorm → SiLU.
- Adding `film_params` kwarg makes ConvEncoder branch on FiLM presence — leaks abstraction. Cleaner: `ConvEncoderFiLM` wrapper.
- Not a blocker, but the diff will be larger than "small kwarg add."

---

## Minor flags

- **Pose-ablation diagnostic semantics.** Zeroing pose ≠ "no pose" once FiLM γ,β are trained — it's a different deterministic input (β-only). Permutation across batch is the cleaner test. Plan should commit to permutation as primary.
- **Phase 5 monotonicity prediction.** `vggt < vggt_tile < vggt_film ≤ vggt_plucker` is a guess, and the "if vggt_film shows zero drop, FiLM is broken" rule is non-falsifiable (γ,β are *trained* — they could legitimately learn zero pose contribution). Replace with a quantitative rule based on `pose_grad_norm` t-test.
- **Eval cost.** 24 HM3D evals × ~30-60 min/eval = ~12-24 H100-hours. Tractable but not budgeted in the plan.
- **MANIFEST + `_blessed/` discipline.** Not called out. The wiki-audit skill needs `run_path`, `slurm_id`, `wandb_id` frontmatter on the writeup; the plan should commit to emitting these per run.
- **Cross-attention KL deferral.** Defensible but not substantiated by any prior wiki evidence.

---

## What the plan got right

- Shared ConvEncoder backbone across 3 of 4 variants — isolates the fusion-mechanism variable cleanly.
- `ConvEncoder` is genuinely shape-agnostic on input channels (Flax `nn.Conv` infers; `mults` derive widths multiplicatively from a base).
- Anchoring on `ab89a0a` (act_entropy=3e-2, 75% SR, 2026-04-30) and freezing hyperparameters across the matrix isolates encoder variation without sabotaging any variant.
- Pose-ablation as a *causal* probe (vs. SR alone) is well-motivated and aligned with the [2026-03-03 Braun meeting](../meetings/2026-03-03-braun.md).
- Wiki/reporting integration mirrors the existing [l1-act-entropy-3e-2.md](../experiments/l1-act-entropy-3e-2.md) template.

---

## Recommended next steps

PR #110 is a *spec*; "merging" freezes the plan as the contract for follow-up commits. Recommend:

1. Patch B5 (4 wrong file paths) — 5 min plan edit.
2. Resolve B1, B2, B3, B4 in writing in the plan before any Phase 1 code lands. Each changes the size and order of Phase 1.
3. Add `vggt_pose_scaled` (M3) as a 5th variant or explicitly justify its absence — issue #87 already proposed it.
4. Add a Phase 0 calibration step: 3-seed baseline run for variance + eval-cost measurement (M2).
5. Then merge as the spec contract and start Phase 1.

## Verification

After plan updates, re-run the 3 Explore audits; B1–B5 should disappear from the blocker list. Run the Phase 0 calibration baseline and confirm 75% SR ± σ before launching the 12-run grid.
