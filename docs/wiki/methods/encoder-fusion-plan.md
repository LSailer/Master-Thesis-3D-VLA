# Encoder Fusion Plan — Geometry-Preserving, Pose-Aware Encoder for VGGT → R2Dreamer

Status: **draft** | Branch: `feat/encoder-fusion-ablation` | Target: H100 cluster

## Context

**The thesis claim under test.** R2Dreamer should perform better with 3D (VGGT) features than with 2D (CNN) features on HM3D ObjectNav. The current 75% SR baseline (commit `ab89a0a`, act_entropy=3e-2) uses a fusion path that *destroys both inputs the thesis depends on*:

1. The VGGT extractor emits a structured `world_points (37, 37, 3)` grid + a 9-D `camera_pose` embedding.
2. The adapter [vggt_adapter.py:14](../../../modules/r2dreamer/adapters/vggt_adapter.py#L14) flattens both into a 4116-D vector.
3. The encoder [networks.py:414](../../../modules/r2dreamer/networks.py#L414) is a single `Dense(4116 → 1024)` layer.

This produces three known confounders for the 3D-vs-2D claim (issues #87, #88, #89 documented at [vggt-r2dreamer-callchain.md:103](vggt-r2dreamer-callchain.md#L103)):

- **Geometry destroyed**: the 37×37 spatial grid is flattened before any conv/attention can exploit it.
- **Pose drowned**: 9 / 4116 = 0.22% of dims, ~100× smaller magnitude → effective contribution at init ≈ 0.002%.
- **No multiplicative interactions**: a single linear projection cannot model `pose × geometry` couplings the way the brain probably does ("if I'm facing the door, the floor on my left matters more").

**Why this needs fixing now.** Without a fix, any 3D-vs-2D win can be attributed to a confound (linear projection vs. CNN), and any 3D-vs-2D loss could mean "VGGT features don't help" or "the encoder threw them away." The encoder must be a fair channel before the comparison is interpretable.

**Outcome we want.** Four fusion variants + a tile-baseline + a magnitude-control baseline that share the same R2Encoder backbone (so the comparison isolates the *fusion mechanism*), a diagnostic harness that measures pose influence on `z` directly (not just SR), and a 15-run experiment matrix on HM3D (12 main + 3 magnitude control) that produces publication-grade ablation rows.

User decisions guiding this plan:
- Implement four approaches: **Tile/CoordConv**, **FiLM**, **Plücker rays**, **Cross-attention**.
- Run budget: **medium (8–12 runs)**.
- Diagnostics: add **gradient-norm + pose-ablation probes**.
- VGGT integration: also expose **DINOv2 patch tokens**.
- Execution target: **H100 cluster** via SLURM, no local debugging beyond unit tests.

---

## Approach overview

All four variants reuse the existing `R2Encoder` ([networks.py:362](../../../modules/r2dreamer/networks.py#L362)) as a shared CNN backbone, treating `world_points` as a 3-channel image at 37×37. They differ only in **how camera pose enters**, which is precisely the variable we want to ablate.

| Variant | Pose injection | Backbone | Expected role |
|---|---|---|---|
| `vggt_tile` | Broadcast 9-D pose to (9, 37, 37), concat to channels (input 12, 37, 37) | R2Encoder | Cheap baseline. If FiLM/Plücker don't beat this, they aren't earning their keep. |
| `vggt_film` | MLP(pose) → (γ, β) per channel; modulate after each Conv block | R2Encoder w/ FiLM hooks | Per-channel pose dominance, no spatial cost. |
| `vggt_plucker` | Per-pixel 6D Plücker rays from Habitat extrinsics+intrinsics, concat to channels (input 9, 37, 37) | R2Encoder | Strongest per-token pose signal. Geometry-aware by construction. |
| `vggt_xattn` | DINOv2 patch tokens as KV, k=8 learned pose queries → MHA | Token-level (no R2Encoder) | Most expressive. Tests whether pose-driven token routing > channel modulation. |
| `vggt_pose_scaled` | pose × 100 through the existing flatten+linear path | Existing `VGGTEncoder` | Issue #87's Trivial fix. Isolates pose magnitude from encoder routing. If this matches Tile/FiLM/Plücker, the encoder rewrite was unnecessary. |

Plus the existing baseline `vggt` (flatten+linear) stays in the registry as the control, and `cnn` (raw RGB → R2Encoder) stays as the 2D comparison.

---

## Phase 0 — Baseline variance calibration

**Why first.** No multi-seed variance baseline exists in the wiki today. The 75% SR anchor is one seed, and the act_entropy=3e-2 rerun ([l1-act-entropy-3e-2.md](../experiments/l1-act-entropy-3e-2.md)) is a single run. Before committing 12+ H100 runs, measure σ on the baseline.

**Action.** Run 3 seeds of the current `vggt` baseline (commit `ab89a0a`, act_entropy=3e-2 hyperparameters, no plan changes) end-to-end. Report mean ± std on SR/SPL.

**Gating rule.**
- If σ ≤ 3pp on SR: proceed with 3 seeds × 4 encoders + 3 seeds × 1 magnitude-control = 15 runs.
- If σ > 3pp on SR: expand to 5 seeds per variant, OR focus the matrix on the top-2 candidates (`vggt_film` + `vggt_plucker`) at 5 seeds each.

**Side task.** Profile one full HM3D eval (val split) to confirm wall-time. The plan budgets ~30–60 min/eval; verify before committing 24 evals worth of cluster time.

---

## Phase 1 — VGGT plumbing: expose patch tokens (compressed)

**Why first.** Cross-attention needs DINOv2 patch tokens. Doing this before any encoder work means all variants see the same buffer schema and we don't ship two adapter changes.

**Buffer budget — corrected.** Current buffer is ~16.5 GB (4116 floats × 1M cap × 4 bytes), memory-resident at [replay_buffer.py:42](../../../modules/shared/replay_buffer.py#L42). New schema with 91732 floats/step is **~367 GB**. The "halve patch tokens to 32 channels" mitigation only saves ~84 GB → still ~283 GB. Before Phase 1 lands, the team must choose ONE of:
- (a) cap buffer at 250k steps (~92 GB at 91732 floats);
- (b) implement a disk-backed wrapper (HDF5 or memmap) — touches `replay_buffer.py` substantially;
- (c) compress patch tokens harder than 64 dims (e.g., 16 dims → 22k floats/step → ~88 GB at 1M cap).

The downstream Phase 1 design assumes 64-dim tokens; if option (c) is chosen, all "91732 floats" references in this plan must be updated.

**Why a fixed projection.** A trainable projection inside the extractor would create a JAX/PyTorch gradient boundary problem (VGGT is frozen and runs in PyTorch acting; gradients flow only in JAX). A **fixed random projection** (Johnson–Lindenstrauss style, frozen at module construction) preserves enough signal for the JAX-side encoder to learn from. If empirically lossy, escalate to **learned PCA on a held-out batch** — still no gradient boundary issue.

**JL caveats.** The JL bound for ε=0.1 with N=1369 patches needs k > ~6000; the proposed 64 dims is two orders below the bound. A more conservative starting point is 512 dims (~15× compression instead of 16×). The projection matrix MUST be seeded (e.g., `torch.manual_seed(42)` before `torch.randn`) and serialized into the model checkpoint, otherwise variants drift silently across SLURM jobs.

### Files to modify

- **Step 0 — verify `aggregated_tokens` shape.** Before any code change, print/assert the actual shape of `aggregated_tokens` at [feature_extractor.py:129](../../../modules/vggt/feature_extractor.py#L129). The plan assumes 37×37×1024 but the variable may be `(B, T_agg, C_agg)` with a different layout. The downstream projection design depends on the answer; if the shape is not spatial, a reshape/unflatten step must be added.

- **[modules/envs/habitat_r2dreamer.py](../../../modules/envs/habitat_r2dreamer.py)** — extend obs dict
  - Currently returns only `{image, reward, is_first, is_last, is_terminal, success, spl}`. The Plücker variant (Phase 2c) needs Habitat extrinsics + intrinsics; without this, Plücker is broken-by-default.
  - Add `agent_state`: 4×4 extrinsics + 3×3 intrinsics (or a flat 25-D representation) to the obs dict.
  - Update [vggt_adapter.py](../../../modules/r2dreamer/adapters/vggt_adapter.py) to also pack `agent_state` into the buffer flat layout.
  - Validate Habitat's coordinate frame matches VGGT's `world_points` frame visually before training (one paired plot of rays + world_points).

- **[modules/vggt/feature_extractor.py](../../../modules/vggt/feature_extractor.py)** (~line 83)
  - In `extract()`, also pull the DINOv2 patch tokens from `aggregated_tokens` (the layer before the DPT head — this is what VGGT calls "patch tokens"; verify the exact slice and shape per Step 0 above). Reshape to (37, 37, 1024) if the layout permits.
  - Apply a fixed projection: `tokens_compressed = tokens @ self._proj` where `self._proj` is a frozen `(1024, 64)` matrix initialized once with `torch.randn` × `1/sqrt(1024)` and stored on the module. Result: (37, 37, 64).
  - Return dict now includes `"patch_tokens": (37, 37, 64) float32`.

- **[modules/r2dreamer/adapters/vggt_adapter.py](../../../modules/r2dreamer/adapters/vggt_adapter.py)**
  - Update `VGGT_FEATURE_DIM` and `_flatten_vggt`: pack `[world_points.flatten() (4107), patch_tokens.flatten() (87616), camera_pose (9)]` → 91732 floats.
  - Keep the flat byte layout (don't switch to a dict) so `ReplayBuffer` doesn't need touching. Encoders unpack on the JAX side via known offsets.

- **[modules/r2dreamer/config.py:27](../../../modules/r2dreamer/config.py#L27)**
  - Update `vggt_feature_dim`. Add new config fields per encoder variant (FiLM hidden dim, x-attn heads, etc.).

### Verification (Phase 1)

- Extractor unit: feed a synthetic RGB image, assert returned shapes and `patch_tokens.std() > 0.05`.
- Adapter unit: assert `_flatten_vggt(...).shape == (91732,)` and that unpacking returns the original components bit-exact.
- Smoke run: 1k env steps with the existing `vggt` baseline encoder pointed at the new buffer schema (it should ignore the extra dims via slicing). SR/loss curve should match the act_entropy=3e-2 baseline within noise.

---

## Phase 2 — Encoder registry expansion

All encoders go in [modules/r2dreamer/networks.py](../../../modules/r2dreamer/networks.py) (Flax). The registry lives in [modules/r2dreamer/launch/registries.py:15](../../../modules/r2dreamer/launch/registries.py#L15) — one entry per variant.

Each encoder takes the flat 91732-D obs and unpacks internally to:
- `wp` reshaped to (B, 3, 37, 37)
- `tokens` reshaped to (B, 64, 37, 37)
- `pose` (B, 9)

### 2a. `TileEncoder` (CoordConv-style)

```python
class TileEncoder(nn.Module):
    embed_dim: int = 1024

    def __call__(self, obs):
        wp, _, pose = unpack_vggt(obs)
        pose_tiled = jnp.broadcast_to(
            pose[:, :, None, None], (pose.shape[0], 9, 37, 37)
        )  # (B, 9, 37, 37)
        x = jnp.concatenate([wp, pose_tiled], axis=1)  # (B, 12, 37, 37)
        x = R2Encoder(mults=(2, 3, 4, 4))(x)           # (B, ~4096)
        return nn.Dense(self.embed_dim)(x)              # (B, 1024) — matches RSSM contract
```

**Note.** R2Encoder's first conv accepts arbitrary input channels, so this works without modification. The `Dense(embed_dim)` projection head is required — without it the output dim (~4096) mismatches the RSSM posterior MLP which expects 1024.

### 2b. `FiLMEncoder`

Refactor `R2Encoder` so the conv loop optionally accepts a list of `(γ, β)` pairs and applies FiLM after each block (after RMSNorm, before SiLU). A small pose MLP produces all (γ, β) at once: `MLP(pose) → 2 * sum(channels)`, split per layer.

```python
class FiLMEncoder(nn.Module):
    def __call__(self, obs):
        wp, _, pose = unpack_vggt(obs)
        gammas, betas = FiLMHead(channels=(32, 48, 64, 64))(pose)
        return R2EncoderFiLM(gammas, betas)(wp)
```

`R2EncoderFiLM` is a parameterized variant of `R2Encoder`. The "small kwarg add" framing understates the refactor: R2Encoder's conv loop ([networks.py:377-383](../../../modules/r2dreamer/networks.py#L377-L383)) is tight (Conv → max_pool → RMSNorm → SiLU); branching on `film_params` leaks abstraction. A cleaner alternative is a separate `R2EncoderFiLM` wrapper class that calls R2Encoder internally. Pick one and document the choice; either is acceptable but the diff will be larger than a single kwarg.

### 2c. `PluckerEncoder`

Compute Plücker rays from **Habitat ground-truth extrinsics+intrinsics**, not from VGGT's learned 9-D pose. This requires:

- The Habitat-wrapper extension added in Phase 1 (extrinsics + intrinsics surfaced via `agent_state` in the obs dict). The current wrapper does NOT expose any pose/sensor metadata, so this is a hard dependency — Plücker is blocked until Phase 1 lands the Habitat change.
- A `compute_plucker(extrinsics, intrinsics, H=37, W=37)` JAX function that returns rays of shape (B, 6, 37, 37). Each pixel (u, v) yields direction `d = R · K⁻¹·[u, v, 1]` and moment `m = origin × d`.
- Concat to world_points: `x = jnp.concatenate([wp, rays], axis=1)` → (B, 9, 37, 37) → R2Encoder.

```python
class PluckerEncoder(nn.Module):
    def __call__(self, obs):
        wp, _, _, extr, intr = unpack_vggt_with_habitat(obs)
        rays = compute_plucker(extr, intr, 37, 37)
        x = jnp.concatenate([wp, rays], axis=1)
        return R2Encoder()(x)
```

**Why Habitat extrinsics, not VGGT's 9-D.** VGGT's pose is a learned embedding without a guaranteed metric interpretation; Plücker math needs a real SE(3) transform. Habitat gives this exactly. (For the other variants, VGGT's 9-D is fine — it's a feature, not a geometric quantity.)

### 2d. `CrossAttnEncoder`

```python
class CrossAttnEncoder(nn.Module):
    embed_dim: int = 1024
    n_queries: int = 8
    n_heads: int = 4

    def __call__(self, obs):
        _, tokens, pose = unpack_vggt(obs)              # tokens: (B, 64, 37, 37)
        B = tokens.shape[0]
        kv = tokens.reshape(B, 64, -1).transpose(0, 2, 1)  # (B, 1369, 64)
        q = nn.Dense(self.n_queries * 64)(pose).reshape(B, self.n_queries, 64)
        out = nn.MultiHeadDotProductAttention(num_heads=self.n_heads)(q, kv)
        return nn.Dense(self.embed_dim)(out.reshape(B, -1))
```

Note: this is the only variant that does **not** use R2Encoder. It uses the patch tokens directly. Include world_points only as an optional KV concat (keep it simple for v1: tokens-only KV).

### Registry update

[modules/r2dreamer/launch/registries.py:15](../../../modules/r2dreamer/launch/registries.py#L15):

```python
encoder_registry = {
    "cnn":               CNNEncoder,
    "vggt":              VGGTEncoder,        # current baseline
    "vggt_pose_scaled":  VGGTEncoder,        # same class, pose × 100 in adapter
    "vggt_tile":         TileEncoder,
    "vggt_film":         FiLMEncoder,
    "vggt_plucker":      PluckerEncoder,
    "vggt_xattn":        CrossAttnEncoder,
}
```

### Verification (Phase 2)

- Per-encoder unit: feed synthetic obs of shape (B=4, 91732), assert output shape (B, 1024), assert `embed.std()` in [0.3, 0.7] at init (LeCun-init healthy range, matches the baseline diagnostic at [vggt-r2dreamer-callchain.md:59](vggt-r2dreamer-callchain.md#L59)).
- Param count log: print `sum(jax.tree.leaves(params))` for each variant; expect Tile ≈ baseline + ε, FiLM ≈ baseline + 10–50k, Plücker ≈ baseline + ε, X-attn ≈ baseline + 100–300k.
- Smoke train: 5k env steps each with `--encoder vggt_film` etc. Loss should descend, no NaNs.

---

## Phase 3 — Diagnostic instrumentation

Goal: directly measure "does pose actually influence the latent `z`," beyond just SR.

### 3a. Gradient-norm probe (training-time)

In [modules/r2dreamer/agent.py:299](../../../modules/r2dreamer/agent.py#L299) (`_loss_fn`), after the forward pass, compute and log:

```python
grad_z_wrt_pose = jax.grad(lambda p: stoch_logits(... pose=p ...).sum())(pose)
log["pose_grad_norm"] = jnp.linalg.norm(grad_z_wrt_pose)
```

Cheap (one extra grad call on a scalar reduction). Expected behavior: rises during training as pose becomes more informative; the variants that fix the failure mode should show ≥10× the baseline's `pose_grad_norm` after warmup.

**Gating caveat — Phase 3a is blocked until encoders accept `pose` as a separable JAX leaf.** Today, pose is bundled inside `batch["obs"]` (a flat array) and unpacked inside each encoder. There is no `pose=...` kwarg path through the encoder/RSSM, so the snippet above cannot run as-is. Two ways forward: (i) restructure the encoder call signatures so `pose` is a separate JAX argument, or (ii) live with the cross-variant interpretation gap — for Tile/Plücker the gradient measures image-space flux through Conv kernels; for FiLM it measures modulation-weight sensitivity; for the baseline `vggt` and `vggt_pose_scaled` it measures linear-projection sensitivity. The numbers are NOT directly comparable across encoder families. If (i) is too invasive, document the per-variant interpretation in the results page.

### 3b. Pose-ablation eval (eval-time only, expensive)

Add a `--ablate-pose` flag to [modules/r2dreamer/launch/evaluate.py:72](../../../modules/r2dreamer/launch/evaluate.py#L72) (there is no `eval/` directory; the entry point is `evaluate.py`) that, at every acting step, replaces the real pose with **a permutation across the batch** (primary mode, breaks pose-image correspondence cleanly), with **zero-pose** as a secondary mode. Then run a full HM3D evaluation. **SR drop = causal contribution of pose.** Note: zero-pose is NOT equivalent to "no pose" once FiLM γ,β are trained — it produces a different (β-only) deterministic input. Permutation is the cleaner test.

Expected ranking:
- Baseline `vggt`: SR drop ≈ 0% (pose was never used).
- `vggt_tile`: small drop.
- `vggt_film`, `vggt_plucker`: large drops (>10pp) if working correctly.
- `vggt_xattn`: large drop, possibly the largest.

This is the **money diagnostic** for the thesis claim: "3D features only help if the encoder lets pose influence the latent."

### Verification (Phase 3)

- Confirm `pose_grad_norm` is non-zero at init for all non-baseline variants and zero (or tiny) for the baseline.
- Confirm pose-ablation runs end-to-end on a single eval episode.

---

## Phase 4 — Experiment matrix (15 runs, H100 cluster)

SLURM scripts go in [modules/r2dreamer/scripts/slurm/](../../../modules/r2dreamer/scripts/slurm/) following the [train_curriculum_l1.sbatch](../../../modules/r2dreamer/scripts/slurm/train_curriculum_l1.sbatch) pattern (the precedent for the `ab89a0a` baseline). Each run uses the same hyperparameters as the 75% SR baseline (act_entropy=3e-2, fixed VGGT) — only the `--encoder` flag changes. Each script must emit `MANIFEST.json` (git_sha, config, wandb_id, slurm_id, start/end timestamps) per the project's data-layout discipline; the reporter skill creates `_blessed/encoder-fusion-ablation/<variant>` aliases after the matrix completes so the wiki-audit skill can cross-check the result table.

**Cluster targeting.** All 15 runs go directly to the H100 cluster — skip local debug runs beyond Phase 1/2 unit tests. The phase-2 smoke trains (5k steps each) are the only local steps; if they don't NaN, push to SLURM. H100 capacity means we can launch the matrix as parallel jobs rather than sequential.

**Eval cost.** Each run produces 2 evals (with-pose + ablate-pose). 15 runs × 2 evals × ~30–60 min/eval = ~15–30 H100-hours of post-training eval. Parallelizable across nodes after training completes. Phase 0 verifies the per-eval wall time before this is committed.

| # | Encoder | Seeds | Purpose |
|---|---|---|---|
| 1–3 | `vggt` (current baseline) | 3 | Anchor — re-confirms 75% SR. Reuses ab89a0a if seed-matched. |
| 4–6 | `vggt_tile` | 3 | Cheap baseline. Tests "preserve geometry" alone. |
| 7–9 | `vggt_film` | 3 | Primary candidate. Tests pose-channel modulation. |
| 10–12 | `vggt_plucker` | 3 | Primary candidate. Tests per-pixel pose injection. |
| 13–15 | `vggt_pose_scaled` | 3 | Magnitude control (issue #87 Trivial fix). Disambiguates "pose was drowned by magnitude" from "encoder needs new architecture". |

**Why not include `vggt_xattn` in the main matrix.** Cross-attention has more knobs (n_queries, n_heads, layer count) and bigger KL-stability risk in the RSSM posterior. Run it as an exploratory follow-up after the matrix completes — adds 3 more runs if the pilot looks good.

**Eval protocol per run.**
- Standard SR/SPL on the HM3D val split (existing eval pipeline).
- Pose-ablation eval (Phase 3b) at the final checkpoint only.
- Per-step `pose_grad_norm` logged to W&B (Phase 3a).

**Reporting.** Output table in [docs/wiki/methods/](.) with mean ± std across seeds for SR, SPL, pose-ablation SR drop, and final `pose_grad_norm`. Include the full ratio `(SR with pose) / (SR without pose)` as the "pose effective use" metric.

---

## Phase 5 — End-to-end verification

Before declaring the experiment shipped:

1. **Reproduce baseline.** Re-run `vggt` with same seed as `ab89a0a`; confirm SR ≈ 75% within 2pp. If not, the buffer-schema change (Phase 1) regressed something.
2. **Sanity-rank the variants.** Pose-ablation SR drop should be monotonic: `vggt < vggt_pose_scaled ≈ vggt_tile < vggt_film ≤ vggt_plucker`. **Quantitative decision rule:** if `vggt_film`'s `pose_grad_norm` (Phase 3a) is statistically indistinguishable from baseline `vggt` (t-test p > 0.05 across the 3 seeds), then FiLM training is not learning to extract pose; rerun with `vggt_pose_scaled` results to disambiguate "encoder is broken" from "magnitude was the only confound." A monotonicity violation (e.g., Plücker < FiLM) is interpretable post-hoc, not a fatal regression — record it in the writeup.
3. **Param-count audit.** No variant should have >2× the baseline's encoder param count without justification.
4. **Wiki entry.** Add [encoder-fusion-ablation.md](encoder-fusion-ablation.md) with the result table and a short post-mortem on which family won and why.

---

## Critical files

| Path | Change |
|---|---|
| [modules/envs/habitat_r2dreamer.py](../../../modules/envs/habitat_r2dreamer.py) | Surface Habitat `agent_state` (4×4 extrinsics + 3×3 intrinsics) in obs dict — required by Plücker. |
| [modules/vggt/feature_extractor.py](../../../modules/vggt/feature_extractor.py) | Expose patch tokens with seeded fixed 1024→K projection (K = 64 default; consider 512). Verify `aggregated_tokens` shape first. |
| [modules/r2dreamer/adapters/vggt_adapter.py](../../../modules/r2dreamer/adapters/vggt_adapter.py) | New flat layout; also pack Habitat extrinsics+intrinsics for Plücker; pose ×100 path for `vggt_pose_scaled`. |
| [modules/shared/replay_buffer.py](../../../modules/shared/replay_buffer.py) | Touched only if option (b) disk-backed wrapper is chosen for the buffer-budget mitigation. |
| [modules/r2dreamer/networks.py](../../../modules/r2dreamer/networks.py) | Add `TileEncoder`, `FiLMEncoder`, `PluckerEncoder`, `CrossAttnEncoder`; extend `R2Encoder` with optional FiLM hooks (or add `R2EncoderFiLM` wrapper). |
| [modules/r2dreamer/config.py](../../../modules/r2dreamer/config.py) | New `vggt_feature_dim`, per-variant config fields. |
| [modules/r2dreamer/launch/registries.py:15](../../../modules/r2dreamer/launch/registries.py#L15) | Register five new encoder names (incl. `vggt_pose_scaled`). |
| [modules/r2dreamer/agent.py:299](../../../modules/r2dreamer/agent.py#L299) | Add `pose_grad_norm` logging in `_loss_fn` (gated on encoder restructure — see Phase 3a). |
| [modules/r2dreamer/launch/evaluate.py:72](../../../modules/r2dreamer/launch/evaluate.py#L72) | Add `--ablate-pose` flag (permutation primary, zero secondary). |
| [modules/r2dreamer/scripts/slurm/](../../../modules/r2dreamer/scripts/slurm/) | 15 SLURM scripts (5 encoders × 3 seeds), H100 partition, modeled on `train_curriculum_l1.sbatch`. Each emits MANIFEST.json. |
| [encoder-fusion-ablation.md](encoder-fusion-ablation.md) | Results writeup at the end. |
| [../index.md](../index.md) | Wiki index entry pointing at this plan. |

## Reused functions (no rewriting needed)

- `R2Encoder` ([networks.py:362](../../../modules/r2dreamer/networks.py#L362)) — backbone for tile/film/plucker variants.
- `RMSNorm`, `R2MLP` ([networks.py:389](../../../modules/r2dreamer/networks.py#L389)) — building blocks.
- `VGGTFeatureExtractor.extract` and KV-cache machinery ([feature_extractor.py:83](../../../modules/vggt/feature_extractor.py#L83)) — only adds an output, doesn't change call-shape.
- `R2RSSM.__call__` ([networks.py:229](../../../modules/r2dreamer/networks.py#L229)) — untouched. The encoder contract (`obs → embed (B, 1024)`) is preserved by all variants, so the RSSM posterior head sees the same input shape.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Patch-token fixed projection loses too much info | Have learned-PCA fallback ready (offline fit on 10k frames, freeze, ship as the projection matrix). |
| Plücker rays diverge from VGGT's internal pose convention → ray-vs-feature misalignment | Use Habitat extrinsics directly; visualize one `(rays, world_points)` pair as a sanity check before training. |
| FiLM γ explodes at init → NaN loss | Initialize the FiLM head's γ-output to predict 1, β-output to predict 0 (zero-init the final layer + add 1 to γ). Standard practice. |
| Cross-attention destabilizes RSSM KL | Keep n_queries small (k=4 or 8), use RMSNorm, run x-attn outside the main matrix as exploratory. |
| Buffer size 22× growth blows out RAM | ~367 GB at 1M cap (`replay_buffer.py:42` is memory-resident — confirmed). Mitigations per Phase 1: cap at 250k, disk-back (HDF5/memmap), or compress to 16 dims (~88 GB). Halving to 32 channels only saves ~84 GB → still ~283 GB. |
| `vggt` baseline regresses after schema change | Phase 5 step 1 catches this; rollback path is reverting Phase 1 commits independently. |

---

## Summary

The plan keeps the existing R2Encoder as a shared backbone across three variants, adds a token-level cross-attention as a fourth, and adds a magnitude-control baseline (`vggt_pose_scaled`) so the encoder rewrite isn't credited for a fix that pose × 100 alone would deliver. Pose is given a real pathway in every variant (channels for tile/plücker, modulation for FiLM, queries for x-attn) — none of them can repeat the "pose drowned in 0.22% of dims" failure mode by construction. Diagnostics measure pose influence directly, so the experiment can distinguish "method works but RSSM ignored it" from "method doesn't help." Phase 0 calibrates baseline seed-variance before the matrix runs. Fifteen runs (12 main + 3 magnitude control) land a clean ablation table with the same hyperparameters as the 75% SR baseline.
