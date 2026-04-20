# L4 Pipeline Profiling (VGGT vs CNN)

**Date**: 2026-04-20
**Source PRD**: [#74](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/74)
**Script**: `modules/r2dreamer/scripts/profile_training.py`
**Plan**: [`docs/plans/l4-profiling.md`](../../plans/l4-profiling.md)

## Motivation

L4 VGGT training was observed reaching only ~1M steps in 48h on H100, well below the CNN baseline. Before committing weeks to a VGGT→JAX port (#72), we instrumented the training loop with per-phase timers to identify which phase of the per-step pipeline is actually slow.

## Hardware & Environment

| Field | Value |
|---|---|
| Host | `uc3n074.localdomain` |
| GPU | NVIDIA H100 (95 GB) |
| Driver | 570.195.03 |
| PyTorch | bf16 autocast, `torch.no_grad`, InfiniteVGGT / StreamVGGT checkpoint `lch01/StreamVGGT` |
| JAX | XLA backend, async dispatch |

## Run Configuration

| Field | CNN | VGGT |
|---|---|---|
| Curriculum | `data/curriculum/level1_1house_1goal.json` | same |
| `prefill_steps` | 2000 | 2000 |
| `acting_steps` | 2000 | 2000 |
| `render_resolution` | 64 | 518 |
| Seed | 0 | 0 |
| Total env steps | 4000 | 4000 |
| Episodes | 8 | 8 |
| JSON | `output/profiling/vggt_vs_cnn_20260420_102405.json` | `output/profiling/vggt_vs_cnn_20260420_103840.json` |

## Results — Per-Phase Wall Time (milliseconds)

Both p50 and p95 are shown; `p50` is the steady-state number after JIT warmup. `delta_ms` column uses p50.

| Phase | CNN p50 | CNN p95 | VGGT p50 | VGGT p95 | delta_ms (p50) |
|---|--:|--:|--:|--:|--:|
| `env_step`      | 1.45   | 2.56   | 2.49   | 3.22   | +1.04 |
| `vggt_forward`  | —      | —      | **168.32** | **170.83** | **+168.32** |
| `vggt_wrapper`  | —      | —      | 0.15   | 0.16   | +0.15 |
| `jax_upload`    | —      | —      | 0.29   | 0.40   | +0.29 |
| `wm_inference`  | 1.52   | 1.69   | 1.44   | 1.63   | −0.08 |
| `buffer_add`    | 0.02   | 0.03   | 0.04   | 0.05   | +0.02 |
| `wm_training`   | 60.41  | 65.14  | 35.76  | 38.50  | −24.65 |

### KV-cache audit

| Encoder | `reset_count` | `boundary_count` | Status |
|---|--:|--:|---|
| CNN | 0 | 9 | N/A (no extractor.reset hook wired) |
| VGGT | 9 | 9 | **PASS** — every `env.reset()` triggers `VGGTFeatureExtractor.reset()` |

### Per-step totals (p50)

Approximate steady-state wall clock per acting step (excluding amortized training):
- **CNN**: env + wm_inference + buffer_add ≈ **3.0 ms** → ~330 steps/s peak
- **VGGT**: env + vggt_forward + vggt_wrapper + jax_upload + wm_inference + buffer_add ≈ **172.7 ms** → ~5.8 steps/s peak

Adding amortized train step (fires every ~2 acting steps at `train_ratio=512 / batch*seq=1024`):
- **CNN**: 3.0 + 30.2 = **33.2 ms/step** → **~30 FPS**
- **VGGT**: 172.7 + 17.9 = **190.6 ms/step** → **~5.2 FPS**

5.2 FPS × 86 400 s × 2 d ≈ **900 k steps in 48 h** — matches the observed ~1 M from wandb run `87u0l6dy`.

## Interpretation

### Finding 1 — `vggt_forward` is ~100% of the VGGT slowdown

- VGGT runs **5.8× slower per step than CNN**.
- Of the `+187 ms/step` slowdown, **`vggt_forward` alone accounts for +168 ms (90%)**.
- Every *other* VGGT-specific phase (`vggt_wrapper`, `jax_upload`) is under 0.5 ms combined — **less than 0.3% of the forward pass**.

### Finding 2 — The PyTorch↔JAX boundary is effectively free

- `vggt_wrapper` (pool + permute + `.cpu().numpy()` + flatten) = **0.15 ms p50**.
- `jax_upload` (numpy → JAX GPU transfer) = **0.29 ms p50**.
- Combined: **0.44 ms p50**, ~0.26% of the forward pass.

Our original hypothesis — that `.cpu().numpy()` + JAX re-upload was a dominant cost — is **disproven by the data**. The H100's GPU↔CPU bandwidth is high enough, and the tensors small enough (37×37×3 float32 = ~16 KB), that the round trip is negligible.

### Finding 3 — `wm_training` is actually *faster* for VGGT than CNN

- VGGT `wm_training` p50 = 35.76 ms
- CNN  `wm_training` p50 = 60.41 ms

VGGT features (4116 floats) are *smaller* than CNN RGB batches (16×64×3×64×64 = 12 MB normalized) to the world model's encoder. So the world-model training step is actually lighter in the VGGT path. Encoder input size, not feature semantics, drives this.

### Finding 4 — `wm_inference` is essentially identical

- CNN p50 = 1.52 ms, VGGT p50 = 1.44 ms.
- The world model's inference cost is dominated by the RSSM + actor, not the encoder. Encoder choice has sub-ms impact here.

### Finding 5 — KV-cache reset contract is correctly wired

9 episode boundaries → 9 VGGT cache resets. The `VGGTObsAdapter(extractor).on_episode_reset = extractor.reset` hook fires on every `env.reset()` — both the initial reset and every post-episode reset during curriculum scene switching. No bug.

## Recommendation for PRD #72 (VGGT JAX port)

**NO-GO as originally scoped.**

Rationale: the hypothesis motivating #72 was that the PyTorch↔JAX boundary crossing dominated the per-step cost. The data shows it's 0.44 ms out of ~173 ms — structurally impossible for a JAX port to meaningfully change the bottom line. Even if a JAX port made `vggt_wrapper` and `jax_upload` literally zero, the per-step cost would drop from 190.6 → 190.2 ms. Not worth weeks of high-risk porting work.

**Repurpose #72 instead.** The actionable paths for reducing `vggt_forward`, in increasing order of effort:

| Option | Expected speedup | Effort | Risk |
|---|---|---|---|
| **`torch.compile(extractor.model)`** with mode=`reduce-overhead` | 10–30% | 1 day | low (try first) |
| **Frame-skip**: run VGGT every *K* env steps, cache features between | 2×–4× at K=2–4 | 1 day | medium (quality) |
| **Async pipeline**: VGGT in a separate process feeding features to the agent loop, agent runs one step behind with stale features | effective ~1× with quality | 1 week | medium (system complexity) |
| **Distilled / pruned VGGT** (student model) | 2×–10× | weeks | high (requires training, quality validation) |

Suggested next step: comment on #72 with this finding and rename it to *"VGGT forward acceleration (torch.compile, frame-skip, or distillation)"*. Start with `torch.compile` — one-line change, quickest to falsify. If it yields ≥20%, ship it. If not, move to frame-skip.

## Update 2026-04-20 — `torch.compile` spike

Tested the 1-day `torch.compile` lever recommended above. Applied via `VGGTFeatureExtractor(compile=True)` — wraps `model.aggregator` + `model.camera_head` with `dynamic=True` (KV cache grows each frame), and `model.point_head` statically. Same 2000/2000 L1-VGGT configuration. JSON: `output/profiling/vggt_vs_cnn_20260420_120455.json`.

| Metric | Uncompiled | Compiled | Δ |
|---|--:|--:|--:|
| `vggt_forward` mean | 161.6 | 170.9 | +9.3 *(compile warmup in first N calls)* |
| `vggt_forward` **p50** | **168.3** | **149.8** | **−18.5 ms (−11%)** |
| `vggt_forward` p95 | 170.8 | 151.8 | −19.0 ms (−11%) |
| `vggt_wrapper` p50 | 0.15 | 0.42 | +0.27 ms *(deferred-work artifact, irrelevant)* |
| `wm_inference` p50 | 1.44 | 1.17 | −0.27 ms (noise) |
| `wm_training` p50 | 35.76 | 34.09 | −1.67 ms (noise) |
| KV-cache audit | 9=9 ✓ | 9=9 ✓ | unchanged |

Per-step total (p50): **190.7 ms → 171.0 ms ≈ −10.3%** → projects ~1 M → 1.1 M steps in 48 h.

### Interpretation

- Real 11% speedup on the bottleneck, with the forward also becoming more consistent frame-to-frame (p95 essentially equals p50 after compile).
- Mean vs p50 gap is expected: dynamo recompiles on new KV-cache shape buckets during warmup; steady-state matters.
- Modest but cheap. Consistent with the mechanism argument: at this model size, both PyTorch and JAX call the same cuBLAS / FlashAttention kernels; framework-level speedup tops out around 10–30%.
- PyTorch also printed a warning: `Consider setting torch.set_float32_matmul_precision('high')`. Not explored — possible further gain.

### Action

`compile` is shipped as an opt-in flag on `VGGTFeatureExtractor` (default `False` to keep dev runs warm-free). Enable in production L4 sbatch via `--compile`. Use as a stacking multiplier with upcoming frame-skip / FastVGGT work, which attacks *how often* the forward runs, orthogonal to how fast a single forward is.

Updated recommendation for #72 repurpose order: **compile ✅ → frame-skip (next) → FastVGGT → distillation**.

## Caveats

- `wm_training` mean vs p50 differ by 5× for both encoders — that's JIT-compile on the first call (visible as XLA `slow_operation_alarm` lines in stderr). Always use p50 / p95, not mean, for steady-state interpretation. Mean values in the JSON are informative for understanding the warmup tax but not representative of sustained throughput.
- `jax_upload` is measured as a separate probe (`jnp.asarray(features).block_until_ready()`) immediately before `agent.act`. The actual upload inside `agent.act` may be deduplicated by JAX; the probe gives a faithful worst-case of "what a fresh upload of this tensor costs" on this hardware.
- Both runs used the same L1 curriculum (`level1_1house_1goal.json`). A larger curriculum (L3/L4) would see more scene switches per unit time, which the profiling result scales linearly with — the per-step breakdown is insensitive to curriculum size.
