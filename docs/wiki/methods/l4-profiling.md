# L4 Pipeline Profiling (VGGT vs CNN)

**Date**: 2026-04-20
**Source PRD**: [#74](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/74)
**Script**: `scripts/r2dreamer/profile_training.py`
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

## Update 2026-04-25 — Re-verification + JAX comparison

Re-ran the same 2000/2000 L1-VGGT and L1-CNN configurations under the current `main` branch (post launcher Phase 1–3 refactor, post `act_entropy` fix #93). Goal: verify the 2026-04-20 numbers still hold, document the JAX port's relative position, and decide on issue #88 (GPU↔CPU round-trip).

JSONs: `output/profiling/vggt_vs_cnn_20260425_135105.json` (CNN), `..._20260425_140803.json` (VGGT no-compile), `..._20260425_142600.json` (VGGT compile).

### VGGT phase reproduction (p50 ms)

| Phase | 2026-04-20 no-compile | 2026-04-25 no-compile | Δ | 2026-04-20 compile | 2026-04-25 compile | Δ |
|---|--:|--:|--:|--:|--:|--:|
| `vggt_forward` | 168.32 | 166.06 | −2.3 | 149.84 | 148.23 | −1.6 |
| `vggt_wrapper` | 0.15 | 0.15 | 0 | 0.42 | 0.43 | 0 |
| `jax_upload`   | 0.29  | 0.27 | 0 | —    | 0.26 | — |
| `wm_inference` | 1.44  | 1.17 | −0.3 | 1.17 | 1.16 | 0 |
| `wm_training`  | 35.76 | 33.70 | −2.1 | 34.09 | 34.22 | 0 |

All deltas within noise. KV-cache audit still 9 = 9 ✓. The launcher refactor and `act_entropy` fix did not introduce a VGGT-path regression.

### CNN side-note

CNN `wm_training` p50 went from 60.41 → 67.57 ms (+12%). Outside noise but unrelated to the VGGT critical path; flagged for separate investigation.

### JAX comparison (autoresearch harness)

Three different harnesses report three different PyTorch baselines — direct cross-harness comparison is unreliable. The only apples-to-apples line is `src/vggt/autoresearch/bench_fast.py` (synthetic input, seq_len=50, fp32):

| Backend | Config | ms / forward | Source |
|---|---|--:|---|
| PyTorch | fp32, seq_len=50 | 72.2 | `src/vggt/autoresearch/.pt_baseline.json` |
| JAX (eager) | baseline | 4698.5 | `results.tsv` |
| JAX (jit camera_head + padded KV) | promoted `90e123d` | 137.6 | `results.tsv` |

JAX is currently **1.9× slower than PT-fp32** on the apples-to-apples bench, despite weeks of iteration in the #78 race. The aggregator-JIT attempt (`f30273b`) is marked "WIP, integration drift". Against PT-bf16+compile in production (~95 ms streaming bench), the gap widens further.

### Decisions

1. **#88 closed as won't-fix.** wrapper + jax_upload = 0.69 ms / forward 148 ms = 0.47% of the bottleneck. Both the original (2026-04-20) and re-verified (2026-04-25) data agree.
2. **JAX port reframed.** Speedup-vs-PT is no longer the primary motivation; **codebase uniformity + trainable heads** (e.g. semantic head #8) is. New parity threshold is "within 1.5× of PT-compile", not "must be faster". Tracked in a new issue (supersedes #72 stub).
3. **Frame-skip is the throughput lever.** 5.9 FPS today × 72 h = 1.53 M steps; 2.4 M target needs 9.3 FPS = ~1.6× more. K=2 frame-skip projects 2× and ships in ~5 LOC at `src/r2dreamer/adapters/vggt_adapter.py:33-36`. Filed as a new issue.

## Update 2026-05-04 — `torch.compile` mode sweep

Frame-skip was rejected as thesis-unsafe (the agent must see every step). The next zero-method-change lever is to try non-default `torch.compile` modes on the same PyTorch path. Extended `VGGTFeatureExtractor.__init__` with a `compile_mode: str | None = None` argument, plumbed through `profile_training.py --compile-mode` and `benchmark_streaming.py --pt-compile-mode`. `None` = torch's default mode (preserves the 2026-04-20 behaviour).

Bench harness: `src/vggt/jax/benchmark_streaming.py --backends pytorch --seq-lens 10 50 100`, run as **four separate `uv run` invocations** (one per mode) on the local H100 to avoid compile-cache pollution. Each invocation: 3 warmup frames + n measured frames per `seq_len` with a fresh extractor.

Selection rule: pick winner on **n=100 median** (closest to a steady-state production run; n=10 / n=50 are warmup-dominated for compiled paths because torch.compile takes the first frames to trace).

### Results — n=100 median latency per mode

| Mode | n=100 mean (ms) | n=100 **median (ms)** | Δ vs no-compile (median) | CSV |
|---|--:|--:|--:|---|
| no compile | 136.7 | 160.6 | — (baseline) | `output/methods/vggt-jax-latency/compile-modes/vggt_streaming_20260504_091152.csv` |
| `default` | 123.0 | **144.8** | **−15.8 ms (−9.8%)** | `output/methods/vggt-jax-latency/compile-modes/vggt_streaming_20260504_091446.csv` |
| `reduce-overhead` | — | — | **FAILED** | `output/methods/vggt-jax-latency/compile-modes/vggt_streaming_20260504_091724_reduce_overhead_FAILED.csv` |
| `max-autotune` | — | — | **FAILED** | `output/methods/vggt-jax-latency/compile-modes/vggt_streaming_20260504_092038_max_autotune_FAILED.csv` |

Both `reduce-overhead` and `max-autotune` use **CUDA graphs** under the hood. CUDA graphs are incompatible with the InfiniteVGGT streaming KV-cache pattern: the `camera_head.trunk_fn` returns a tensor (`activated_pose = torch.cat([T, quat, fl], dim=-1)`) whose storage gets overwritten by the next graph replay, so reading `camera_pose.cpu()` after the next forward call raises:

> `RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run … To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.`

The same trace happens in both modes (camera_head.py:94 → head_act.py:24). Fixing this would require either an extra `.clone()` on every cached tensor (defeating reduce-overhead's purpose) or a `cudagraph_mark_step_begin()` call inside `extract()` (cuts further into the savings). Both are method changes outside this lever's scope.

### Winner & speedup

- **Winner: `default`** — by elimination, since the two CUDA-graph modes are incompatible with this model.
- vs **no-compile**: −9.8% on n=100 median.
- vs the **2026-04-20 / 04-25 numbers** (reported as ~−11% on `vggt_forward` p50 inside the full training loop, 168 → 150 ms): consistent, since `default` is exactly the mode that 04-20 ran under (`compile_mode=None` = torch default = the only one tested back then). The bench harness measures the full extractor (load + forward + downsample + .cpu()), so the −9.8% wall-clock gain matches the −11% forward-only p50 gain to within noise.
- The n=10 and n=50 medians visibly include compile-trace warmup overhead (e.g. `compile-default` has 61.8 ms median at n=10 — that is "most frames after warmup are fast") and are **not used** for the steady-state decision per the selection rule.

### Recommendation for L4 production sbatch

- Keep `--compile` enabled.
- **Do not pass `--compile-mode`** (i.e. leave `compile_mode=None`, the default, which forwards no `mode=` kwarg to `torch.compile`). This is identical to passing `--compile-mode default`; we keep `None` as the wire-format default so the production sbatch doesn't have to change at all.
- Do **not** flip to `reduce-overhead` or `max-autotune` until the CUDA-graph aliasing in `streamvggt/heads/camera_head.py` is fixed upstream — the failure is deterministic on the first non-warmup frame and would crash an L4 run.

### Compile-time costs

- `default`: ~30 s tracing on the first forward, then steady-state. Already paid in current production runs.
- `reduce-overhead` / `max-autotune` (had they worked): documented to take **minutes** on the first call while CUDA graph capture and (for max-autotune) Triton kernel autotuning runs. Not relevant here since both fail before reaching steady state.

### Stack progress

L4 cycle target: ~4 d → ~3 d. With **only** `--compile` (default mode), projected gain is ~10% on the bottleneck phase, which lands at ~3.6 d, not ~3 d. **This lever alone is insufficient to hit the target.** The remaining gap must come from orthogonal levers (FastVGGT, distillation, async pipeline) — frame-skip remains rejected.

### Files touched

- `src/vggt/feature_extractor.py:34-43,52-58,92-111` — added `compile_mode` arg, validated against `{None, "default", "reduce-overhead", "max-autotune"}`, forwarded as `mode=` to all three `torch.compile` calls when not `None`.
- `scripts/r2dreamer/profile_training.py` — new `--compile-mode` CLI option on the existing `--compile` flag (no new compile flag introduced).
- `src/vggt/jax/benchmark_streaming.py` — new `--pt-compile-mode {default,reduce-overhead,max-autotune}` CLI option; CSV `config` column records the resolved mode (e.g. `bf16-autocast+compile-default`).
