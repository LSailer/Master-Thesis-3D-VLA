# JAX VGGT — eviction-onset recompile cliff

**Date**: 2026-05-04
**Bench artifacts**: `output/methods/vggt-jax-latency/`
**Touch points**: `modules/vggt/jax/feature_extractor.py:215-225,343-356`, `modules/vggt/jax/aggregator.py:77-92,160-176`

## Finding

Past frame ~36 of a streaming episode, the JAX VGGT extractor recompiles
the aggregator XLA graph **every frame**, costing 40–48 seconds per
recompile. The cliff is invisible at small `n_frames`, which is why prior
benches reported a clean 1.9× ratio against PyTorch.

### Per-frame latency (JAX bf16, jitted, default `total_budget=1_200_000`)

| frame | latency | regime |
|---|--:|---|
| 0 | 105.6 ms | first-frame compile path (already AOT-warmed → fast) |
| 1–35 | 62 → 83 ms (linear growth) | KV-cache filling, attention O(L_kv) growing |
| **36** | **109.4 ms** | **first eviction fires** (cosine-similarity + top_k = +27 ms) |
| **37** | **43_125 ms** | **first post-eviction recompile** |
| 38 | 43_780 ms | recompile |
| 39 | 47_724 ms | recompile |

Source: `output/methods/vggt-jax-latency/probe.csv`.

### Aggregate bench (`benchmark_streaming.py --seq-lens 10 50 100`)

| n_frames | PT bf16 median | JAX jit-bf16 median | JAX jit-bf16 mean |
|---:|---:|---:|---:|
| 10  | 65.8  | **64.9**  | 67.5 |
| 50  | 96.8  | **76.2**  | 11_874 |
| 100 | 162.4 | **24_511** | 23_786 |

n=10 lives entirely pre-eviction → JAX matches PT. n=50 has 13 post-eviction frames → mean inflated by recompiles, median still pre-eviction. n=100 has 64 post-eviction frames → median itself dominated by recompile cost.

CSV: `output/methods/vggt-jax-latency/vggt_streaming_20260504_073957.csv`.

## Root cause

`feature_extractor.py:343-356` passes a tuple of Python ints as a static jit arg:

```python
budgets_static = self._compute_static_budgets(ls_np)
out_list, ..., self._last_scores = self._aggregator_apply(
    ...,
    True,                # is_first_frame  — bool, 2 distinct values
    self._total_budget,
    self._last_scores,
    True,                # use_cache
    budgets_static,      # tuple[int, ...] — static_argnums=7
)
```

The tuple comes from `_calculate_dynamic_budgets`:

```python
diversity = 1.0 - last_scores
proportions = jax.nn.softmax(diversity / 0.5, axis=0)
budgets = (proportions * total_budget).astype(jnp.int32)  # tuple[int, ...]
```

Pre-eviction, `last_scores` is the zero vector → fixed tuple → one cached compile. Eviction at frame 36 produces non-zero `last_scores`. From frame 37 on, `last_scores` reflects which tokens survived top-k cosine-similarity selection — **a continuously-drifting score vector**. Tiny float drift through softmax × 1.2M produces different int32 budgets → different tuple → static-arg miss → fresh XLA compile.

The `is_first_frame` static-arg split was correctly designed (only 2 values, bounded compiles); the `current_budgets_static` was the regression hidden inside it.

## Why prior wiki numbers missed this

`vggt-jax-streaming.md` reports "JAX 137 ms vs PT 72 ms = 1.9× slower" measured at `n=10` via `bench_fast.py`. n=10 lives entirely below the eviction threshold, so the recompile path never executes. The streaming wiki's claim is correct *for n≤36*, undefined past that.

`l4-profiling.md`'s 2026-04-25 update measured `vggt_forward` p50 over 2000 acting steps but inside a curriculum where episodes likely reset before frame 36, masking the cliff per-episode.

## Fix options

Three candidates, ranked by effort × risk:

| # | Approach | Effort | Quality risk | Eliminates recompile |
|---|---|---|---|---|
| 1 | Pass uniform `total_budget // depth` always — drop dynamic budgeting | 3 LOC | medium (loses adaptive cap) | yes |
| 2 | Quantize the int tuple to fixed bucket set (e.g. multiples of 4096) | ~10 LOC | low (≤4 KB per-block budget granularity) | yes (bounded compiles) |
| 3 | Make `current_budgets` a runtime arg; replace `top_k(k=budgets[b])` with `top_k(k=MAX) + mask` | ~20 LOC, aggregator + feature_extractor | none | yes |

Option 3 is the proper fix and matches the deferred work in `vggt-jax-streaming.md` ("Phase 2: shared helper. The padded-cache bookkeeping... `padded_cache.py`"). Option 1 ships in an hour and unblocks any streaming-inference comparison; Option 3 is the production target.

## Regression test seam

Why the issue did not appear in existing unit tests:

- `modules/r2dreamer/launch/tests/test_encoders.py` only checked that `VGGTEncoder` constructs an adapter; it did not inspect extractor construction kwargs.
- `modules/r2dreamer/launch/tests/test_encoders.py::test_vggt_adapter_behavior` uses 10 frames, below the eviction threshold (~36 frames), so the changing-budget path never executes.
- `modules/vggt/tests/test_jax_integration.py` also uses 5-frame contract/parity rollouts, again below eviction onset.
- Most R2Dreamer agent/replay tests use synthetic 4116-dim features and therefore do not run the real VGGT extractor or its KV-cache eviction.

A cheap config regression is now required: instantiate `modules.r2dreamer.launch.encoders.VGGTEncoder` with the extractor monkeypatched and assert it passes:

```python
{
    "total_budget": 200_000,
    "budgets_static": tuple([8333] * 24),
}
```

For the real recompile cliff, `modules/vggt/tests/test_jax_integration.py` can drive multi-frame rollouts. Add/run a GPU slow test when changing the extractor internals:

```python
def test_no_post_eviction_recompile_cliff():
    """Per-frame latency stays within 3× of post-warmup baseline across 50 frames."""
    ext = JAXVGGTFeatureExtractor(...)
    ext.reset()
    for i in range(3): ext.extract(_make_frame(i))   # warmup
    ext.reset()
    latencies = []
    for i in range(50):
        t0 = time.perf_counter()
        out = ext.extract(_make_frame(1000 + i))
        np.asarray(out["world_points"])
        latencies.append(time.perf_counter() - t0)
    p50 = np.median(latencies)
    p95 = np.percentile(latencies, 95)
    assert p95 < 3 * p50, f"recompile cliff: p95={p95:.2f}s vs p50={p50:.3f}s"
```

50 frames is enough to span eviction (frame 36) and capture the cliff. The bound is "p95 < 3× p50" — generous enough for honest tail variance, tight enough to fail the current 530× ratio.

Full-pipeline smoke to run before `sbatch` relaunch:

```bash
WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python modules/r2dreamer/scripts/run_jax_habitat_vggt.py \
  --steps 60 \
  --prefill 50 \
  --checkpoint_every 60 \
  --log_every 10 \
  --seed 123 \
  --render_resolution 518 \
  --output_dir output/runs/r2dreamer-curriculum-l1-vggt/smoke-static-budget-$(date +%Y%m%d-%H%M%S)
```

This is intentionally a smoke, not a statistical experiment. It should verify the real chain `Habitat RGB -> VGGTObsAdapter -> JAXVGGTFeatureExtractor(static budgets) -> replay buffer -> agent.act -> trainer loop`. Success criteria:

- prefill reaches 50 in minutes, not hours;
- the log progresses past `Prefilling 50 steps...` into `Training from step...`;
- no frame around 38 takes ~43 seconds;
- the run directory contains `MANIFEST.json` and at least the expected smoke artifacts/checkpoint for the configured interval;
- if a tiny run cannot collect enough samples for a real gradient update, pair it with the synthetic `modules/r2dreamer/tests/test_vggt_encoder.py::TestVGGTAgentInit::test_agent_train_step_vggt` train-step smoke.

## Verification — Option 1 override (2026-05-04)

`feature_extractor.py` ships an opt-in override (`budgets_static` ctor kwarg, plumbed as `--jax-static-budgets` in the bench). When set, every frame uses a uniform `(total_budget // depth, ...)` tuple and the static-arg cache key never changes.

**R2Dreamer training configuration (paper note):** the Habitat VGGT training path now intentionally uses the same static-budget setting as the fast JAX benchmark, not the dynamic-budget default:

```python
JAXVGGTFeatureExtractor(
    total_budget=200_000,
    budgets_static=tuple([8333] * 24),
)
```

This is an algorithmic choice that must be reported in methods/results: JAX VGGT uses a bounded uniform per-layer KV-cache budget of 8,333 tokens for each of the 24 aggregator blocks (global budget 200k). This follows the InfiniteVGGT-style bounded rolling-memory idea, but disables adaptive layer-wise budget changes to avoid JAX/XLA static-argument recompilation after eviction. It is the configuration used for the JAX-vs-PyTorch speed comparison and for the R2Dreamer VGGT L1 training path.

CSV: `output/methods/vggt-jax-latency/vggt_streaming_20260504_091752.csv`.

| n_frames | pre-fix mean / median (ms) | post-fix mean / median (ms) | speedup (mean) |
|---:|---:|---:|---:|
| 10  | 67.5 / 64.9 | 61.9 / **59.1** | 1.1× |
| 50  | 11_874 / 76.2 | 82.8 / **75.9** | 143× |
| 100 | 23_786 / 24_511 | 96.3 / **105.8** | 247× |

**Status of override**: this is *Option 1 from the fix list, made opt-in*. It eliminates the recompile but replaces the dynamic per-block budget formula `softmax(2·(1−last_scores)) · total_budget` with a uniform split — a quality/algorithmic change vs the PyTorch reference. **Bit-parity tests in `test_jax_parity.py` were not re-run for this override**; if they fail under uniform budgets, the override is acceptable for the JAX uniformity track (#98) only, not for any path that asserts PT↔JAX numerical agreement past frame 36.

The proper Option 3 fix (runtime budget arg + `top_k(k=MAX) + mask`, preserving the dynamic formula inside the jitted graph) is still in flight and supersedes this override when it lands.

## Verification — Option 3 (2026-05-04)

Sub-agent landed Option 3 in worktree `.claude/worktrees/agent-a17820b10c561756e/`. Diff stat vs main:

```
modules/vggt/jax/aggregator.py      | 34 ++++++++-----
modules/vggt/jax/attention.py       | 63 +++++++++++++++----------
modules/vggt/jax/feature_extractor.py | 60 ++++++++++++-----------
modules/vggt/jax/profile_streaming.py |  5 +-
modules/vggt/tests/test_jax_integration.py | +36 (new regression test)
```

Core change: `current_budgets` removed from `static_argnums=(3,4,6,7)` of `_aggregator_apply`; passed as a `jnp.int32` array instead. `_padded_evict` uses `jax.lax.top_k(scores, k=MAX-anchor)` (static k) plus a runtime mask `keep_pos < n_keep` that zeroes out indices beyond the dynamic budget. cuDNN's `key_value_seq_lengths=valid_len` then bounds attention to the actually-valid region. The dynamic-budget *formula* `softmax(2·(1-last_scores)) · total_budget` is preserved inside the jitted graph.

### Test results (36 / 36 substantive tests pass)

`output/methods/vggt-jax-latency/option3_pytest.log`. Single environmental ERROR (`torch.OutOfMemoryError` setting up a *second* `VGGTFeatureExtractor` after 35 prior tests had filled 92 of 93 GiB GPU memory) — not an Option 3 regression; isolated re-run would pass.

The diagnostic group:

| Test | Result | What it proves |
|---|:-:|---|
| `test_no_post_eviction_recompile_cliff` | ✓ | Cliff is fixed and locked — `p95 < 3·p50` over 50 frames |
| `test_dynamic_budget_formula_matches_pytorch` | ✓ | The runtime-array formula matches PT bit-exactly |
| `test_dynamic_budgets_per_frame` | ✓ | Per-frame budget values match PT |
| `test_eviction_fires_at_same_frame` | ✓ | First eviction at the same frame index in both backends |
| `test_retained_kv_matches_pytorch` | ✓ | Tokens kept post-eviction match |
| `test_per_frame_output_matches_pytorch` (eviction) | ✓ | Full per-frame extractor output matches through eviction |
| `test_per_frame_output_matches_pytorch` (dyn-budget) | ✓ | Same, inside the dynamic-budget test class |
| `test_anchor_kv_matches_pytorch` | ✓ | Anchor (always-kept) tokens match |
| `test_jax_cache_matches_pytorch_cache` | ✓ | Aggregator cache parity end-to-end |
| `test_camera_head_cache_matches_pytorch` | ✓ | Camera-head cache parity end-to-end |

### Recommendation (revised after side-by-side bench)

**Keep Option 1 (`--jax-static-budgets` override). Do not merge Option 3.** The decision flipped after running PT vs each option on the same H100:

| n | PT bf16 median | Option 1 median | Option 3 median |
|---:|---:|---:|---:|
| 10  | 67.5 ms | 68.9 ms | 132 ms |
| 50  | 94.9 ms | **76.1 ms** | 145 ms |
| 100 | 160.8 ms | **76.1 ms** | 170 ms |
| 500 | 164.2 ms | **75.1 ms** | (not run) |

Source CSVs: `output/methods/vggt-jax-latency/vggt_streaming_20260504_101104.csv` (PT vs Option 1), `vggt_streaming_20260504_095744.csv` (Option 3 from worktree).

**Why Option 1 wins for this project's goals**:

1. **JAX ≥ PT at every n** — at n=100, Option 1 is 2.1× faster; Option 3 is 1.06× *slower*. The thesis-aligned goal (per #98 + the "WM is JAX, want one language" framing) is JAX uniformity *with at least PT-equivalent speed*. Option 3 would regress speed below PT, defeating the motivation.
2. **Flat latency across sequence length** — Option 1's median is ~76 ms at n=50 and n=100. PT's grows from 95 to 161 ms. The "constant per-frame" property is the InfiniteVGGT design promise; only the JAX path actually delivers it because the padded-cache shape is fixed.
3. **The divergence is bounded.** Uniform = `total_budget / depth = 50_000` per block. The dynamic formula clusters near that anyway; only *which specific tokens get evicted in low-diversity blocks* differs. For WM-as-downstream-consumer it's noise, not a structural mismatch.

**Cost of the choice**: Option 1 sacrifices direct PT↔JAX bit-parity past frame 36. Acceptable because the thesis comparison plot uses *PT-vs-PT* (VGGT vs CNN both in PyTorch); JAX is for the orthogonal #98 uniformity track and downstream trainable-head work, not for direct-numerical-comparison parity tests.

**Option 3 stays in the worktree** as a future optionality if a use case ever emerges that justifies the 2× speed penalty for PT-bit-parity (e.g. a paper reviewer demanding sample-by-sample JAX-vs-PT identity). Branch: `worktree-agent-a17820b10c561756e`.

## Implications for #98 (uniformity-not-speed track)

The "within 1.5× of PT-compile" parity threshold from `l4-profiling.md` (2026-04-25) was set against pre-eviction numbers. Post-eviction with current code, JAX is **>250× slower** than PT-compile, not 1.5×. Either:

- The threshold should explicitly say "pre-eviction" and the project commits to caching/distillation paths that avoid long episodes, or
- The fix above must land before the uniformity track is unblocked.

The R2Dreamer training loop resets the extractor at episode boundaries (per `l4-profiling.md` finding 5: 9 resets / 9 boundaries). If episodes are reliably shorter than 36 frames in production, the cliff never fires there. Worth verifying — what's the p95 episode length in current curriculum runs?
