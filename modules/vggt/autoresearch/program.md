# autoresearch: JAX VGGT forward-pass speedup

This is an experiment to have the LLM optimize the JAX port of VGGT for forward-pass throughput.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date, lowercase abbreviated (e.g. `apr23`, `may07`). The branch `autoresearch/vggt-jax-<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/vggt-jax-<tag>` off `feat/vggt-jax-port` (NOT main).
3. **Verify the HF checkpoint cache**: check that `/tmp/vggt_test_cache` exists. If not, tell the human to run the parity tests once manually — that populates the cache.
4. **Cache the PyTorch baseline**: run `uv run python -m modules.vggt.autoresearch.bench_fast --setup` ONCE. This caches the PyTorch baseline timing so every iteration measures only JAX.
5. **Initialize results.tsv**: create `modules/vggt/autoresearch/results.tsv` with this tab-separated header (8 columns):
   ```
   commit	jax_ms	pt_ms	speedup	parity_maxabs	peak_mem_gb	status	description
   ```
6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The fitness metric is `speedup = pt_baseline_ms / jax_median_ms` at n=50 frames. Higher speedup is better. Target: speedup >= 1.0 means JAX beats the PyTorch baseline. You launch a run simply as: `uv run python -m modules.vggt.autoresearch.bench_fast`.

**What you CAN do — modify exactly these 6 files:**
- `modules/vggt/jax/attention.py`
- `modules/vggt/jax/block.py`
- `modules/vggt/jax/aggregator.py`
- `modules/vggt/jax/backbone.py`
- `modules/vggt/jax/rope.py`
- `modules/vggt/jax/feature_extractor.py`

Everything inside those files is fair game: kernels, layouts, dtypes, fusion, sharding, compile flags, attention implementations, rope formulation, scan vs unroll, etc.

**What you CANNOT do:**
- Modify `modules/vggt/jax/weight_transfer.py`. Changing this breaks parity silently — the weight-loading invariant must be untouchable.
- Modify anything under `modules/vggt/tests/`. The parity tests are the referee; they cannot rewrite the rules they are graded by.
- Modify `modules/vggt/jax/benchmark_streaming.py`. It defines the metric for guardrail sweeps.
- Modify `modules/vggt/autoresearch/bench_fast.py`. It defines the fitness metric.
- Modify `modules/vggt/autoresearch/program.md` (this file).
- Relax `jax_default_matmul_precision='highest'` in the parity test. That is the correctness invariant.
- Add packages, edit `pyproject.toml`, or install anything. You use only what is already installed.

**The goal is simple: maximize `speedup` at n=50 frames while keeping parity green.** Peak memory is a soft constraint — some increase is acceptable for meaningful speedup, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 speedup improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 speedup improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: your first run establishes the baseline. No code changes. Just run `bench_fast` and record row 1 with `status=baseline` and description `eager fp32 baseline (Step 8 HEAD)`. If `smoke.sbatch` already wrote that baseline row, skip this and start with a real experimental change.

## Output format

Once `bench_fast` finishes it prints a summary. Extract the key lines:

```
grep "^speedup:\|^jax_median_ms:\|^pt_baseline_ms:\|^parity_maxabs:\|^peak_mem_mb:" run.log
```

Expected lines:

```
speedup:          1.21
jax_median_ms:    98.1
pt_baseline_ms:   118.7
parity_maxabs:    3.1e-6
peak_mem_mb:      8601.6
```

## Logging results

When an experiment is done, append a row to `modules/vggt/autoresearch/results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 8 columns:

```
commit	jax_ms	pt_ms	speedup	parity_maxabs	peak_mem_gb	status	description
```

1. git commit hash (short, 7 chars)
2. jax_median_ms (e.g. 98.1) — use 0.0 for crashes
3. pt_baseline_ms (e.g. 118.7) — use 0.0 for crashes
4. speedup (e.g. 1.21) — use 0.0 for crashes
5. parity_maxabs (e.g. 3.1e-6) — write `n/a` if not measured this iteration
6. peak memory in GB, round to .1f (divide peak_mem_mb by 1024) — use 0.0 for crashes
7. status: one of `baseline`, `keep`, `discard`, `crash_parity`, `crash_bench`, `crash_full`
8. short text description of what this experiment tried

Example:

```
commit	jax_ms	pt_ms	speedup	parity_maxabs	peak_mem_gb	status	description
a1b2c3d	142.3	118.7	0.83	3.2e-6	8.4	baseline	eager fp32 baseline (Step 8 HEAD)
b2c3d4e	98.1	118.7	1.21	3.1e-6	8.4	keep	jax.nn.dot_product_attention
c3d4e5f	0.0	118.7	0.0	n/a	0.0	crash_parity	bf16 without rescaling rope
```

Do NOT `git add` results.tsv — leave it untracked.

## The experiment loop

The experiment runs on the dedicated branch `autoresearch/vggt-jax-<tag>`.

LOOP FOREVER. Each iteration is exactly 8 steps:

1. **Inspect state.** Look at git state: current branch, last commit. Run `tail -5 modules/vggt/autoresearch/results.tsv` to see recent experiments.
2. **Propose one idea.** Pick ONE experimental idea. Edit 1-3 files in the CAN list. Keep the diff small.
3. **Fast parity gate.**
   ```
   uv run pytest modules/vggt/tests/test_jax_parity.py::TestLevel1WeightTransfer modules/vggt/tests/test_jax_parity.py::TestLevel2SharedBlockParity -q
   ```
   If fail: `git restore modules/vggt/jax/`, append a `crash_parity` row, return to step 1.
4. **Bench.**
   ```
   uv run python -m modules.vggt.autoresearch.bench_fast > run.log 2>&1
   ```
5. **Parse.**
   ```
   grep "^speedup:\|^jax_median_ms:\|^peak_mem_mb:" run.log
   ```
   If the grep is empty, the run crashed. Run `tail -n 50 run.log`, attempt ONE fix, re-run. If still broken: `git restore modules/vggt/jax/`, append a `crash_bench` row, return to step 1.
6. **Compare vs best.** Find the best `speedup` value (column 4) across all `keep` rows in `results.tsv` so far.
   - If `new_speedup <= best_speedup * 1.02`: `git restore modules/vggt/jax/`, append a `discard` row, return to step 1.
   - Else continue.
7. **Full parity gate.**
   ```
   uv run pytest modules/vggt/tests/ -q
   ```
   If fail: `git restore modules/vggt/jax/`, append a `crash_full` row, return to step 1.
8. **Commit + log.** `git commit -am "<concise description>"`. Append a `keep` row to `results.tsv`. Return to step 1.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. You are advancing the branch so that you can iterate.

## Advisor — when to ask Opus for help

Escalate to Opus in exactly three situations, each enumerated separately:

**(a) Three consecutive `discard` rows in `results.tsv`.** You are stuck and need new ideas.

**(b) A parity failure you cannot diagnose in 2 fix attempts.** A numerical mystery — get another set of eyes.

**(c) A planned change would touch >50 lines OR >2 files.** Big commit, get the plan reviewed before burning compute.

**Invocation.** Use the `Agent` tool with `subagent_type: "general-purpose"` and `model: "opus"`. The prompt MUST be self-contained — Opus has no memory of this run. Include:
- the last 5 rows of `results.tsv`
- `git diff` of the last `keep` commit (for triggers a and b) OR the planned change (for trigger c)
- your current hypothesis in one sentence

**Required response format for Opus** — embed this verbatim in your prompt to enforce it:
```
HYPOTHESIS: <one sentence — what's bottlenecking>
EXPERIMENT: <concrete code change, <= 20 lines>
FALSIFIABLE BY: <what number tells us if the hypothesis was right>
```

This format is load-bearing. If Opus returns an essay instead, re-prompt and demand the three-line format.

## Guardrail — run every ~10 keep commits

After every ~10 `keep` commits, run once:

```
uv run python -m modules.vggt.jax.benchmark_streaming --seq-lens 10 50 100 --backends jax > guardrail.log
```

Verify that the n=10 and n=100 medians have not regressed more than 20% vs the baseline sweep. If they have, note it in `results.tsv` and consider reverting a recent change.

## Timeout

Each experiment should take a few minutes total. The outer SLURM wrapper enforces a **hard 15-minute cap per iteration via `timeout(1)`** — if your iteration exceeds 900 seconds, the outer process will kill your invocation and advance. Within an iteration, if a single pytest / bench call has been running for more than 8 minutes, kill it yourself, treat it as a failure (discard and revert via `git restore modules/vggt/jax/`), and move on.

## JIT compilation — expect a one-time cost

First-time `jax.jit` compilation of the full aggregator at `matmul_precision=highest` on an H100 can take **5-15 minutes** (XLA tracing, HLO optimization, kernel selection). This cost is real but amortized across subsequent invocations. Two rules to keep jit experiments inside the 15-min iteration cap:

- **AOT-compile inside the extractor's `__init__` or a one-time warmup, NOT inside the timed bench.** Idiom:
  ```python
  self._fn = jax.jit(self._forward)
  # Warmup call with a dummy input so compilation happens before anything timed
  _ = self._fn(dummy_input).block_until_ready()
  ```
- **The fast parity gate also triggers compilation** (once, on the first test that hits a new shape). Its ~45s budget assumes eager. When you enable jit, the first fast-gate run may take 5-10 min — this is expected and doesn't count as a "slow experiment."

## Crashes

If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log the appropriate `crash_*` status in the tsv, and move on.

## NEVER STOP

Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you a few minutes then you can run many dozens per night. The user then wakes up to experimental results, all completed by you while they slept.
