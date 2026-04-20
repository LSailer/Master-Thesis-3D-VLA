# Plan: L4 pipeline profiling + KV-cache audit

> Source PRD: #74
> Related stubs: #72 (VGGT JAX port — blocked by findings here), #73 (paper deliverables — independent)

## Architectural decisions

These are locked for the whole plan — do not re-open in later slices.

- **Timer primitives**:
  - PyTorch/CUDA phases (`vggt_forward`, `vggt_wrapper`): `torch.cuda.Event` pairs. Start + end events on the current stream, read via `end.elapsed_time(start)` after `torch.cuda.synchronize()`. GPU-side timing, no CPU overhead in the fast path.
  - JAX phases (`jax_upload`, `wm_inference`, `wm_training`): `time.perf_counter()` brackets around a call that ends with `.block_until_ready()` on the returned array. Without `block_until_ready`, async-dispatch times only dispatch, not execution.
  - CPU-only phases (`env_step`, `buffer_add`): plain `time.perf_counter()`.
- **Instrumentation placement**: optional `phase_times: dict[str, list[float]] | None = None` kwarg on `VGGTFeatureExtractor.extract()`. When `None` (production path) behavior is bitwise unchanged.
- **Phase list (7, final)**: `env_step`, `vggt_forward`, `vggt_wrapper`, `jax_upload`, `wm_inference`, `buffer_add`, `wm_training`. CNN runs have zeros for `vggt_*` and `jax_upload`.
- **Encoder dispatch**: `--encoder {vggt,cnn}` flag on the profile script. Two runs are invoked separately and their JSONs merged at report time.
- **Output**: per-run JSON at `output/profiling/vggt_vs_cnn_<timestamp>.json` (raw per-call timings + aggregates) + stdout table (mean / p50 / p95 / delta_ms).
- **Runner**: `modules/r2dreamer/scripts/profile_training.py` — reimplements the loop body inline rather than wrapping `Trainer`, because CUDA-Event timing must bracket individual sub-steps. Keeps `Trainer` untouched.
- **Run length defaults**: `prefill_steps=2000`, `acting_steps=2000`. CLI-configurable.
- **Execution**: interactive H100 session. No SLURM sbatch — the whole run is ~15 min wall time.

---

## Phase 1: CNN-path plumbing

### What to build

An end-to-end profile script that runs the training loop for **CNN only** and produces the full output artifacts (JSON + stdout table), but with only the 4 CNN-path phases populated: `env_step`, `wm_inference`, `buffer_add`, `wm_training`. This validates every piece of the pipeline (timer plumbing, JSON schema, stdout formatting, CLI handling) against the simpler encoder before touching `feature_extractor.py`.

Files created:
- `modules/r2dreamer/scripts/profile_training.py` with:
  - `PhaseTimer` (context manager, two modes: `cpu` and `jax_block`)
  - `RunResult` dataclass (phase_times dict-of-lists, kv_audit counts, episode stats)
  - `run_loop(encoder: str, prefill_steps: int, acting_steps: int, ...) -> RunResult`
  - `aggregate(results: RunResult) -> dict` (mean/p50/p95 per phase)
  - `format_report(results_by_encoder: dict) -> str`
  - `save_json(results_by_encoder: dict, output_dir: Path) -> Path`
  - `main()` parses CLI, runs one encoder, saves JSON, prints table

KV-cache audit counters are wired in the loop (reset_count, boundary_count) but not yet asserted — the CNN path has no `extractor.reset()` so the counts are zero.

### Acceptance criteria

- [ ] `uv run python modules/r2dreamer/scripts/profile_training.py --encoder cnn --prefill_steps 200 --acting_steps 200` completes end-to-end on the H100 session.
- [ ] JSON written to `output/profiling/vggt_vs_cnn_<timestamp>.json` containing keys: `{"cnn": {"phase_times": {...}, "aggregate": {...}, "kv_audit": {...}, "episodes": {...}}}`.
- [ ] Stdout table prints 7 rows (with vggt rows showing `—` / zeros for CNN-only run) and columns: `phase, cnn_mean_ms, cnn_p50_ms, cnn_p95_ms, vggt_mean_ms, vggt_p50_ms, vggt_p95_ms, delta_ms`.
- [ ] `wm_inference` and `wm_training` phase values are non-zero and reasonable (wm_training fires only after buffer reaches `batch_size * seq_len`).
- [ ] `block_until_ready` is called on every JAX-phase measurement — verified by reading the code, no per-call `<1µs` artifacts in the timing distributions.

---

## Phase 2: Add VGGT path + KV-cache audit

### What to build

Extend the script to cover the VGGT encoder and the three PyTorch/boundary phases. Touch `feature_extractor.py` once — add the `phase_times` kwarg — and use it from the loop.

Files modified:
- `modules/vggt/feature_extractor.py`:
  - Add optional `phase_times: dict[str, list[float]] | None = None` parameter to `extract()`.
  - When provided, create two `torch.cuda.Event` pairs: one around the aggregator+camera_head+point_head forward (`vggt_forward`), one around the pool+permute+`.cpu().numpy()` (`vggt_wrapper`). Record `event.elapsed_time` / 1000 into the list (convert ms → s or keep ms consistent with JAX phases; pick one unit and document).
  - When `None`, no events created — production path is branchless.

Files extended:
- `modules/r2dreamer/scripts/profile_training.py`:
  - `--encoder vggt` branch: constructs `VGGTFeatureExtractor` + `VGGTObsAdapter`, passes `phase_times` into each `extract()` call.
  - Three new phases wired: `vggt_forward` and `vggt_wrapper` (populated by the kwarg), `jax_upload` (measured in the loop by bracketing `jnp.asarray(features).block_until_ready()` *before* `agent.act`).
  - KV-cache audit becomes active: `reset_count` increments each time `obs_adapter.on_episode_reset` fires, `boundary_count` increments on `is_first=True` transitions. At end of loop, assert `reset_count == boundary_count + 1` (the +1 is the initial env.reset before the loop). On failure, raise `AssertionError` with both counts + episode list — the script exits non-zero.
  - The `_flatten_vggt` call is timed inside the `vggt_wrapper` span (via `perf_counter`, merged into the same phase list slot — the CUDA event covers the `.cpu().numpy()`, perf_counter adds the tiny numpy reshape/concat).

Files unchanged (guaranteed):
- `Trainer`, `R2DreamerAgent`, buffer, env wrappers — the profile script reimplements the loop body inline.

### Acceptance criteria

- [ ] `uv run python modules/r2dreamer/scripts/profile_training.py --encoder vggt --prefill_steps 200 --acting_steps 200` completes end-to-end.
- [ ] JSON now contains keys for both `cnn` and `vggt` once both runs have been invoked (two separate invocations; script appends to the same run file by reading-then-merging, OR emits per-encoder JSON and `format_report` loads both — choose simpler, pick at implementation time).
- [ ] All 7 phases populated for VGGT: `env_step`, `vggt_forward`, `vggt_wrapper`, `jax_upload`, `wm_inference`, `buffer_add`, `wm_training`.
- [ ] KV-cache audit assertion passes under curriculum scene switching: `reset_count == boundary_count + 1`. If it fails, script exits non-zero and the mismatch is printed.
- [ ] `pytest modules/vggt/tests/test_feature_extractor.py` passes unchanged (production path behavior preserved).
- [ ] VGGT stdout table shows the 7 rows with non-zero values for all phases, and a non-zero `delta_ms` column versus the CNN run.

---

## Phase 3: Data + written conclusion

### What to build

Execute the full diagnostic on the H100 session (using the real curriculum configs — L1-VGGT for VGGT, stock L1 for CNN, at `prefill=2000, acting=2000`) and document the findings.

Files created:
- `docs/wiki/methods/l4-profiling.md`:
  - **Hardware**: H100 node name, GPU name from `nvidia-smi`, driver/CUDA version.
  - **Run config**: encoder, prefill_steps, acting_steps, curriculum, seed, timestamp.
  - **Results table**: the stdout table verbatim + a one-paragraph interpretation per row (which phase dominates, which surprises vs. expectation).
  - **KV-cache audit**: reset_count, boundary_count, pass/fail.
  - **Recommendation for #72**: go/no-go on the VGGT JAX port. Concretely: if `vggt_wrapper + jax_upload` together account for >30% of per-step wall time, #72 is go. If `vggt_forward` dominates independently, #72 won't help enough and the follow-up is instead a distilled/smaller encoder (new PRD).

Files modified:
- `docs/wiki/index.md`: add a row for the new page.
- `docs/wiki/log.md`: append an entry per conventions.

### Acceptance criteria

- [ ] Real diagnostic run executed at `prefill=2000, acting=2000` for both encoders. JSON lives under `output/profiling/` and is referenced from the wiki page.
- [ ] `docs/wiki/methods/l4-profiling.md` exists with all five sections (hardware, run config, results table, KV-cache audit, recommendation for #72).
- [ ] `docs/wiki/index.md` lists the new page.
- [ ] `docs/wiki/log.md` has a dated entry summarizing the finding.
- [ ] Recommendation in the wiki page is unambiguous (go / no-go / alternative path) and grounded in the table numbers.
