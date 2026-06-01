# DRY + Modularity Review — `3d-50-hybrid-cnn-vggt` branch

**Date:** 2026-06-01
**Scope:** the ~2,200-line diff of `lucasailerls/3d-50-hybrid-cnn-vggt` vs `main`
(merge-base `a7c917e`) — the hybrid CNN(RGB)+gated-MLP(VGGT) encoder and the WP-grid
ablations.
**Method:** 10 read-only reviewer agents (one per module slice) fanned out over the
diff with a DRY + modularity rubric, each returning structured findings; a synthesizer
deduped cross-module findings and prioritized them. 33 raw findings → 22 consolidated.
**Applied in:** worktree `worktrees/dry-modularity`, branch
`lucasailerls/3d-50-dry-modularity` (off the branch HEAD).

---

## Verdict

The branch is **structurally sound** — clean `Encoder`/`EncoderSpec` hierarchy, good
SLURM `extends:` chains, a properly parametrized WP-pooling refactor (zero findings in
`feature_extractor.py`). The DRY/modularity debt clusters around **one dominant theme:
encoder metadata has no single source of truth.** Encoder type strings and the feature
dimensions (VGGT `4116`, hybrid `16404`, RGB resolution `64`) are re-typed across the
encoders, adapters, config, agent dispatch, parser/`main` choices, the registry, and the
run shims — so adding an encoder or changing a grid touches 3–5+ files with silent
train-time drift on a typo.

Severity calibration is deliberately strict: DRY duplication is only **must-fix** when the
copies can *drift apart and change behavior*. Two such cases exist (parser `--steps`
default; the SLURM copy-paste defaults). The rest is ordinary should/nice consolidation.

---

## Cross-cutting themes

1. **Encoder metadata has no single source of truth** *(dominant)* — type strings + feature
   dims echoed across `encoders/__init__.py`, `registries.py`, `agent.py` dispatch,
   `parser.py` choices, `main.py` (two identical `choices=` lists), `config.py` docstring,
   and the run shims.
2. **Launch + run-config boilerplate** — curriculum resolution duplicated in `train()`/
   `evaluate()`; agent-config assembly diverges between them; 13 run shims duplicate the
   `sys.path` bootstrap + a hardcoded `train()` call.
3. **Config-as-source-of-truth eroded by drifting defaults/docs** — parser `--steps`
   default (2.4M) overrides `config.total_steps` (1M); stale `config.py` enum comment;
   stale `tests/.../AGENTS.md` "No conftest.py" claim (one exists).
4. **Test setup duplication** — 10 inline `FakeExtractor` stubs, 5+ minimal-config builders,
   ad-hoc RNG init that ignores the fixed-key convention.
5. **Intra-module shape/forward-pass + CNN duplication** *(jit paths — higher apply-risk)* —
   `ConvEncoder`/`WPConvEncoder` share an identical conv stack; hybrid RGB-slice extraction
   duplicated in `loss.py` + `agent.reconstruct()`; `reconstruct()` reimplements
   `_world_model_forward()`.

---

## Applied in this worktree (low-risk, test-verified)

| # | Change | Theme | Files |
|---|--------|-------|-------|
| 1 | **Run-shim registry** — new `scripts/r2dreamer/_run_configs.py` holds `RUN_CONFIGS` (one table) + `launch_run(name)` that validates the encoder against `encoder_registry` at launch; the 3 new shims collapse from ~20→7 lines. Files kept so SLURM `script:` paths stay valid. | 2 | `_run_configs.py` (new), `run_jax_habitat_hybrid.py`, `…_vggt_wp_cp_64.py`, `…_vggt_wp_dense.py`, `AGENTS.md` |
| 2 | **Shared `resolve_curriculum_path()`** — extracted the verbatim curriculum block out of `train()` and `evaluate()` into `launch/_helpers.py`. | 2 | `launch/_helpers.py` (new), `train.py`, `evaluate.py` |
| 3 | **`LATENT_PRESETS` table** — replaced the `if/elif` + manual field-loop preset logic with a `config.LATENT_PRESETS` lookup. | 5 | `config.py`, `train.py` |
| 4 | **`HYBRID_FEATURE_DIM` derived** — `3*64*64 + VGGT_FEATURE_DIM` instead of the magic `16404`; VGGT dim now has one owner in the adapter layer. | 1 | `adapters/hybrid_adapter.py` |
| 5 | **Dimension-consistency guard test** — pins `VGGT_FEATURE_DIM == HYBRID_VGGT_DIM == config.vggt_feature_dim` and `HYBRID_FEATURE_DIM == RGB + VGGT`. Enforces the single-source-of-truth invariant **without** cross-layer import coupling. | 1 | `tests/r2dreamer/test_dims_consistency.py` (new) |
| 6 | **Package exports** — added `HybridEncoder`, `ConvDecoder` to `world_model/__init__.py` `__all__`. | 1 | `world_model/__init__.py` |
| 7 | **Doc fixes** — `tests/.../AGENTS.md` no longer claims "No conftest.py"; `config.py` encoder comment points to the registry instead of a stale enum list. | 3 | `tests/r2dreamer/AGENTS.md`, `config.py` |

Net **−21 lines** + 3 new focused modules. CPU suite green (see below).

### Two reviewer suggestions deliberately *not* followed as written

- **Import `VGGT_FEATURE_DIM` into `encoders.py`/`config.py`.** Rejected: it would drag the
  heavy VGGT extractor stack into those lightweight, widely-imported modules — a coupling
  regression. Used a **guard test** to enforce equality instead (finding #5).
- **`choices = list(encoder_registry)` in `main.py`.** Rejected for now: `registries.py`
  eagerly imports the Flax/VGGT encoder classes, so this would make every `src.main` CLI
  invocation (and every shim `from src.main import train`) pull in the heavy stack at import.
  Proper fix is a *names-only* `ENCODER_TYPES` tuple in a dependency-free module (deferred).

---

## Deferred (documented — apply with care / behavior-affecting / jit paths)

### Must-fix (behavior divergence — left for an explicit decision, not silently changed)

- **Parser `--steps` default (2.4M) ≠ `config.total_steps` (1M).** `parser.py:9`
  `default=2_400_000`. When `--steps` is omitted the parser default wins over the config,
  violating config-first SSOT. *In practice the SLURM configs always pass `steps`, so this
  only bites a bare launcher CLI run.* Fix: set the parser default to `None`, resolve
  `total_steps = args.steps if args.steps is not None else cfg.total_steps` in `train.py`.
  Not applied here because it changes effective run length — wanted explicit sign-off rather
  than a silent change during a refactor pass.

- **SLURM `_base` hoist.** *Attempted and reverted.* The reviewer claimed 8/9 configs
  copy-paste universal defaults; a finer audit showed they are **family-specific**
  (`aggregator_mlp_v1`, `offline_buffer_3d25`, `l4_cnn` legitimately diverge/omit
  `prefill`/`checkpoint_every`/`render_resolution`/`seed`/`wandb_project`). Only
  `log_every: 250` is declared by *every* config, so it was the one safe hoist — but hoisting
  it into `_base.args` changed the **render order** of flags in the generated sbatch (parent
  keys merge first), which broke `test_l1_vggt_dry_run_matches_legacy_sbatch`, a deliberate
  byte-parity-with-legacy contract. Reverted. **Proper fix:** introduce an intermediate
  `_curriculum_base.yaml` for the VGGT/hybrid family (l1–l4_vggt, wp_*, hybrid) that the
  non-curriculum configs do *not* extend, and hoist the shared keys there — then either
  accept the new golden output or sort flags deterministically.

### Should-fix

- **Encoder type strings — single source of truth.** Auto-build `encoder_registry` from
  `Encoder` subclasses; expose a dependency-free `ENCODER_TYPES` + capability helpers
  (`is_vggt_encoder()`, `is_rgb_encoder()`); feed `parser.py`/`main.py` choices and shim
  validation from it; replace `agent.py`'s `_make_encoder` dict + `act()` VGGT tuple. Medium
  risk (touches jit'd `act()`/`_loss_fn`) — stage the non-jit parts first.
- **`build_agent_config_from_args()`** shared by `train()`/`evaluate()` — today eval recovers
  arch fields from `MANIFEST.json` via a separate `_ARCH_FIELDS` list, so a new arch field can
  be missed at eval (checkpoint-load failure).
- **Test dedup** — hoist 2–3 reusable `FakeExtractor` fakes + a `minimal_r2dreamer_cfg`
  factory + an `agent_rng` fixture into `conftest.py` / `tests/r2dreamer/_helpers.py`
  (the AGENTS.md doc fix #7 unblocks this). Left untouched to keep the suite green; high-touch.

### Nice-to-have / jit-path (verify under jit + GPU before applying)

- `_conv_stack(x, depth, kernel, mults)` shared by `ConvEncoder`/`WPConvEncoder`.
- `extract_rgb_from_hybrid_obs(obs, rgb_dim)` shared by `loss.py` + `agent.reconstruct()`.
- `_encode_and_observe()` shared by `reconstruct()` + `_world_model_forward()`.
- `HybridEncoder` `rgb_shape` field (or `isqrt(rgb_dim//3)`) instead of the hardcoded
  `(B, 3, 64, 64)` reshape at `encoders.py:210`.
- `_norm_act(x, name)` helper for the recurring `RMSNorm→SiLU`; `vggt_dim` shape assertion
  in `HybridEncoder._branches`.

---

## Verification

```
uv run pytest tests/r2dreamer/ tests/slurm/ -m "not gpu" \
  --ignore=tests/r2dreamer/test_cross_framework.py
```

`test_cross_framework.py` is excluded: it unconditionally imports the external PyTorch
reference (`from rssm import RSSM`) which needs `external/r2dreamer` on `PYTHONPATH`; it is
pre-existing and untouched by this branch. GPU-marked tests are auto-skipped on CPU.
