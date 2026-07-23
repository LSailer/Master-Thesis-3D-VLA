# HANDOFF — implement the encoder split end to end and open a PR

Read `IDEA.md` first (binding decisions), and open `design.html` for the full
design with diagrams (card ids like 7a are referenced below). Reference ADR
0006 (`docs/adr/0006-decouple-trainer-from-env-and-buffer.md`) for the style
of migration expected here — this change follows the same discipline.

## Scope

Refactor `src/r2dreamer/` so that:

- `agent.py` no longer builds or owns the encoder. It shrinks to (or is
  replaced by) three modules: `world_model` (encoder-free RSSM + heads),
  `behavior` (actor-critic), and a thin learner exposing
  `policy(params, latent, state, key)` and `train_step(state, batch, key)`.
  Constructor input: `latent_dim: int` (+ `num_actions`) — no encoder types,
  no obs_shape inference, no factory imports.
- A new `src/r2dreamer/encoders/composite.py` provides `CompositeSpec`,
  `CompositeEncoder`, and the `FUSIONS` map (`concat`, `gate`, `concat_mlp`).
  Gate fusion must reproduce the current `WMHybridEncoder` computation
  exactly (same submodule shapes and param structure semantics).
- A new `src/r2dreamer/encoders/recipes.py` defines
  `EncoderRecipe(make_adapter, composite)` and a `RECIPES` registry covering
  at minimum: `cnn`, `hybrid`. Port further encoder types incrementally;
  unported types keep the old path until ported (do not break them).
- `launch/train.py` wires: recipe → adapter → `ExperienceCollector` →
  infer obs spec from the first prepared frame → `enc.init` → TrainState with
  three optimizer states (WM incl. encoder / actor / critic, LaProp from
  `src/shared/optim.py`, today's hyperparameters) → loops.
- `train_step` implements card 7a: encoder applied inside the WM loss fn,
  `jax.value_and_grad` over `(enc_params, wm_params)`, WM optimizer updates
  both subtrees as one pytree; actor/critic on `sg(feat)` with their own
  optimizers; decoder targets from the batch, recon loss inside the WM loss.
- Checkpointing keeps loading old checkpoints (params pytree keeps an
  `"encoder"` subtree; add a migration shim for the optimizer-state layout,
  documented in the PR description).

## Non-goals

- No per-module learning rates (keep hyperparameters identical everywhere).
- No behavioral change of adapters, buffer, collector, or env code.
- No RNG-order changes: preserve the exact order of `jax.random.split` calls
  and of env-step / buffer-add / sample operations (golden-run contract).

## Migration plan (each step lands green on its own)

1. `CompositeEncoder` + `FUSIONS` + unit tests (shape + param-tree structure,
   parity test vs `ConvEncoder` and `WMHybridEncoder` outputs on fixed seed).
2. `RECIPES` for `cnn` and `hybrid` + fail-fast key check + parametrized CPU
   test over all registry entries: fake frame → infer spec → init → apply →
   assert `f32[B,T,E]`.
3. Split `agent.py` internals into world_model / behavior / learner with the
   three-optimizer TrainState, keeping the public API as a shim. Golden run
   here (see Verification).
4. Rewire `launch/train.py` to recipe-based wiring; delete the shim and the
   encoder factory branches that became registry entries; migrate
   `heldout_eval.py` and tests.
5. Port remaining encoder types to recipes one by one (separate commits).

## Verification (hard gates, in order)

1. **Unit tests (CPU):** `JAX_PLATFORMS=cpu uv run pytest -x -q` — full suite.
2. **Golden-run equivalence:** fixed-seed short run (smoke config, ≈500 steps
   + prefill) on `main` vs. the branch for `--encoder cnn` and
   `--encoder hybrid`; sorted `metrics.csv` must be identical modulo `perf/*`
   rows. Same procedure as ADR 0006. Cluster: judge by `MANIFEST.json`
   status, not exit code.
3. **Lint:** run the repo's configured linters (see `pyproject.toml`; use
   pylint if configured, otherwise the configured tool). Record the error
   count on `main` first — the branch count must be ≤ that baseline. Same
   rule for pytest: no new failures relative to `main`.

## PR instructions

- Branch: `refactor/encoder-split-composite`.
- Before opening the PR: run gates 1 and 3 locally; attach the golden-run
  diff evidence (or the two sorted CSV checksums) to the PR description.
- PR description must contain: link to this folder, the decision list from
  `IDEA.md`, the lint/test baseline vs branch numbers, the checkpoint
  migration note, and which encoder types are ported vs still on the old path.
- Do not squash away the per-step commits — the reviewable unit is the
  migration step.
