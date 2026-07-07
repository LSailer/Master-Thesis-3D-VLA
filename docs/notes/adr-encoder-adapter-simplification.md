# ADR: Simplify the r2dreamer encoder/adapter architecture

**Status:** Proposed
**Date:** 2026-07-07
**Deciders:** Luca
**Companion docs:** [encoder-registry-refactor-draft.md](encoder-registry-refactor-draft.md) (launcher-layer design, evaluated here), [live-house-context-explainer.md](live-house-context-explainer.md)

## Context

Selecting one encoder (e.g. `vggt_house_global_embedding`) currently threads
through **four parallel name mappings** plus a class hierarchy:

| # | Layer | Where | What it maps |
|---|-------|-------|--------------|
| 1 | Launcher class tree | `encoders/base.py` + `house_points_pose.py`, `gnn_house.py`, `house_global_embedding.py`, `__init__.py` (~700 lines) | name → `Encoder` subclass with 5–7 property overrides each |
| 2 | Launch registry | `launch/registries.py:35-52` | name → launcher class (hand-maintained, 16 entries, 16-name import block) |
| 3 | VGGT variant specs | `observation_preparation/vggt.py:180` (`VGGT_DREAMER_SPECS`) | `variant_key` → readout/storage/dreamer axes |
| 4 | Agent module table | `agent.py:144-160` | `encoder_type` string → Flax module class (redundant: `EncoderSpec.module_cls` already carries it) |

Adding one encoder today touches 4–5 files; the knowledge for one encoder is
split across a subclass, a dict entry, possibly a spec entry, and the agent
table. Mechanisms that exist only to serve this split:

- `variant_encoder_class()` — builds subclasses via `type()` (`base.py:241`).
- `_VariantDescriptor` + `_LazyVGGTVariants` — descriptor/lazy-mapping
  machinery so class attributes can resolve a second registry (`base.py:43-76`).
- `VGGTEncoder`'s seven pass-through `@property`s that just forward to
  `self.variant.*` (`base.py:167-200`).
- `VGGTHouseContextEncoder` overriding four properties/methods to express one
  live-vs-static switch (`__init__.py:67-148`).

The duplication is already producing drift: `house_points_pose.py:33-35`
still documents "`on_episode_reset` is deliberately left unset", but the
adapter has set it since the prefill-orphaning fix
(`hybrid_adapter.py:608-627`). Same fact, two owners, one stale.

**Adapter layer** (out of scope for the draft, in scope here):

- `adapters/hybrid_adapter.py` is an 830-line grab-bag holding **seven**
  adapters from four unrelated families; only one of them is the "hybrid"
  encoder the filename promises. Every new experiment has appended here.
- Two contract styles coexist: `VGGTObsAdapter`/`HybridObsAdapter` build a
  `contract` object (`build_vggt_contract`/`build_hybrid_contract`) that
  `Encoder.spec()` prefers; the five newer house adapters hand-assemble
  `buffer_dtype/buffer_shape/normalize_on_sample/agent_obs_shape` dicts and
  set no contract, forcing `spec()`'s fallback path and pushing
  `encoder_type`/`module_cls`/`agent_overrides` back onto launcher classes.
  This dual resolution in `spec()` (`base.py:108-131`) is why layer 1 can't
  shrink on its own.
- The object-vs-legacy-Mapping VGGT output shim is written three times
  (`_vggt_output_field`, `_global_tokens_from_output`, and inline in
  `VGGTHouseGlobalTokenObsAdapter`).
- The `on_episode_reset=lambda scene_id="scene": extractor.reset_for_scene(scene_id)`
  lambda is copy-pasted in four adapters.

Constraints: this is an active thesis codebase with production runs launched
from `scripts/slurm/launch.sh` + YAML; manifests (`manifest.py`,
`contract_snapshot`) are used to judge runs, so any refactor must keep
manifest output byte-identical and each migration step independently green.

## Decision (proposed)

Adopt the declarative-registry refactor from the draft for the launcher
layer, and add a scoped adapter-layer cleanup. Three tiers, ordered by
value/risk; each lands separately.

## Options Considered

### Option A: Status quo

| Dimension | Assessment |
|-----------|------------|
| Complexity | High and rising — every new encoder adds a subclass + 3 dict entries |
| Cost | Zero now, compounding later |
| Risk | Zero immediate; drift bugs accumulate (stale-docstring class already exists) |
| Maintainability | Poor — 4 sources of truth per encoder |

**Pros:** No migration risk during active experiments.
**Cons:** The class tree grows by one subclass per experiment; the
`type()`/descriptor machinery raises the bar for every reader; drift between
the four mappings is caught only at runtime.

### Option B: Declarative registry (the existing draft), launcher layer only

One `EncoderDef` record per encoder + `EncoderRuntime`; deletes the class
tree, the `registries.py` dict, and (step 4) the agent table.
~650 lines of plumbing → ~250 lines of records.

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low — data + plain factory functions, zero subclasses |
| Cost | ~1–2 days incl. smoke runs; draft's 5-step order keeps each step green |
| Risk | Low-medium — `spec()` moves verbatim; registration side-effect footgun is called out in the draft |
| Maintainability | Good — one record per encoder, duplicate-name check for free |

**Pros:** Kills mappings 1, 2, 4 (and 3 in its phase 2); variants become
`dataclasses.replace` one-liners; `--help`/error listings come from one dict.
**Cons:** Leaves `hybrid_adapter.py` untouched; the contract asymmetry that
forces `spec()`'s dual path survives (draft keeps it "verbatim" — correct for
the migration, but it is the next thing to remove).

### Option C: Option B + adapter-layer cleanup

Everything in B, plus:

1. **C1 — file split (zero behavior change).** Break `hybrid_adapter.py`
   into family files: keep `HybridObsAdapter` where it is; move
   `_RGBLiveTokenObsAdapter` + full/global-token + global-embedding adapters
   to `adapters/token_adapters.py`; `VGGTHouseContextObsAdapter` to
   `adapters/house_context_adapter.py`; `VGGTHousePointsPoseObsAdapter` +
   hybrid subclass + the point/pose helpers to
   `adapters/house_points_adapter.py`. Pure move; import sites are the lazy
   `import_module` closures, so only the factory strings change.
2. **C2 — one output shim.** Normalize the extractor output once (either the
   extractor always returns the structured object, or a single
   `as_vggt_output(out)` helper) and delete the three Mapping-fallback
   copies.
3. **C3 — one reset-callback helper.** `scene_reset_callback(extractor)`
   replacing the four copy-pasted lambdas; `None` stays expressible for the
   adapters that opt out.
4. **C4 — contract unification (the real prize, most effort).** Give the
   five hand-assembled adapters a real contract (or a slim
   `ObservationContract.from_fields(...)` builder), then delete the fallback
   branch of `spec()`. After C4, `encoder_type`/`module_cls`/
   `agent_overrides` have exactly one owner (the `EncoderDef` record) and
   the manifest snapshot has exactly one producer.

| Dimension | Assessment |
|-----------|------------|
| Complexity | Lowest end-state of the three |
| Cost | B + ~1 day (C1–C3 are mechanical) + C4 as its own reviewed PR |
| Risk | C1–C3 near-zero; C4 medium (touches manifest production — needs snapshot-diff test) |
| Maintainability | Best — one record + one contract per encoder, files named for what they contain |

## Trade-off Analysis

- **B vs A:** The draft's numbers hold up against the code — the four-family
  house group really is 4 near-identical subclasses that become one entry +
  three `replace()` lines, and `agent.py`'s table is provably redundant
  (`module_cls` already travels in `EncoderSpec`; the `issubclass`-based
  kwargs dispatch from 953f05c is untouched). The one genuinely behavioral
  launcher (`VGGTHouseContextEncoder`) fits the factory pattern without
  special cases. B is worth doing on its own.
- **C1 timing:** the file split does not depend on B at all and de-risks it —
  smaller files make the B review easier. Do C1 first or alongside B step 1.
- **C4 timing:** only after B has landed and a couple of production runs have
  confirmed byte-identical manifests. It is the step that finally lets
  `spec()` become three lines, but it touches the run-judging artifact
  (MANIFEST.json), so gate it on a golden-file test comparing
  `contract_snapshot` before/after for every registered encoder.
- **What not to do:** do not try to merge `ObsAdapter` subclasses themselves
  into declarative records. Unlike the launchers, the adapters hold real
  per-step behavior (buffer management, PLY dumps, JIT-stable snapshots) —
  inheritance there (`VGGTHybridHousePointsPoseObsAdapter` extending the
  points-pose adapter via `_observation_contract_kwargs`) is earning its keep.

## Consequences

Easier: adding an encoder = one `register(EncoderDef(...))` (or one
`replace()` line for a variant); listing encoders for `--help`; finding an
adapter by filename; keeping docs true (one owner per fact — the
`on_episode_reset` drift class disappears).

Harder / to watch: registration-by-import means a bare `import registry`
sees an empty dict (draft flags this — accept or add `load_all()`); git
history for the moved adapters fragments (use `git log --follow`).

Revisit later: draft phase 2 (folding `VGGT_DREAMER_SPECS` readout/storage
axes into `EncoderDef`) — after C4, since both touch the same contract
plumbing.

## Action Items

1. [ ] C1: split `hybrid_adapter.py` into family files (no behavior change).
2. [ ] B steps 1–3 per the draft's migration order (registry + defs, migrate
       variants, delete class tree), smoke via `scripts/slurm/launch.sh`
       after each step, judging by MANIFEST.json.
3. [ ] B step 4: thread `module_cls` from `EncoderSpec` into `agent.py`,
       delete the string table.
4. [ ] C2 + C3 (shim + reset-callback dedup), can ride along with any step.
5. [ ] Golden-file test: `contract_snapshot` equality for all encoders.
6. [ ] C4: contract unification; delete `spec()` fallback branch.
7. [ ] Fix the stale `on_episode_reset` docstring immediately if B is
       deferred (it dies with the class otherwise).
