# Problems, hazards, research notes

Append as they come up.

## Known hazards going in

- **Static self on jitted act/train_step**: with `static_argnums=(0,)`, `self`
  is hashed by identity and never retraced. Any future mutable attribute read
  inside the body silently freezes at trace time. `params`/`train_state` must
  ALWAYS be explicit arguments. Document this prominently in agent.py.
- **`self.cfg` is an unfrozen dataclass** (src/configs/agent_config.py:75) and
  therefore unhashable - it can only be reached through static self, never be
  a static arg itself. Do not "clean up" by passing cfg explicitly.
- **Eval numbers may shift** (decision 31): switching eval metrics from
  last-obs reads to `HabitatEpisodeMetrics` changes the source. If the two
  computed differently, eval SR/SPL changes. The A/B plan (37B) does NOT
  include an eval-checkpoint comparison - if eval numbers matter for a running
  ladder, add one before trusting new eval output.
- **Seed default flips 0 -> 42** for training. YAMLs set `seed: ${SEED}`
  explicitly, so production is unaffected; ad-hoc CLI runs change.
- **Prefill semantics change** (20): episodes during prefill were
  `summarize=False`; decide in implementation whether prefill episodes are
  logged (they will be, unless suppressed - old behavior suppressed them).
- **`eval_habitat.py` relies on the `argv=None` -> `sys.argv[1:]` fallback**
  of today's `evaluate()`; it dies with run.py. The notebook
  `notebooks/r2dreamer/baseline_evaluation.ipynb:34` references it (and with a
  wrong flag `--output`).
- **cprofile_run.py** (scripts/profiling/cprofile_run.py) runs run.py via
  runpy and dies with it - needs migration to `-m src.main` or deletion.
- **Boolean YAML string trap** (11C): `decoder: "true"` (string, not bool)
  would render `--decoder true` and crash at parse. The renderer change must
  only treat real YAML booleans as bare flags; maybe assert on str values
  "true"/"false".
- **`extra="forbid"`** on LaunchConfig: removing `run_id` from the schema
  makes every YAML still carrying `run_id:` a hard validation error - all 46
  must be migrated in the same commit as the schema change.
- **Habitat needs a Python int action** (agent.py:514-517 comment): the
  `int()` now lives in `inference()`; it is the per-step device sync. Do not
  "optimize" it away; there is nothing to overlap with in a single-env loop.
- **A/B runs**: judge by MANIFEST.json, not exit code; grep sacct for exit 134
  (node uc3n089, L4 GL contention) after every round.

## Test casualties (expected, planned replacements)

- `tests/r2dreamer/launch/test_effective_curriculum_inputs.py` - models the
  parse_known_args split; dies with it.
- `tests/r2dreamer/launch/test_shim_invocation.py` - replaced by
  render-and-parse test (35A).
- `tests/slurm/test_launch.py:362` drift guard - replaced by the same.
- `tests/slurm/test_launch.py:109` `training_command` helper - must learn the
  `-m src.main` entrypoint shape and bare boolean flags.
- `tests/r2dreamer/test_training_loops.py`, parser tests, evaluate-manifest
  tests - rewritten against the new surface.
- `tests/adapters/test_routed_pipeline.py` hand-copied composition - calls the
  public composition function instead (14A).

## Carry-over items for PR3 (from PR1 review, 2026-07-28)

- `materialize` as a static argname on the jitted train_step is arguably
  redundant now: the float() host sync moved OUT of the jit into
  `materialize_metrics` at the call sites, and the static flag costs a second
  compilation. Decision 43 mandated it; revisit in PR3 (drop the flag, always
  return device metrics, let main decide when to materialize).
- docs/adr/0006 still shows the pre-refactor train_step shape and the
  snapshot/restore dance - needs a superseded-by note in PR3's doc pass.
- Static-self jit retains every agent instance + executable for the process
  lifetime (weakref-probed; jit's own clear_cache() does not release them).
  Test suite got an autouse jax.clear_caches() in conftest.py as the fix -
  this was a real contributor to the earlier OOM, alongside node contention.
  Long-lived multi-agent processes must reuse ONE agent instance.

## Carry-over items for PR3 (from PR2 review, 2026-07-28)

- Launcher boolean guard only catches quoted "true"/"false"; YAML 1.1
  scalars yes/no/on/off stay strings under ruamel and slip through as
  `--full_bf16 yes`. Add to the string-trap guard in T3c.
- `src/r2dreamer/launch/parser.py` docstring and --help still claim the
  launcher renders `--full_bf16 True` - stale since PR #219; the file dies
  in PR3 anyway, but the new src/launch/parser.py must not inherit the claim.
- A repo-wide `uvx ruff format src` (17 files) was attempted by the old gate
  lint step and reverted in PR2; if wanted, land it separately on a quiet
  tree, never inside a feature PR.

## Dead ends

(none yet)
