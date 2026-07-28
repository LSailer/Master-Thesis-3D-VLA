# Implementation plan

Basis: DECISIONS.md (43 settled decisions). Repo must stay green after every
landed PR, so the work is staged as three PRs plus a verification phase.
Decision 13 (dead scripts/docs in the same PR as the refactor) is honored
inside PR3, which is the refactor PR.

## PR structure

| PR | Content | Repo stays green because |
|---|---|---|
| PR1 | Agent API refactor (act + train_step directly jitted) + adapt existing callers | old loops/evaluate are adapted in the same PR |
| PR2 | Launcher renders YAML booleans as bare flags | current `--full_bf16` (`nargs="?", const=True`) accepts the bare form already |
| PR3 | New `src/launch/` + new `src/main.py` orchestrator + YAML migration + deletions + docs + test replacements | everything switches atomically |

PR1 and PR2 are independent of each other. PR3 depends on both.

## Tasks

### T1 - Agent API refactor (shipmate, own worktree, PR1)

Files: `src/r2dreamer/agent.py`, `src/r2dreamer/launch/loops.py`,
`src/r2dreamer/launch/evaluate.py`, `src/baselines/random_agent.py` (NOT yet),
`tests/r2dreamer/test_agent.py`, `tests/r2dreamer/test_habitat_act_state_parity.py`,
`tests/adapters/test_routed_pipeline.py`, `tests/r2dreamer/test_training_loops.py`.

- `act(self, params, obs, is_first, state, rng_key, training) -> (jax.Array, ActState)`,
  `@partial(jax.jit, static_argnums=(0,))`; `_batch_live_obs` moves inside the
  jit; delete `act`, `act_with_state`, `act_with_state_pure`,
  `snapshot_act_state`, `restore_act_state`, the mutable `_act_state`.
- `train_step(self, train_state, batch, rng_key, materialize) -> (train_state, metrics)`,
  directly jitted, `static_argnums=(0,)`, `static_argnames=("materialize",)`;
  delete `_train_step_pure` / `_jitted_train_step` plumbing; agent no longer
  mutates `self._train_state` from inside train_step (caller reassigns).
- Prominent docstring warning: static self => never read mutable attributes in
  the jitted bodies; params/train_state must be explicit arguments.
- Adapt current callers minimally so PR1 is green: `train_loop`/`val_loop`
  thread a local carry (snapshot/restore usage disappears naturally),
  `evaluate.py` call site updated, `int()` at the call sites.
- Acceptance: full suite green (`uv run --no-sync pytest`), especially
  `test_habitat_act_state_parity.py`; no `_act_state` references left in src/.

### T2 - Launcher boolean rendering (shipmate, own worktree, PR2, parallel to T1)

Files: `scripts/slurm/launch.py`, `tests/slurm/test_launch.py`,
`scripts/slurm/README.md`.

- `_format_arg`: YAML bool `True` -> bare `--flag` line, `False` -> line
  omitted; string values `"true"`/`"false"` -> hard error (the string trap
  from PROBLEMS.md).
- `training_command` test helper learns bare flags; new render cases incl.
  `hybrid_hpp_bf16_prodshape_probe` -> bare `--full_bf16`.
- README rendering-contract paragraph updated.
- Acceptance: `tests/slurm/` green; rendered bf16 config parses with the
  current train parser.

### T3 - The orchestrator (sequential stages in THIS worktree, PR3)

Stages are sequential (each builds on the previous); no two agents ever touch
the same file concurrently.

**T3a - `src/launch/parser.py`** (shipmate)
- One parser per the interview sketch, minus all val flags, minus
  `_str2bool`; plus `--mode {train,eval}` (default train), `--episodes`,
  `--checkpoint`, `--random`, `--max_episode_steps` (default 500),
  `--seed` default 42, `--curriculum` default "L1", `--wandb_project`
  default "3d-vla-objectnav", all booleans `store_true`.
- Unit tests: defaults, boolean forms, `--buffer_capacity` alias kept.

**T3b - new `src/main.py` + `src/launch/`** (shipmate)
- `src/launch/session.py`: `run_session(logger, collector, env, hard_exit)`
  CM - owns logger.finish(status), collector.close(), env.close(), final
  checkpoint, KeyboardInterrupt -> "interrupted", `os._exit(0)` under
  `R2DREAMER_HARD_EXIT_ON_FINISH`.
- `src/main.py` per GOAL.md target shape: explicit composition (public
  composition function replaces `_compose_run`), env-step loop, prefill
  segment, visible `train_credit` gate, functional carry +
  `train_state` locals, `int(action)` in `inference()` before env.step,
  eval break on `episodes_done == args.episodes`, eval artifacts written by
  main at episode end (eval_results.json, topdown PNGs, W&B videos),
  overfit branch before the loop, `main` returns None.
- `RandomAgent` implements the act signature (unused params/state); all four
  isinstance branches die. Eval metrics from `HabitatEpisodeMetrics`.
- Delete: `src/r2dreamer/launch/` (parser.py, loops.py, evaluate.py -
  surviving pieces move to `src/launch/`), `TrainingRun`,
  `_effective_curriculum` + precedence logic.
- Tests: rewrite `test_parser.py`-adjacent config-translation tests against
  the new surface, composition-function tests replace the hand-copies,
  delete `test_effective_curriculum_inputs.py`.

**T3c - production path migration + deletions + docs** (shipmate)
- 46 YAMLs: `script: -m src.main`, drop `run_id:`, add
  `mode`/`env`/`adapter`/`curriculum` to `args:` (extends/_base carries
  repetition), drop `val_every` lines. `LaunchConfig`: remove `run_id`,
  `curriculum_check` stays.
- New render-and-parse test: every YAML renders and its flags pass
  `build_parser().parse_args()` (replaces `test_shim_invocation.py` and the
  old drift guard).
- Delete: `scripts/r2dreamer/run.py`, `_run_configs.py`, `eval_habitat.py`,
  `scripts/train_baseline.sh`, `scripts/smoke_test_pipeline.sh` (dead
  `src.dreamerv3` calls), `scripts/slurm/train.sbatch`, legacy
  `scripts/r2dreamer/slurm/*.sbatch`, stale profiling configs;
  migrate or delete `scripts/profiling/cprofile_run.py` and
  `scripts/smoke_test_r2dreamer.sh` (point at `-m src.main`).
- Docs: `scripts/r2dreamer/AGENTS.md` (dash-flags/`--no-wandb` lie),
  `scripts/slurm/README.md` (schema, variant table), `src/r2dreamer/AGENTS.md`,
  README launcher section, notebook assert message.
- Frozen prototype call sites stay untouched (decision 41).

### T4 - Verification (slurm-runner agents, after PR3 is assembled)

- A/B on SLURM, seed 42, ~5k steps, smoke-shape probe configs:
  - Arm 1: `l1_cnn` old code (main checkout) vs new code (this worktree).
  - Arm 2: `hybrid` (VGGT arm) old vs new - exercises the functional carry
    and prefill on the expensive path.
- Judge by MANIFEST.json, not exit code; grep `sacct` for exit 134 after
  every round (bad node uc3n089 / L4 GL contention); compare W&B loss curves
  and metrics.csv shape. Bit-identity is NOT expected (prefill logging, act
  path, seed semantics changed) - trends and shapes must match.
- Plus: existing smoke config green end-to-end through the new entry point.
- Known residual risk (accepted in interview): eval SR/SPL source changes to
  `HabitatEpisodeMetrics`; no eval-checkpoint comparison planned.

## Who does what

- **Firstmate (this session)**: sequencing, scoping each shipmate, reviewing
  diffs between stages, keeping DECISIONS/HANDOFF current, assembling and
  shipping PRs via no-mistakes, reporting.
- **Shipmates**: T1, T2, T3a, T3b, T3c - each gets goal, file list,
  acceptance test, and the worktree-hygiene block (setup_worktree.sh,
  `uv run --no-sync`, PYTHONPATH, no GPU on login node).
- **slurm-runner**: T4 probe runs.

## Order

```
Wave 1 (parallel): T1 (PR1), T2 (PR2)
Wave 2:            T3a -> T3b -> T3c (sequential, PR3 branch = this worktree)
Wave 3:            T4 verification, then PR3 via no-mistakes
```
