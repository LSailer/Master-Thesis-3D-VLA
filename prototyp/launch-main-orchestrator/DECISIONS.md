# Decision record (grill-me interview, 2026-07-28)

Numbered as asked in the interview. All settled unless marked OPEN.

## Scope and loop semantics

| # | Decision |
|---|---|
| 1 | Structure AND semantics change in one pass (no bit-identity goal) |
| 2 | Loop unit = env step; gradient steps fire when the train_ratio gate fires |
| 3 | `inference` = exactly one env step (act -> env.step -> buffer.add), returns new carry |
| 4 | `evaluate` disappears as loop owner: same main loop without train(), params from `--checkpoint` |
| 5 | Explicit function arguments, no RunContext object |
| 6 | Validation deleted entirely; overfit is the only special case |
| 16 | `train_credit` is a local variable in main's loop; `train()` does exactly one gradient step |
| 20 | Prefill stays a distinct phase (own leading segment, uniform-random actions) |
| 21/28 | Overfit = own branch BEFORE the loop (~30 lines, one frozen batch, min-loss-drop check). NOT the curriculum variant - that would measure small-data, not one-batch |
| 29 | Eval budget: step loop breaks when `episodes_done == args.episodes`; per-episode cap `--max_episode_steps` default 500 (replaces the hardcoded 500 in evaluate.py:283) |

## Parser

| # | Decision |
|---|---|
| 7/23 | ONE `--mode` flag for workflow AND episode split, values `train` / `eval` |
| 8 | One parser, one `--seed` (default 42), one `--mode`, one `--wandb_project`; no precedence rules |
| 10 | Parser lives in `src/launch/parser.py` (new top-level package); `src/r2dreamer/launch/` dissolves |
| 11 | Booleans: launcher renders `key: true` as bare `--flag` (option C, one line in `_format_arg` in scripts/slurm/launch.py:182 + training_command test helper) |
| 33 | All parser booleans become `store_true`; `_str2bool` is deleted |
| 12/24 | No curriculum validator. `--curriculum` gets `default="L1"`; crafter simply ignores it. `_effective_curriculum` and its precedence logic die |
| 25 | `--wandb_project` default `"3d-vla-objectnav"` for both modes; `--wandb_project ""` disables |
| 15 | All val flags AND `val_every: 0` YAML lines deleted |

## Production path

| # | Decision |
|---|---|
| 9 | `scripts/r2dreamer/run.py` and `_run_configs.py` are deleted; SLURM calls `python -m src.main --mode ... --env ... --adapter ...` |
| 26 | `run_id:` disappears from YAML schema; each of the 46 configs writes env/adapter/curriculum in its `args:` block, `extends`/`_base.yaml` carries repetition |
| 34 | YAML entrypoint: `script: -m src.main` (renderer unchanged; accepted schema wart) |
| 13 | Same PR: delete dead scripts (`scripts/train_baseline.sh`, `scripts/smoke_test_pipeline.sh` calls to nonexistent `src.dreamerv3`, stale `scripts/slurm/train.sbatch`), fix wrong docs (`scripts/r2dreamer/AGENTS.md:22` dash-flags/`--no-wandb` claim), stale profiling configs |

## Rollout / act path

| # | Decision |
|---|---|
| 17 | ONE functional act path; carry is a local in main, threaded through inference |
| 38 | `act` becomes public and directly jitted: `act(self, params, obs, is_first, state, rng_key, training) -> (jax.Array, ActState)` with `static_argnums=(0,)`. `params` explicit (kills stale-params hazard). `_batch_live_obs` moves inside the jit. Old `act`, `act_with_state`, `act_with_state_pure`, `snapshot_act_state`, `restore_act_state` deleted |
| 39 | `train_step` is migrated to the SAME convention (public, directly jitted, static self) - one convention for the whole file, accepted bigger blast radius. `materialize` escape hatch semantics preserved |
| 40 | `int(action)` conversion happens in `inference()` in main, directly before env.step |
| 18 | Both modes roll out through `ExperienceCollector` (eval: `buffer=None`, `auto_reset=False`) |
| 30 | `RandomAgent` implements the same act signature (unused params/state args); selected via `--random`; all four isinstance branches die |
| 31 | Eval metrics come from `HabitatEpisodeMetrics` (same source as train), NOT from reading the last obs |

## Lifecycle / teardown

| # | Decision |
|---|---|
| 22/32 | Context manager around the LOOP ONLY: `with run_session(logger, collector, env, hard_exit=...)`. Owns logger.finish(status), collector.close(), env.close(), final checkpoint, KeyboardInterrupt -> "interrupted", and `os._exit(0)` under `R2DREAMER_HARD_EXIT_ON_FINISH`. Composition stays visible in main |
| 19 | Eval artifacts stay (eval_results.json, topdown PNGs, W&B videos), written by main at episode end |
| 36 | `main` returns None; tests call the now-public composition function instead |

## Tests / verification

| # | Decision |
|---|---|
| 14 | Hand-copied composition in tests dies; `_compose_run` (or successor) becomes public and tests call it |
| 35 | New test: render every YAML config and push its flags through `build_parser().parse_args()` - replaces both `test_shim_invocation.py` and the weaker drift guard `tests/slurm/test_launch.py:362` |
| 27/37 | Verification = A/B on SLURM: old main vs new main, same seed, ~5k steps, l1_cnn AND a VGGT arm (hybrid), compare loss curve + metrics.csv shape; plus existing smoke config. Check sacct for exit 134 after every round (bad node uc3n089 / L4 GL contention) |

## Final round (settled 2026-07-28)

| # | Decision |
|---|---|
| 41 | Frozen prototype call sites of the old act API (`prototyp/live_vggt/live_vggt.py:160`, `prototyp/train_scheduling/scheduling_loops.py:148,226,314`) are left untouched - frozen artifacts, not production code |
| 42 | `--max_episode_steps`: one flag, default 500, passed to `make_env`, applies to BOTH modes; the duplicate hardcoded 500 in evaluate.py dies with the file |
| 43 | `R2DTrainState` stays the carrier: `train_step(self, train_state, batch, rng_key, materialize) -> (train_state, metrics)`, public, directly jitted, `static_argnums=(0,)`, `static_argnames=("materialize",)` (two cache entries). main holds `train_state` as a local next to the act carry |

Interview complete - no open questions. Implementation starts only on explicit go.
