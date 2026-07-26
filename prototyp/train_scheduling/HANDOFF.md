# HANDOFF — train_scheduling

## State (2026-07-23, session 1)

Scaffold complete, **no GPU run submitted yet**.

- `locked_buffer.py` — `LockedReplayBuffer`: real `ReplayBuffer` behind one
  `threading.Lock` (add/sample/sample_transitions/size); attribute reads
  delegate. Swap into `ExperienceCollector.buffer` for threaded mode.
- `scheduling_loops.py` — `run_interleaved` (production credit scheme,
  loops.py:437-465 faithful), `run_episode` (credit deferred to
  `result.done` + final drain), `run_threaded` (actor/learner threads, stop
  event, learner paces against the actor's credit-eligible step counter).
  All timed with `jax.block_until_ready(agent.params)` before the stop clock.
- `run_scheduling_experiment.py` — **removed in the adapter-routing refactor**
  (it imported `src.r2dreamer.launch.registries` and the `launch.train`
  helpers, none of which survived; the smoke below was never submitted, so
  nothing measured is lost). What it did: one mode per process. Composed the
  real stack by importing the production helpers from
  `src.r2dreamer.launch.train` (`_make_encoder_bundle`, `_make_env_instances`,
  `_make_agent_config`, `_make_collectors`) + run ids from
  `scripts/r2dreamer/_run_configs.py`; forces `val_every=0` (no val env).
  Writes `results_<mode>.json` + accumulating `MANIFEST.json`
  (status ok/failed) into the shared `--out_dir`; hard-exits on success
  (habitat teardown poisons exit codes). `--summarize DIR` combines modes and
  prints the overlap gain.
- `run_scheduling_experiment.sbatch` — removed with the script it launched.
  It was gpu_h100, gpu:1, excludes uc3n089, `.venv/bin/python` (not uv), all
  three modes + summary. Smoke shape: steps 600, prefill 200, B=4, T=16,
  ratio 16, run_id `habitat-l1-cnn`.
- Tests: `tests/prototyp/train_scheduling/` — locked-buffer thread hammer +
  loop-semantics tests with fakes (CPU, no habitat). Run:
  `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/prototyp/train_scheduling/ -x -q`

## Next step

Blocked as written: steps 1-2 needed the runner and sbatch wrapper removed in
the adapter-routing refactor. Resuming this investigation means rewriting the
runner against the adapter contract in `src/adapters/` and the composition root
in `src/main.py`; `locked_buffer.py` and `scheduling_loops.py` are unaffected
and still carry the actual idea under test.

1. ~~Submit the smoke: `sbatch prototyp/train_scheduling/run_scheduling_experiment.sbatch`~~
2. Judge by `outputs/prototype/train_scheduling/run-<jobid>/MANIFEST.json`
   (status + `comparison.overlap_gain_vs_interleaved`).
3. If overlap gain > ~1.3x: rerun with prod-ish shape (B=16 T=64 ratio 512,
   more steps) and consider what a graduated threaded `train_loop` needs
   (params handoff, adapter lock — see PROBLEMS.md #2/#3).
4. If ~1.0x: profile where the GIL is held (py-spy on the job) before
   declaring the idea dead.

## Watch out

- Only `habitat-l1-cnn` is safe for `--mode threaded` (stateless adapter
  paths). House/live-VGGT adapters share mutable state between the env path
  and the sample path — PROBLEMS.md #2.
- `--mode` of this runner shadows the train parser's `--mode` (env split);
  the env split is pinned to `train`.
- Modes run in separate processes: each pays JAX compile again; compare wall
  times of the *timed section* only (that's what LoopStats measures — prefill
  and compile of `act` happen before... NOTE: first `train_step` compile IS
  inside the timed section for every mode equally; interpret absolute
  numbers accordingly or subtract via a longer run).
