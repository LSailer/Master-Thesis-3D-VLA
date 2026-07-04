# House-points-pose train-loop profiling (job 5762154, 2026-07-04)

## Question

Compare jobs 5736907 (GNN) / 5736908 (MLP) ran at ~213–230 ms/env-step, while
the remembered baseline was ~158 ms/step. Regression, or measurement artifact?
Where does the time actually go?

## Method

`scripts/profiling/profile_house_points_pose.py` (launched via
`scripts/slurm/launch.sh profile_house_points_pose`) runs the **real**
production stack — built through `launch_run("habitat-l1-gnn-house-points-pose")`
exactly like `run.py` — and instruments the constructed `Trainer` instance by
wrapping its hot-path methods with device-synced wall-clock timers
(`Trainer.run` monkeypatch). Production shape: batch 16, seq 64,
train_ratio 512, render 518, H100 (uc3n078). 1300 prefill + 400 acting steps;
stats over the trailing 50% of samples.

## Result: 218.8 ms/env-step, reconciles with the observed 213–230

| phase (amortized/env-step) | ms | share |
|---|---|---|
| `vggt_extract` (in obs_transform) | 132.2 | 61.8% |
| `replay_sample` × 0.5 | 59.4 | 27.1% |
| `train_step` × 0.5 | 19.7 | 9.0% |
| `agent_act` (WM inference + policy) | 2.3 | 1.1% |
| `env_step` (Habitat sim + render) | 2.0 | 0.9% |
| `house_add` + `house_snapshot` + glue | 3.1 | 1.4% |
| **total** | **218.8** | |

Per-call: `train_step` 39.5 ms, `replay_sample` 118.7 ms (each runs 0.5×/env
step at ratio 512 with 1024-frame batches). First-call JIT/compile costs:
`train_step` 139.5 s, `obs_transform` 26.6 s, `agent_act` 2.1 s.

## Answers

1. **No regression, and not the GNN.** The GNN branch is invisible in the
   profile (`agent_act` 2.3 ms total). The old 158 ms figure came from smoke
   runs at light train shape (batch 4 × seq 16, ratio 16); the compare runs use
   the production shape. Light-shape arithmetic (132 VGGT + ~7 loop + tiny
   amortized train) ≈ 145 ms — matching the observed 141–155 ms smokes.
   Production-shape arithmetic ≈ 219 ms — matching the compare runs. The two
   numbers were never comparable.
2. **The loop is VGGT-bound.** 132 ms of every env step is InfiniteVGGT
   extraction — a hard floor regardless of train shape. Reducing it needs
   torch.compile-style work on the extractor, lower render resolution, or
   extracting less often; nothing in the Dreamer side touches it.
3. **`ReplayBuffer.sample` is the #1 cheap win.** 118.7 ms per 16×64 batch —
   3× the actual `train_step` math (39.5 ms). The buffer stores per-transition
   Python objects (`deepcopy` on `add`, line ~231 of
   `src/buffer/replay_buffer.py`), so sampling gathers and packs 1024 objects
   frame-by-frame on the host. Vectorizing storage (preallocated arrays,
   contiguous window gather) would cut up to ~59 ms/env-step (~27%) — 2M-step
   runs would drop from ~121 h to ~89 h. `profile_jax_replay_buffer.py` and
   the modal-replay profiling configs are prior art in this area.

## Caveats

- Phase timers add sync barriers, so overlap between host env stepping and
  device compute is slightly suppressed; the amortized total landing within
  the observed 213–230 ms band says the distortion is small.
- `replay_sample` cost scales with batch·seq frames, not buffer fill level
  (compare runs were flat 213 ms from step 500 to 50k).
- Raw per-phase samples: `output/profiling/house-points-pose/run-5762154/
  profile_house_points_pose_20260704_081815.json`.
