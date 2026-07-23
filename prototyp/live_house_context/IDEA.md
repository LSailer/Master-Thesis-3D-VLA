# Live house context — make `ResetMode.PERSIST_SCENE` actually work end-to-end

## Goal

The live per-scene house point buffer (`HouseContextPoseBuffer`) must
accumulate **one geometrically-coherent map per house** across prefill + train +
multiple episodes, so the `vggt_house_points_pose` encoder (production MLP
branch, and the prototype GNN branch that subclasses it) can be trained on a
faithful 3D scene memory instead of a stack of misaligned ghost copies.

`ResetMode.PERSIST_SCENE` was wired (commits this session) but the first smoke
(job 5738008) showed the buffer still grows to 3.77 M points in 1200 steps —
not the saturation PERSIST should produce. This folder is the protocol,
diagnosis, fix, and verification for making the full-pipeline smoke genuinely
succeed (not just the `metrics.csv ≥ 5 rows` canonical assertion).

## Hypothesis

The wiring is correct in unit tests but **does not fire during prefill**,
because the trainer discards every `env.reset()` frame in the prefill loop, so
`is_first` never reaches the extractor there, `reset_for_scene` never runs, and
the first train episode does a fresh `reset()` (re-anchor) — orphaning the
prefill frame and producing a second, misaligned copy of the house.

## Deliverables

1. `PROTOCOL.md` — assumptions, decisions (with web-sourced best-practice
   arguments), concepts, and learning findings.
2. `PROBLEMS.md` — open problems + decision log.
3. `HANDOFF.md` — session-to-session state.
4. `check_persist_alignment.py` — the cross-episode IoU diagnostic adapted to
   verify PERSIST (the growth curve alone cannot; see PROTOCOL §3).
5. Code fix in `src/r2dreamer/trainer.py`, `src/r2dreamer/adapters/obs_adapter.py`,
   `src/r2dreamer/adapters/{vggt_adapter,hybrid_adapter}.py`, with inline
   comments documenting the decision.