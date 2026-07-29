# Handoff

## State (2026-07-29)

Working. `complexity_report.py` runs clean over 97 modules / 498 functions and
writes `outputs/prototype/complexity-analysis/report.md`.

## Findings so far

- The import graph is a **DAG**, zero cycles. Architecturally clean.
- 8 hotspots, 6 of them the vendored JAX VGGT port under `src/vggt/jax/`.
- The two genuinely authored hotspots are
  `src.buffer.replay_buffer` (34 decisions, blast 21) and
  `src.environments.habitat` (48 decisions, `__init__` at CC 18, blast 6).
- `src.main` is the single most complex module (98 decisions, `run_loop` at
  CC 31) but has **blast radius 0** and instability 1.00, so it is a leaf.
  Splitting `run_loop` is the cheapest large complexity win in the repo.
- `src.adapters.contract` is the sharpest hub: Ca 10, blast 16, only 16
  decisions. Worth keeping thin.

## Class level (class_metrics.py)

107 classes, 23 of them declaration-only. Mean LCOM4 0.71, mean CBO 1.58.

- **Cohesion is a non-issue.** After filtering the constructs that are isolated
  by construction (see PROBLEMS.md), exactly one LCOM4 finding survives, and it
  is cosmetic. No class in `src` needs splitting on cohesion grounds.
- **CBO is the metric that carries information.** Four classes sit at or above
  the 90th percentile: `R2DreamerAgent` (CBO 11), `RoutedCompositeEncoder` (8),
  `JAXVGGTFeatureExtractor` (8), `HouseVoxelsAdapter` (6).
- `JAXVGGTFeatureExtractor` is the heaviest class in the repo at WMC 89 across
  31 methods, but cohesive (LCOM4 1). It is big, not tangled.
- Inheritance is flat: DIT is 0 or 1 everywhere. No deep hierarchies.

## Next steps if resumed

1. Add a `--exclude src.vggt` flag to see the ranking of authored code alone.
2. Weight import edges by imported-name count (see PROBLEMS.md).
3. Add git churn as a third axis for a true Tornhill hotspot map.

## Not done

No test under `tests/prototyp/complexity-analysis/`. The script is read-only
analysis with no consumers, so it was not worth a test yet. If it graduates to
`scripts/` or a CI gate, it needs one.
