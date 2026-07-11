## ADDED Requirements

### Requirement: Characterize VGGT point-cloud output at three temporal scales
The investigation SHALL measure how VGGT's per-frame `world_points` output
accumulates and drifts at three scales — within one image, across one episode, and
across multiple episodes of one scene — using only read-only exercise of the
existing extractor (no new accumulation mechanism).

#### Scenario: Single-image output profile
- **WHEN** a single frame is passed through the VGGT DPT head
- **THEN** the per-pixel point yield, confidence distribution, and point-map noise
  are recorded as the single-image baseline

#### Scenario: Within-episode consistency
- **WHEN** an episode's frames are streamed through the extractor in order
- **THEN** the world-frame consistency of overlapping geometry across frames is
  measured, and the frame index at which cache eviction begins dropping previously
  observed geometry is identified

#### Scenario: Cross-episode behavior under reset modes
- **WHEN** multiple episodes of the same scene are streamed under `ResetMode.FULL`
  and again under `ResetMode.PERSIST_SCENE`
- **THEN** the drift and geometric consistency of the accumulated output are compared
  between the two modes for the same scene

#### Scenario: KV-cache saturation across many episodes
- **WHEN** a long sequence of many episodes of one scene is streamed under
  `ResetMode.PERSIST_SCENE` until the aggregator KV cache reaches and exceeds its
  eviction budget
- **THEN** the point at which the bounded cache saturates is identified, and the
  resulting geometric degradation past saturation (drift, dropped previously observed
  regions, loss of world-frame consistency) is characterized as a function of
  episode count — capturing the catastrophic-drift regime that motivates external
  persistent memory (and that InfiniteVGGT's rolling memory targets)

#### Scenario: Visual export around saturation
- **WHEN** the diagnostic flag is enabled and the many-episode stream approaches,
  reaches, and passes KV-cache saturation
- **THEN** colored PLY snapshots of VGGT's accumulated cloud are exported at those
  checkpoints (reusing the existing PLY writer) so the drift and degradation can be
  inspected and compared in a point-cloud viewer, not only read from plots

### Requirement: Derive a criterion for whether and where external memory is needed
The investigation SHALL produce an explicit, measurement-backed criterion stating
whether an external persistent point-cloud memory is required and, if so, in which
temporal regime VGGT's internal (bounded, evicting) memory stops sufficing.

#### Scenario: Regime decision gates downstream tasks
- **WHEN** the three-scale characterization is complete
- **THEN** the report SHALL state the regime(s) in which external memory is necessary,
  and the mechanism comparison (Task 1b) SHALL be run only within that regime

#### Scenario: Internal memory suffices
- **WHEN** the characterization shows VGGT's internal memory keeps the cloud
  consistent across all scales of interest
- **THEN** this SHALL be reported as a finding, and the external-mechanism comparison
  SHALL be scoped down or marked moot accordingly
