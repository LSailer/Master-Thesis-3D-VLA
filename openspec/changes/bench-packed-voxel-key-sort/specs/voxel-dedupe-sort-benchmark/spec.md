## ADDED Requirements

### Requirement: Packed-key sort variant is semantically equivalent
The packed uint64 single-key sort variant of `_unique_frame_voxels` SHALL
select exactly the same representative row (XYZ and RGB) per voxel key as the
lexsort baseline for every frame whose voxel keys lie within ±(2^20 − 1) per
axis, including frames with duplicate keys, non-finite points, and
low-confidence points.

#### Scenario: Randomized equivalence check
- **WHEN** a randomized frame with heavy voxel-key duplication, NaN rows, and
  sub-threshold confidence rows is processed by both sort variants
- **THEN** the set of active (key → representative XYZ/RGB) pairs is
  identical between the two variants

#### Scenario: Out-of-range keys are invalidated
- **WHEN** a frame contains points whose voxel key exceeds ±(2^20 − 1) on any
  axis
- **THEN** the packed variant treats those rows as invalid (sentinel key,
  never active), the same way non-finite points are treated

### Requirement: Landing decision is benchmark-gated
The packed variant SHALL only replace the lexsort baseline if
`scripts/r2dreamer/bench_graph_vs_buffer.py` measures a clear steady-state
buffer-add improvement on the target GPU (>15% median improvement over at
least 50 timed iterations). Otherwise the lexsort SHALL be kept and the
measured numbers recorded in the change.

#### Scenario: Packed variant wins
- **WHEN** the GPU benchmark shows the packed variant beats the baseline by
  more than the decision threshold
- **THEN** the packed variant becomes the only implementation, the lexsort
  path is removed, and the key-range precondition is documented at the pack
  site

#### Scenario: Packed variant does not win
- **WHEN** the GPU benchmark shows no clear improvement
- **THEN** the lexsort baseline is kept, the packed variant is removed, and
  the benchmark numbers are recorded in the change before archiving

### Requirement: Benchmark supports variant selection
The buffer benchmark SHALL allow selecting the sort variant
(lexsort or packed) so both can be timed in the same session on identical
synthetic frames.

#### Scenario: A/B timing run
- **WHEN** the benchmark is invoked once per variant with the same seed and
  frame size
- **THEN** it reports steady-state buffer-add timings for each variant that
  are directly comparable
