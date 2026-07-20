## ADDED Requirements

### Requirement: Mechanism-agnostic ingestion of the VGGT DPT stream
The benchmark SHALL ingest the same DPT-head stream (`world_points`, `confidence`,
RGB) into any scene-memory backend behind a common interface, applying the same
confidence gate to every backend so measured differences reflect the mechanism and
not input or thresholding differences.

#### Scenario: Same stream to both backends
- **WHEN** a recorded VGGT point stream is replayed into the voxel hash table and
  into the graph memory
- **THEN** both receive identical points and confidence values, and the same
  confidence gate is applied to both

### Requirement: Measure ingestion speed with hot-path and consolidation reported separately
The benchmark SHALL measure ingestion throughput in points per second on warmed-up,
jitted code with device synchronization, and SHALL report the per-frame hot-path
throughput separately from any amortized consolidation cost.

#### Scenario: Throughput measurement excludes compilation warmup
- **WHEN** ingestion speed is measured
- **THEN** the first (compilation) call is excluded and device work is synchronized
  before timing stops

#### Scenario: Consolidation cost is not folded into hot-path throughput
- **WHEN** a backend performs periodic consolidation
- **THEN** the consolidation cost is reported as a separate figure from the per-frame
  ingestion throughput

### Requirement: Measure false-positive additions via a revisit oracle
The benchmark SHALL define false positives as points added on a frame whose correct
answer is "add nothing," using a controlled revisit oracle to establish that ground
truth.

#### Scenario: Repeated-frame revisit
- **WHEN** the same frame is ingested N times
- **THEN** points added beyond the first ingestion are counted as false positives

#### Scenario: Trajectory revisit
- **WHEN** a trajectory that loops back to previously observed geometry is replayed
- **THEN** additions during the revisit segment are measured against the near-zero
  oracle expectation

### Requirement: Measure coverage against a dense oracle without a ground-truth mesh
The benchmark SHALL measure coverage as the fraction of a dense, no-dedup oracle
cloud's occupied space (a fine reference voxelization) that lies within epsilon of a
stored point, requiring no ground-truth mesh.

#### Scenario: Coverage recall against the oracle
- **WHEN** a backend's stored points are compared to the dense oracle for a scene
- **THEN** the fraction of oracle-occupied reference cells within epsilon of a stored
  point is reported as coverage recall

### Requirement: Report point-set growth at three scales
The benchmark SHALL record cumulative stored-point count versus frame index at three
scales — within one image, across one episode, and across multiple episodes/scenes.

#### Scenario: Growth curves per scale
- **WHEN** a stream is ingested
- **THEN** growth curves are produced for each of the three scales, per backend

### Requirement: Matched-budget comparison protocol
The benchmark SHALL compare mechanisms on a coverage-versus-stored-point-count
(memory) frontier by sweeping each mechanism's scale knob, and SHALL read speed and
false-positive figures at matched budget rather than from a single arbitrary knob
pairing.

#### Scenario: Frontier rather than single point
- **WHEN** two mechanisms are compared
- **THEN** each mechanism's knob is swept to trace a coverage-vs-memory frontier, and
  speed/false-positive comparisons are taken at matched memory budget

#### Scenario: Recommendation states dominance or its absence
- **WHEN** the comparison is complete
- **THEN** the deliverable applies the decision rule (coverage and speed first,
  false-positive rate as tie-breaker) and states explicitly if neither mechanism
  dominates

### Requirement: Diagnostic point-cloud export for visual inspection
The benchmark SHALL export the accumulated cloud (and the dense oracle) as colored
PLY snapshots at defined checkpoints, in a format loadable by standard point-cloud
viewers (CloudCompare, Blender, MeshLab, Open3D), reusing the existing PLY writer.
Export SHALL be gated behind a diagnostic flag that is off during timing runs so
disk I/O never contaminates the ingestion-speed measurement.

#### Scenario: Export is off the timing hot path
- **WHEN** an ingestion-speed timing run executes
- **THEN** the diagnostic export flag is off and no PLY is written during the timed
  region, so points-per-second is unaffected by disk I/O

#### Scenario: Checkpoint export
- **WHEN** the diagnostic flag is enabled and a checkpoint is reached (per episode,
  at KV-cache saturation, and before and after each consolidation pass)
- **THEN** a colored PLY snapshot of the current stored cloud is written with a
  consistent, ordered naming so a sequence can be scrubbed/animated in a viewer

#### Scenario: Matched-checkpoint export across mechanisms and oracle
- **WHEN** the diagnostic flag is enabled and multiple backends are run on the same
  stream
- **THEN** each backend's cloud and the dense oracle are exported at matched
  checkpoints, enabling cloud-to-cloud (C2C) comparison of hash-table vs graph vs
  oracle in a viewer

#### Scenario: Attribute coloring to make changes visible
- **WHEN** a diagnostic PLY snapshot is written
- **THEN** points MAY be colored by a chosen attribute (DPT confidence,
  new-versus-existing this frame, or node id) so accumulation, drift, and
  consolidation changes are visible, not just the raw geometry
