## ADDED Requirements

### Requirement: Persistent point-cloud memory with running-centroid nodes
The graph scene memory SHALL accumulate ingested points into a fixed-shape device
state whose nodes are running centroids of merged points, storing a float32 position
accumulator and integer count per node, and publishing node positions as bfloat16
only at snapshot time.

#### Scenario: Merge into an existing node
- **WHEN** an ingested point falls within the merge radius of an existing node
- **THEN** the point is accumulated into that node's float32 position accumulator and
  count, refining its centroid, without creating a new node

#### Scenario: Spawn a new node
- **WHEN** an ingested point has no existing node within the merge radius
- **THEN** a new node is allocated in a free slot, subject to fixed capacity, with
  overflow tracked when capacity is exhausted

#### Scenario: Snapshot precision
- **WHEN** the memory publishes its stored cloud for downstream consumption
- **THEN** node positions are emitted as bfloat16 derived from the float32
  accumulator divided by count, while accumulation itself remains float32

### Requirement: Edge-based consolidation that reclaims capacity
The graph scene memory SHALL maintain fixed-degree adjacency edges and periodically
consolidate connected nodes whose centroids have drifted within the merge radius,
merging them via bounded parallel connected-components and freeing the reclaimed
slots.

#### Scenario: Drifted nodes merge at consolidation
- **WHEN** two connected nodes' centroids are within the merge radius at a
  consolidation pass
- **THEN** their statistics are combined into one root node and the other slot is
  freed for reuse

#### Scenario: Deterministic merge under ties
- **WHEN** multiple nodes are mutually mergeable
- **THEN** merging resolves to a canonical root deterministically, independent of
  scatter order, so results are reproducible

#### Scenario: Consolidation cadence
- **WHEN** an episode boundary is reached or occupancy exceeds the configured
  threshold
- **THEN** a consolidation pass runs; otherwise the per-frame path does no global
  consolidation work

### Requirement: Single scale knob comparable to the hash table
The graph scene memory SHALL expose a single scale knob (merge radius) that governs
both the merge-versus-spawn decision and the consolidation merge, semantically
equivalent to the voxel hash table's `voxel_size`, so the two mechanisms differ only
by mechanism in the benchmark.

#### Scenario: Tied radii yield one knob
- **WHEN** the graph memory is configured
- **THEN** the assignment radius and consolidation radius are tied to a single merge
  radius `r` that matches the physical meaning of the hash table's `voxel_size`
