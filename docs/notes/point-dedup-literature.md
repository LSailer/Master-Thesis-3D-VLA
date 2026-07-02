# Literature: fast point-cloud deduplication / voxel membership

Goal: replace the ~770 ms/step "is this voxel already known?" comparison in
`src/buffer/house_context_pose_buffer.py::add` with something compatible with
1M–2M closed-loop environment steps (budget: ~1 ms/step on GPU).

First, a framing correction: the measured 770 ms (see
`scratchpad/reduce_comparison_time/README.md`) was **not** all-pairs O(P²)
comparison — it was XLA recompilation from data-dependent shapes, plus
`jnp.unique`'s lexicographic sort, plus a host round-trip. The current
fixed-shape hash-table implementation already removes those. The literature
below is about the two standard GPU-scale answers — **spatial hashing** and
**sort-based compaction on space-filling-curve codes** — plus supporting
techniques.

---

## 1. Spatial hashing (the lineage the current code implements)

- **Teschner, Heidelberger, Müller, Pomerantes, Gross — "Optimized Spatial
  Hashing for Collision Detection of Deformable Objects", VMV 2003.**
  [PDF](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf)
  The origin of the exact hash function in `_hash_voxel_keys`
  (`x·73856093 ^ y·19349663 ^ z·83492791`). Studies hash-table sizing and cell
  size vs. collision rate — directly relevant to choosing
  `hash_table_size / capacity` load factor (they recommend load factors well
  below 0.5 for few collisions; your 2:1 ratio sits at the edge).

- **Nießner, Zollhöfer, Izadi, Stamminger — "Real-time 3D Reconstruction at
  Scale using Voxel Hashing", ACM TOG (SIGGRAPH Asia) 2013.**
  [PDF](https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf) ·
  [project](https://niessnerlab.org/projects/niessner2013hashing.html)
  The canonical "incrementally fuse depth frames into a sparse voxel world on
  GPU in real time" paper — your exact problem (they insert whole voxel blocks
  per depth frame at 30 Hz). Key trick: hash **blocks of 8³ voxels**, not
  individual voxels — cuts table pressure and insert count by ~512× and
  amortizes probing. Uses chaining + per-bucket locks; a JAX analogue is
  coarser keys with per-block payloads.

- **Dong, Lao, Kaess, Koltun — "ASH: A Modern Framework for Parallel Spatial
  Hashing in 3D Perception", IEEE TPAMI 2023.**
  [arXiv:2110.00511](https://arxiv.org/abs/2110.00511) ·
  [PDF](https://www.cs.cmu.edu/~kaess/pub/Dong23pami.pdf)
  The most directly relevant modern reference. Benchmarks GPU hash-map designs
  specifically on **point-cloud voxelization** and volumetric reconstruction,
  and its core design decision matches your code: decouple the hash **index
  structure** from flat **value buffers** addressed by integer indices (your
  `key_xyz/occupied` table vs. `store_xyz/store_rgb`). Open-sourced in Open3D
  (`open3d.core.HashMap`) — useful as a performance yardstick: they voxelize
  millions of points in a few ms, so ~1 ms/frame for 268k points is realistic.

- **Ashkiani, Farach-Colton, Owens — "A Dynamic Hash Table for the GPU",
  IPDPS 2018 (SlabHash).** [arXiv:1710.11246](https://arxiv.org/abs/1710.11246)
  · [code](https://github.com/owensgroup/SlabHash)
  Warp-cooperative chaining; ~512M updates/s and ~937M queries/s on a 2013-era
  K40. Good for understanding why *contention handling* (your
  `winner_by_slot` scatter-min per probe round) is the expensive part of
  parallel insertion, and how warp-level cooperation avoids it.

- **Alcantara et al. — "Real-Time Parallel Hashing on the GPU", ACM TOG
  (SIGGRAPH Asia) 2009.** Cuckoo hashing on GPU: guaranteed O(1) **bounded**
  lookups (≤4 probes) instead of open-addressing's unbounded probe chains.
  Relevant because your `max_probe_count=128` while-loop is the JIT-unfriendly
  part — cuckoo-style bounded probing turns it into a fixed, unrollable number
  of rounds.

## 2. Sort-based dedup on space-filling-curve codes (the alternative family)

The other standard GPU pattern: encode each voxel as **one integer** (Morton
code or packed key), radix-sort, mark first-of-run, compact. No hash table, no
probe loop, fully shape-static — and `jnp.sort` on a single int key is a fast
radix/merge sort, unlike the 4-array `lexsort` currently in
`_unique_frame_voxels`.

- **Karras — "Maximizing Parallelism in the Construction of BVHs, Octrees, and
  k-d Trees", HPG 2012.**
  [PDF](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf)
  The reference for Morton-code + radix-sort pipelines; shows the whole
  "quantize → 1 int key → sort → adjacent-diff unique → prefix-sum compact"
  toolkit that sparse-voxel systems build on.

- **TorchSparse++ (Tang et al., MICRO 2023)**
  [arXiv:2311.12862](https://arxiv.org/pdf/2311.12862) and **"Optimizing Sparse
  Convolution on GPUs with CUDA" (2024)**
  [arXiv:2402.07710](https://arxiv.org/html/2402.07710v1) — production
  sparse-conv engines whose voxelization step is exactly "quantize coords →
  unique". They use hashing (MinkowskiEngine) or sort-based unique
  (TorchSparse/spconv); both voxelize LiDAR-scale clouds in ~1 ms. Reading
  their voxelization kernels is the fastest way to see the state of practice.

- **GPU-Voxels (FZI), `octree/PointCloud.cu`**
  [code](https://github.com/fzi-forschungszentrum-informatik/gpu-voxels/blob/master/packages/gpu_voxels/src/gpu_voxels/octree/PointCloud.cu)
  — a readable open-source implementation of the Morton-sort-unique pipeline
  with Thrust.

## 3. Cross-frame novelty without a global table (probabilistic / hierarchical)

- **Bloom, CACM 1970 — "Space/time trade-offs in hash coding with allowable
  errors."** A Bloom filter as a *pre-filter*: a fixed-shape bitfield where
  `k` scatter-reads answer "definitely new / maybe seen". In JAX this is a few
  gathers on a `(m,)` uint32 array — no probe loop. False positives only drop
  duplicate-looking new points (bounded, tunable), never corrupt the store.
  Cheap first stage; full table only touches survivors.

- **Hornung, Wurm, Bennewitz, Stachniss, Burgard — "OctoMap: an efficient
  probabilistic 3D mapping framework based on octrees", Autonomous Robots
  2013.** The robotics-standard occupancy map; hierarchical membership tests
  and lossless pruning. More relevant conceptually (multi-resolution occupancy)
  than as a GPU recipe.

- **Museth — "VDB: High-Resolution Sparse Volumes with Dynamic Topology", ACM
  TOG 2013** (and **NanoVDB**, 2021): the film/sim-industry sparse voxel
  structure — a shallow B+-tree over voxel blocks with O(1) average random
  access. The "blocks, not voxels" lesson again.

- **Vizzo et al. — "KISS-ICP: In Defense of Point-to-Point ICP", RA-L 2023.**
  Uses a plain voxel hash map for online LiDAR mapping at >100 Hz on CPU —
  evidence that with block granularity and bounded probing this problem is
  sub-millisecond even without a GPU.

## 4. Books

- **Ericson — *Real-Time Collision Detection*, ch. 7 "Spatial Partitioning".**
  The best single textbook treatment of uniform grids, spatial hashing (incl.
  the Teschner scheme), hierarchical grids, and open addressing vs. chaining
  trade-offs.
- **Hwu, Kirk, El Hajj — *Programming Massively Parallel Processors* (4th
  ed.), chapters on radix sort, prefix sum (scan), and histogramming** — the
  primitives every sort-based dedup pipeline is composed of.
- **Pharr (ed.) — *GPU Gems 2/3***: chapters on scan (Blelloch-style) and
  stream compaction — the "mark + prefix-sum + scatter" idiom that replaces
  data-dependent `concatenate`.

---

## 5. Mapping to the current JAX implementation

Concrete ideas the literature suggests, in rough expected-payoff order:

1. **Pack the voxel key into one int32/int64** (e.g. 21 bits/axis into int64,
   or hash to int32) and sort *that* instead of the 4-key `lexsort` in
   `_unique_frame_voxels`. Single-key radix sort is the pattern from Karras
   2012 / TorchSparse; lexsort over 4 arrays is several times slower.
2. **Block granularity (Nießner 2013 / VDB):** dedup at 8³-voxel-block level
   in the hash table, keep per-voxel occupancy as a 512-bit mask payload.
   ~512× fewer inserts/probes per frame.
3. **Bloom-filter pre-filter (Bloom 1970):** fixed-shape bitfield lookup kills
   the common case (point already seen) with ~3 gathers, so the probe loop
   only runs over genuinely-new candidates. In a mostly-static house, after
   the first sweep almost everything is a duplicate — this is the asymptotic
   win for 1M–2M steps.
4. **Bound and unroll probing (Alcantara 2009):** replace the
   `lax.while_loop` (which serializes 128 potential rounds, each doing a
   full-table `scatter-min`) with a fixed small number of unrolled rounds +
   overflow counter; or restructure so the per-round `winner_by_slot`
   scatter-min operates on a `(P,)`-sized segment array instead of the
   `(hash_table_size,)` table (ASH §insertion discusses exactly this
   contention-resolution cost).
5. **Sort-merge against the store (sort-based family):** since
   `_unique_frame_voxels` already sorts the frame, keep the *global* store
   sorted by packed key too and do a shape-static merge
   (`searchsorted` + equality check) instead of hashing — O((P+N)) gathers,
   no probe loop at all. This is the Thrust `sort + unique + merge` idiom
   from GPU-Voxels.
