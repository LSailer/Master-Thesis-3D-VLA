# Open problems / risks / research notes

Append as they come up.

## Open risks (pre-first-run)

1. **GIL overlap is unproven.** The whole threaded hypothesis rests on
   habitat-sim's C++ `step()` and JAX's device execution releasing the GIL.
   Neither is documented for our exact versions; the smoke run is the test.
   If overlap gain ~1.0x, check whether the bottleneck is the GIL or a shared
   resource (single GPU used by both VGGT-free CNN train_step and habitat GL).

2. **Adapter thread-safety is NOT covered by the buffer lock.** In threaded
   mode the actor calls `adapter.prepare_env_step()` while the learner calls
   `adapter.augment_replay_batch()` (inside `ExperienceCollector.sample`).
   For the CNN adapter both paths look stateless, so the smoke uses
   `habitat-l1-cnn`. Live house-point adapters (PERSIST_SCENE, growing point
   buffers) mutate shared state on the env path and read it on the sample
   path — running those under `--mode threaded` needs an adapter-level lock
   or a snapshot handoff. Do not benchmark house modes threaded until then.

3. **Agent state races (accepted for the prototype).** `train_step` reassigns
   `agent.params` (and opt/EMA state) while the actor thread reads
   `agent.params` in `act()`. Attribute rebinding is atomic under CPython, so
   `act` sees either the old or the new whole tree — acceptable staleness for
   a schedule benchmark, but a real production loop should hand over params
   explicitly (e.g. versioned snapshot) before graduating.

4. **GPU contention / compile serialization.** `act()` (small JIT fn) and
   `train_step` (big JIT fn) share one H100. Even with the GIL released the
   two threads' XLA launches serialize on the device stream; the measurable
   gain is then limited to overlapping habitat CPU/GL time with GPU compute.

5. **Episode mode may see zero mid-run boundaries on tiny smokes.** L1
   episodes run up to 500 steps; a 600-step smoke may hit only one boundary.
   The final credit drain guarantees identical train-step totals regardless,
   but the "burstiness" measurement needs >= a few episodes to be meaningful.
   Bump `--steps` for a real comparison.

6. **Exit codes are meaningless on this cluster.** Habitat GL teardown can
   SIGABRT after a fully successful run. Judge every run by
   `outputs/prototype/train_scheduling/<run>/MANIFEST.json` (`status` field).
   The runner hard-exits (`os._exit(0)`) after writing the manifest to skip
   env teardown in habitat runs.

7. **Buffer lock granularity.** One lock around the whole `sample()` gather
   holds writers off for the duration of the fancy-index copy (~59 ms at
   production shape per MEMORY). If the threaded mode wins, a follow-up is a
   finer scheme (copy ring indices under lock, gather outside), which needs
   care with ring wraparound overwrites.

## Dead ends

(none yet)
