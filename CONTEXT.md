# Master Thesis 3D-VLA

This context defines the project language for R2Dreamer/VGGT ObjectNav experiments.

## Language

**Adapter**:
The per-variant boundary that turns one environment observation into the routed fields for that experiment arm. It owns the arm's observation language: render resolution, frozen extraction, shapes, dtypes, normalization, and which branch consumes what.
_Avoid_: encoder, preprocessing, observation preparation

**Routed Field**:
One named value an Adapter emits for a step, carrying its own routing: the Encoder Branch that consumes it, whether it is replayed, and whether it is the Decoder Target. `AdapterField` in `src/adapters/contract.py`.
_Avoid_: input mode, encoder input contract, shape contract, prepared observation

**Encoder Branch**:
The Flax module that consumes one Routed Field and produces its part of the Dreamer embedding. Selected by the field's `Encoder` enum member; the modules live in `src/r2dreamer/encoders/`.
_Avoid_: encoder contract, VGGT extractor, adapter

**Composite Encoder**:
The Flax module inside `R2DreamerAgent` that runs one Encoder Branch per Routed Field and fuses their outputs into the embedding the RSSM consumes. Fusion applies exactly when a variant has more than one branch.
_Avoid_: encoder module, hybrid encoder

**Feature Extractor**:
An external or auxiliary model that derives features from raw environment observations before replay or acting, such as VGGT. Adapters that need one hold it and call it themselves.
_Avoid_: encoder branch, Dreamer encoder

**Decoder Target**:
The single Routed Field the debug decoder probe reconstructs. It must be replayed, because the probe reads its target from the sampled batch.
_Avoid_: decoder input, reconstruction key

**ReplaySequenceBatch**:
Raw fixed-length sequences sampled from replay storage before conversion into agent training format.
_Avoid_: raw batch, sampled batch, replay dict

**TrainingBatch**:
Agent-ready fixed-length sequences consumed by R2Dreamer training.
_Avoid_: batch, train dict, agent dict

## Engineering rules

- Prefer `jax.Array`/`jax.numpy` for arrays that are part of the training data path.
  Use NumPy only at explicit I/O or library-compatibility boundaries, preferably as
  a documented fallback.
- Prefer `bfloat16` over `float32` for training-path floating-point arrays unless
  numerical precision or external-library contracts require another dtype.

**ObservationBatch**:
Time-aligned observation sequences for one sampled replay window.
_Avoid_: obs dict, observation dict, input batch

**EpisodeBoundaryBatch**:
Time-aligned flags that mark starts and episode ends within sampled sequences.
_Avoid_: done flags, reset flags, terminal dict

