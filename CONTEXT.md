# Master Thesis 3D-VLA

This context defines the project language for R2Dreamer/VGGT ObjectNav experiments.

## Language

**Observation Preparation**:
The boundary that turns environment observations into replay-buffer observations and agent-ready observations for a chosen input mode. It owns the input mode's contract language: shapes, dtypes, normalization, reset hooks, and packing into Encoder Module tensors.
_Avoid_: encoder, preprocessing, adapter-only

**Encoder Module**:
The Flax module inside `R2DreamerAgent` that maps prepared observation tensors to Dreamer embeddings consumed by the RSSM.
_Avoid_: encoder contract, VGGT extractor, adapter

**Feature Extractor**:
An external or auxiliary model that derives features from raw environment observations before replay or acting, such as VGGT.
_Avoid_: encoder module, Dreamer encoder

**Encoder Input Contract**:
The agreement inside Observation Preparation that connects replay-buffer observations and agent-ready observations to an Encoder Module for one input mode.
_Avoid_: encoder, adapter spec

**Observation Form Contract**:
The form vocabulary inside the Encoder Input Contract for the observations that Observation Preparation accepts, produces, stores, and turns into Encoder Module input. A form includes structure, shape, dtype, and observation-specific metadata.
_Avoid_: shape constants, buffer schema, shape contract

**Prepared Observation**:
The result of preparing one environment observation for both replay storage and immediate agent use. It contains a replay-buffer observation and an agent-ready observation, even when those two forms are identical.
_Avoid_: transformed observation, adapter output

**ReplaySequenceBatch**:
Raw fixed-length sequences sampled from replay storage before conversion into agent training format.
_Avoid_: raw batch, sampled batch, replay dict

**TrainingBatch**:
Agent-ready fixed-length sequences consumed by R2Dreamer training.
_Avoid_: batch, train dict, agent dict

**ObservationBatch**:
Time-aligned observation sequences for one sampled replay window.
_Avoid_: obs dict, observation dict, input batch

**EpisodeBoundaryBatch**:
Time-aligned flags that mark starts and episode ends within sampled sequences.
_Avoid_: done flags, reset flags, terminal dict

