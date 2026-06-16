# R2Dreamer Context

Shared language for the R2Dreamer training pipeline and replay data flow.

## Language

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
Time-aligned flags that mark starts, ends, and true terminals within sampled sequences.
_Avoid_: done flags, reset flags, terminal dict
