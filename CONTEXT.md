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

**Review Automation**:
An autonomous repository workflow that reviews pull requests against their linked Linear issue and may merge only low- or medium-risk pull requests. High-risk pull requests require human review.
_Avoid_: cronjob, auto-merge bot

**Risk Tier**:
The review automation's classification of a pull request's merge risk as low, medium, or high. Experiment and training code is medium risk when the linked issue is clear and validation passes; high risk is reserved for automation authority, infrastructure, compatibility, ambiguous criteria, or weakened validation.
_Avoid_: confidence score, priority

**Correctness Standard**:
The review automation's basis for deciding whether a pull request satisfies its linked Linear issue without violating repository expectations. The linked Linear issue is primary, but repository tests, documentation, and a dedicated review standard also constrain correctness.
_Avoid_: review prompt, acceptance check

**Review Standard**:
A dedicated repository policy document that defines merge eligibility, risk tiers, validation expectations, and Linear follow-up behavior for pull request review. Agents and automation reference it, but it is not itself an agent instruction file.
_Avoid_: AGENTS.md, bot prompt

**Needs-Human Follow-up**:
A Linear issue or sub-issue labeled `needs-human` that records work the review automation must not decide or complete silently. Medium-risk fixes and feature ideas found during review become needs-human follow-ups unless a human explicitly commands the automation to handle them.
_Avoid_: human label, TODO, review note

**Linear Command**:
A human instruction written in Linear that grants or withholds review automation authority for a linked pull request. The command vocabulary is `review: fix`, `review: merge`, and `review: hold`; Linear commands are preferred over pull request comments because the Linear issue is the source of task authority.
_Avoid_: PR command, slash command

**Review Completion**:
The review automation's finalization of a correct low- or medium-risk pull request by merging it, commenting on the linked Linear issue, and moving that Linear issue to Done.
_Avoid_: closeout, cleanup
