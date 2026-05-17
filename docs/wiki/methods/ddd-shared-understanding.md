# DDD Shared Understanding — R2Dreamer ObjectNav

**Status:** draft for domain review  
**Date:** 2026-05-11  
**Scope:** shared domain language before any DDD refactor. No production code changes are implied by this page.

## Purpose

This page is the starting point for refactoring with Domain-Driven Design (DDD). The goal is to make the code speak the same language as the thesis domain, so future refactors improve clarity without changing behavior.

Following Colla and Acerbis (2025, p. 33), the first step is shared understanding:

1. understand the domain before refactoring;
2. identify places where the code does not express the domain model well;
3. apply DDD patterns only where they clarify the model;
4. preserve behavior with tests and small changes;
5. integrate frequently;
6. review the language and model with domain experts.

## Domain statement

The project evaluates whether an R2Dreamer world-model agent performs better on HM3D ObjectNav when its observations are encoded as 3D VGGT features instead of 2D CNN image features.

The thesis-critical comparison is structural: **3D-only encoder vs 2D-only encoder at the same observation cadence**. ADR-0002 explicitly rejects frame-skip because stale features or mixed 2D/3D inputs would confound that comparison.

## Subdomains

| Subdomain | Type | Why it matters |
|---|---|---|
| ObjectNav task | Core | Defines what success means: navigate to the goal object category in Habitat/HM3D within an episode horizon. |
| Perception encoding | Core | Defines the comparison under study: CNN RGB encoding vs VGGT 3D feature extraction. |
| World-model learning | Core | R2Dreamer learns latent dynamics, reward, continuation, actor, critic, and representation alignment. |
| Training orchestration | Supporting | Runs prefill, acting, replay sampling, train-ratio updates, logging, checkpointing, and resume. |
| Curriculum and evaluation | Supporting | Selects scenes/goals/levels and reports SR/SPL/path metrics for comparable experiments. |
| Experiment execution | Supporting | SLURM, chained runs, W&B, manifests, output paths, and reproducibility contracts. |
| Agent-loop automation | Generic/supporting | Repository automation for issues and PRs; not part of the ObjectNav learning domain. |

## Proposed bounded contexts

These contexts are a first draft. They should be reviewed before code movement.

### 1. Navigation Task Context

Owns the task semantics: episode, ObjectNav goal, action, reward, termination, success, SPL, path length, scene, and curriculum level.

Current code/docs:

- `modules/envs/habitat.py`
- `modules/envs/habitat_r2dreamer.py`
- `modules/r2dreamer/launch/habitat_setup.py`
- `data/curriculum/level*.json`
- `docs/wiki/experiments/*.md`

Key language:

- Episode
- ObjectNav goal / object category
- Scene / house
- Step
- Action: `stop`, `forward`, `left`, `right`
- Reward
- Success
- SPL
- Path length
- Shortest path
- Curriculum level: L1, L2, L3, L4

Context boundary:

- Produces environment observations and task metrics.
- Should not know whether the agent uses CNN, VGGT, RSSM, or JAX internals.

### 2. Perception Encoding Context

Owns conversion from environment observation to agent-consumable representation.

Current code/docs:

- `modules/r2dreamer/adapters/obs_adapter.py`
- `modules/r2dreamer/adapters/vggt_adapter.py`
- `modules/r2dreamer/launch/encoders.py`
- `modules/vggt/`
- `docs/wiki/methods/vggt-r2dreamer-callchain.md`
- `docs/wiki/methods/vggt-jax-streaming.md`
- `docs/wiki/methods/vggt-jax-eviction-recompile.md`
- `docs/wiki/methods/encoder-fusion-plan.md`

Key language:

- Observation
- RGB frame
- Agent state
- Camera extrinsics
- Camera intrinsics
- Encoder
- CNN encoder
- VGGT encoder
- VGGT feature extractor
- World points
- Camera pose
- Aggregator tokens
- Replay feature
- Observation adapter
- Feature vector
- KV-cache
- Cache eviction
- Static budget

Context boundary:

- Receives task observations from Navigation Task Context.
- Emits `buffer_obs` and `agent_obs` for training/acting.
- Should hide framework-specific VGGT details from the world-model agent.

Possible DDD interpretation:

- `ObsAdapter` is currently an anti-corruption layer between the environment observation model and the agent/replay input model.
- `Encoder` in `launch/encoders.py` is a factory/port for perception variants, not the same concept as the trainable Flax `ConvEncoder` or `VGGTEncoder` in `networks.py`.

Naming risk:

- The term `VGGTEncoder` currently exists in two contexts:
  - `modules/r2dreamer/launch/encoders.py::VGGTEncoder`: external perception extractor wiring.
  - `modules/r2dreamer/networks.py::VGGTEncoder`: trainable projection from precomputed VGGT feature vector to RSSM embedding.
- A DDD refactor should disambiguate these names before moving behavior.

### 3. World-Model Agent Context

Owns the learning agent itself: latent state, dynamics, representation, imagination, actor, critic, losses, and optimizer state.

Current code/docs:

- `modules/r2dreamer/agent.py`
- `modules/r2dreamer/networks.py`
- `modules/r2dreamer/config.py`
- `modules/shared/optim.py`
- `docs/wiki/methods/world-model-training-loop.md`
- `docs/wiki/methods/shape-debugging.md`
- `docs/wiki/methods/cross-correlation-matrix.md`

Key language:

- R2Dreamer agent
- World model
- RSSM
- Prior
- Posterior
- Stochastic state / `stoch`
- Deterministic state / `deter`
- Feature / `feat`
- Embedding / `embed`
- Actor
- Critic
- Slow target critic
- Reward head
- Continuation head
- Projector
- Representation loss
- Barlow Twins loss / cross-correlation matrix
- KL loss
- Imagination
- Imagination horizon
- Lambda return
- Return EMA
- Train step

Context boundary:

- Consumes agent observations, replay batches, and RNG keys.
- Produces actions and training metrics.
- Should not know about Habitat scene paths, curriculum JSONs, SLURM, W&B naming, or VGGT cache budgets except through explicit input contracts.

Possible DDD interpretation:

- `R2DreamerAgent` is an aggregate-like boundary around parameters, optimizer state, acting state, target critic, and return EMA.
- Pure functions such as `_imagine`, `_lambda_return`, and loss functions are domain services inside the World-Model Agent Context.
- `R2DreamerConfig` is currently both a model configuration and a carrier of training defaults; split only if this ambiguity starts causing mistakes.

### 4. Experience Replay Context

Owns storage and sampling of agent experience.

Current code/docs:

- `modules/shared/replay_buffer.py`
- `modules/r2dreamer/trainer.py::convert_batch`
- `modules/r2dreamer/launch/parity/batch_utils.py`
- `docs/wiki/methods/world-model-training-loop.md`
- `docs/wiki/methods/training-orchestration.md`

Key language:

- Transition
- Replay buffer
- Ring buffer
- Buffer observation
- Replay feature
- Sequence window
- `seq_len`
- `is_first`
- `done` / `is_last`
- `terminal` / `is_terminal`
- Validation replay dataset
- Batch conversion

Context boundary:

- Stores observations/actions/rewards/episode boundary flags.
- Samples fixed-length windows for world-model training.
- Should not choose actions, compute losses, or know episode metrics beyond boundary flags.

Naming risk:

- `done`, `is_last`, `terminal`, and `is_terminal` need explicit definitions because they are easily confused:
  - `done` / `is_last`: environment episode ended.
  - `terminal` / `is_terminal`: terminal success condition used for continuation semantics.

### 5. Training Orchestration Context

Owns the training run lifecycle.

Current code/docs:

- `modules/r2dreamer/trainer.py`
- `modules/r2dreamer/launch/train.py`
- `modules/r2dreamer/launch/evaluate.py`
- `modules/r2dreamer/manifest.py`
- `docs/wiki/methods/training-orchestration.md`
- `docs/wiki/methods/launcher-refactor.md`

Key language:

- Training run
- End-to-end run
- Prefill
- Acting
- Train loop
- Train ratio
- Checkpoint
- Resume
- Validation loss
- Episode metrics
- Manifest
- W&B run
- Output directory

Context boundary:

- Coordinates env, perception adapter, agent, replay buffer, checkpointing, and metrics.
- Should contain workflow policy, not model math.

Possible DDD interpretation:

- `Trainer` is an application service. It coordinates bounded contexts but should not become the owner of their domain rules.
- `TrainerConfig` is a run lifecycle value object.

### 6. Experiment Execution Context

Owns platform-specific execution and reproducibility.

Current code/docs:

- SLURM sbatch files
- `output/`
- `wandb/`
- `docs/wiki/methods/l4-profiling.md`
- `docs/adr/0002-no-frame-skip-thesis-integrity.md`
- `CONTEXT.md`

Key language:

- SLURM run window
- Chained training run
- H100
- Login node
- Dev GPU partition
- Wall-clock cap
- Job id
- Run id
- Artifact
- Metrics CSV

Context boundary:

- Launches and resumes experiments.
- Should not leak scheduler concerns into the World-Model Agent Context.

## Ubiquitous language draft

| Term | Definition | Preferred code/docs name | Avoid / clarify |
|---|---|---|---|
| ObjectNav task | Navigation task where the agent must find an object category in a Habitat scene. | ObjectNav | generic `task` when the context is ambiguous |
| Episode | One environment rollout from reset until success, failure, or max-step timeout. | episode | trajectory, rollout, run when specifically meaning env episode |
| Step | One environment transition caused by one action. | step / env step | frame when action semantics matter |
| End-to-end run | A complete training experiment including acting, replay, training, logging, checkpoints. | end-to-end run | episode, job |
| Training run | The software lifecycle managed by `Trainer.run()`. | training run | experiment when only code lifecycle is meant |
| SLURM run window | Maximum scheduler wall-clock duration for one job. | SLURM run window | training horizon |
| Chained training run | Multiple SLURM jobs resuming checkpoints for one experiment. | chained training run | restart, rerun |
| Curriculum level | Named difficulty/data slice L1-L4. | curriculum level / L1-L4 | dataset split unless referring to train/eval split |
| Observation | Environment output before perception adaptation. | observation / obs_dict | feature, embedding |
| Buffer observation | Representation stored in replay. RGB for CNN, feature vector for VGGT. | buffer_obs | agent_obs |
| Agent observation | Representation passed to the agent for action selection. | agent_obs | buffer_obs |
| Encoder | Perception variant selected for an experiment. | perception encoder | trainable projection layer unless context is World-Model Agent |
| CNN encoder | 2D RGB path where R2Dreamer's internal CNN maps image to embedding. | CNN encoder | image encoder if comparing with VGGT |
| VGGT feature extractor | Frozen external 3D perception model producing world points and camera pose. | VGGT feature extractor | VGGTEncoder when referring to extractor wiring |
| VGGT projection encoder | Trainable Dense projection from VGGT feature vector to RSSM embedding. | VGGT projection encoder | VGGT feature extractor |
| World points | VGGT per-patch 3D point output. | world_points | pointcloud unless discussing visualization |
| Camera pose | VGGT episode-relative pose output. | camera_pose | absolute pose |
| Agent state | Habitat camera extrinsics + intrinsics vector. | agent_state | RSSM state |
| RSSM state | Latent recurrent state with stochastic and deterministic parts. | RSSM state | agent_state |
| Stochastic state | Discrete RSSM latent categorical state. | stoch | latent if the part matters |
| Deterministic state | RSSM recurrent deterministic state. | deter | hidden state if ambiguous |
| Embedding | Encoder output consumed by RSSM posterior. | embed / embedding | feature when meaning RSSM feature |
| RSSM feature | Concatenated latent feature used by heads. | feat / RSSM feature | embedding |
| Imagination | Model rollout from replay states without real observations. | imagination | planning if no model rollout is involved |
| Replay window | Fixed-length contiguous sampled sequence from replay. | replay window / sequence window | episode |
| `is_first` | Flag that resets RSSM state at the start of a sampled or live episode segment. | is_first | start unless documented |
| `is_last` | Environment episode ended. | is_last / done | terminal success |
| `is_terminal` | Success terminal condition used for continuation semantics. | is_terminal / terminal | done |
| Success rate | Fraction of episodes reaching ObjectNav success. | SR / success rate | accuracy |
| SPL | Success weighted by path length. | SPL | efficiency without definition |

## Context map

```text
Navigation Task Context
  produces Observation + task metrics
        |
        v
Perception Encoding Context
  ObsAdapter translates Observation -> buffer_obs + agent_obs
        |
        +------------------------+
        |                        |
        v                        v
Experience Replay Context     World-Model Agent Context
  stores transitions           acts and trains from agent_obs/replay batches
        |                        |
        +-----------+------------+
                    v
Training Orchestration Context
  coordinates prefill, acting, train-ratio updates, checkpoints, logging
                    |
                    v
Experiment Execution Context
  schedules, resumes, records artifacts, supports thesis comparisons
```

## Existing model/code mismatches to consider before refactoring

| Area | Current symptom | DDD concern | Candidate next step |
|---|---|---|---|
| Encoder naming | `VGGTEncoder` means different things in launcher and networks. | Ubiquitous language conflict. | Rename launcher concept to `VGGTPerceptionEncoder` or network concept to `VGGTProjectionEncoder` in a small tested refactor. |
| Observation language | `obs`, `features`, `buffer_obs`, and `agent_obs` cross boundaries. | Missing explicit value objects/contracts. | Introduce typed aliases or dataclasses only at context boundaries, not deep inside JAX hot paths. |
| Done/terminal flags | `dones -> is_last`, `terminals -> is_terminal`. | Domain semantics can be misread. | Add focused tests and glossary comments for episode-ended vs success-terminal. |
| Trainer ownership | `Trainer` knows env, adapter, replay, agent, metrics, checkpoint, W&B. | Application service risks becoming god object. | Keep as coordinator; extract domain rules only when behavior changes require it. |
| Config shape | `R2DreamerConfig` carries architecture and some run defaults. | Mixed value objects. | Split only if mistakes appear between model config and run config. Current `TrainerConfig` already separates run lifecycle. |
| VGGT flattened vector | `4116 = 37*37*3 + 9` appears as a technical shape. | Shape hides domain meaning. | Wrap constants around `world_points` + `camera_pose` vocabulary; keep flat array at JAX/replay boundary. |
| Agent state ambiguity | Habitat `agent_state` is camera calibration/pose; R2Dreamer also has acting state. | Same words for unrelated concepts. | Prefer `camera_agent_state` or `habitat_camera_state` if this resurfaces. |

## DDD refactoring strategy

Use this order; do not start by moving files.

1. Validate the bounded contexts and glossary with domain experts.
2. Update `CONTEXT.md` only for terms that are accepted and stable.
3. Add characterization tests around boundary contracts before renaming:
   - adapter output shape/dtype;
   - CNN vs VGGT agent observation shape;
   - replay `is_first` / `is_last` / `is_terminal` semantics;
   - train/eval launcher preset wiring.
4. Refactor one ubiquitous-language conflict at a time.
5. Keep hot JAX paths array-based unless a value object can be erased before JIT boundaries.
6. Run the existing fast test slice after every small change.
7. Review with domain experts after each context-boundary refactor.

## Candidate first refactor slice

The safest first DDD refactor is a naming-only boundary clarification:

**Goal:** disambiguate VGGT perception extraction from the trainable VGGT projection layer.

Proposed language:

- `VGGTPerceptionEncoder`: launcher/perception object that owns the frozen VGGT extractor and creates `VGGTObsAdapter`.
- `VGGTProjectionEncoder`: Flax module that projects the flat VGGT replay feature into the RSSM embedding space.

Why first:

- It addresses an actual ubiquitous-language conflict.
- It is small and testable.
- It does not change model behavior, experiment semantics, or JAX array contracts.

Minimum tests before/after:

```bash
uv run pytest modules/r2dreamer/launch/tests/test_registries.py modules/r2dreamer/launch/tests/test_encoders.py -m "not gpu" -q
uv run pytest modules/r2dreamer/tests/test_agent.py modules/r2dreamer/tests/test_vggt_encoder.py -q
```

On the HPC, GPU tests/runs should be launched through `srun` on the `dev_gpu_h100` partition/name; direct JAX CUDA initialization on the login node is expected to fail.

## Open questions for domain review

1. Should the core thesis domain name be **ObjectNav thesis comparison**, **3D-vs-2D encoder comparison**, or something else?
2. Is **Perception Encoding Context** the right name, or should it be **Observation Encoding Context**?
3. Should `agent_state` from Habitat be renamed in the ubiquitous language to avoid collision with R2Dreamer acting/RSSM state?
4. Is `is_terminal` truly best defined as success-terminal in this project, or should the continuation semantics be documented differently?
5. Should VGGT `camera_pose` always be described as **episode-relative** in code comments, not just docs?
6. Are curriculum levels L1-L4 part of the Navigation Task Context or a separate Curriculum Context?

## References

- `CONTEXT.md` — current canonical glossary.
- `docs/adr/0002-no-frame-skip-thesis-integrity.md` — preserves thesis comparison semantics.
- `docs/wiki/methods/world-model-training-loop.md` — acting vs replay windows vs imagination.
- `docs/wiki/methods/training-orchestration.md` — Trainer and ObsAdapter extraction.
- `docs/wiki/methods/vggt-r2dreamer-callchain.md` — VGGT to R2Dreamer data flow.
- `docs/wiki/methods/launcher-refactor.md` — encoder ABC, shims, and test pyramid.
