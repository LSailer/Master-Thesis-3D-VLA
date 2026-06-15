# Observation Preparation Decisions

Date: 2026-06-13

This note records the decisions from the architecture grilling session for deepening the encoder input path. It is not an ADR yet; the interface can still change before implementation.

## Language

- **Observation Preparation** is the boundary that prepares environment observations for replay-buffer storage and immediate agent use.
- **Encoder Module** is the Dreamer-side Flax module that maps prepared observation tensors to embeddings for the RSSM.
- **Feature Extractor** is an auxiliary model, such as VGGT, that derives features from raw environment observations.
- **Encoder Input Contract** is the agreement inside Observation Preparation that connects prepared observations to an Encoder Module.
- **Observation Form Contract** is the form vocabulary inside the Encoder Input Contract for the observations that Observation Preparation accepts, produces, stores, and turns into Encoder Module input.
- **Prepared Observation** is the per-step result containing both replay-buffer observation and agent-ready observation.

See `CONTEXT.md` for glossary definitions.

## Observation Form Contract

The `ObservationFormContract` is stored in the `EncoderInputContract`.

- `ObservationField`: the shared metadata type for one observation field. It carries at least shape and dtype, with observation-specific metadata such as normalization when needed.
- `env_observation`: the raw environment observation form accepted by `prepare_env_step()`.
  - Example: CNN `{"image": (3, 64, 64), "is_first": ()}`, VGGT/hybrid `{"image": (3, 518, 518), "is_first": ()}`.
  - This drives environment render resolution and validates what `prepare_env_step()` accepts.
- `replay_observation`: the replay observation fields, shapes, dtypes, and normalization metadata.
  - Example: hybrid stores `{"image": (3, 64, 64), "wp_cp": (4116,)}`.
  - This drives replay buffer allocation and dtype/normalization metadata.
- `agent_observation`: the agent-ready observation form from `PreparedObservation.agent_obs`.
  - Example: often same as replay, but hybrid one-step acting is still a structured dict.
  - This validates `PreparedObservation.agent_obs`.
- `encoder_input`: the tensor form the Encoder Module consumes.
  - Example: hybrid packed tensor `(16404,)`, VGGT WP/CP `(4116,)`, CNN `(3, 64, 64)`.
  - This is the shape the Encoder Module actually consumes.
- `decoder_target`: the decoder target form, or `None` if no decoder target is available.
  - Example: `(3, 64, 64)` if RGB reconstruction is available, else `None`.
  - This pairs with `provides_decoder_target`.

Derived dimensions like `4116`, `16404`, `3072`, and `1402880` should be named constants or factory-derived values, not repeated in config, adapter, world model, and tests. The contract should carry the resolved values, but the formulas should live in one shape-definition module inside `observation_preparation`.

## Decisions

1. Create a new package for the boundary: `src/r2dreamer/observation_preparation/`.
2. Treat Observation Preparation as the primary deepened concept. The Encoder Input Contract lives inside that module.
3. Observation Preparation owns the encoder type and contract language for an input mode.
4. Observation Preparation prepares replay-buffer observations and agent-ready observations, but it does not own replay-buffer storage.
5. The trainer remains responsible for deciding when and how to write replay-buffer observations into replay.
6. The per-step API should return both outputs explicitly, even when they are identical:

   ```python
   PreparedObservation(
       replay_obs=...,
       agent_obs=...,
   )
   ```
7. Observation Preparation owns sampled replay conversion into Encoder Module tensors.
8. Observation Preparation owns decoder target extraction from sampled replay observations.
9. Public launcher language should move from `encoder` to `observation-preparation`, not only internal module names.
10. The migration should be a hard break: new code should stop accepting `--encoder` as the canonical public launcher flag. Existing run configs, manifests, checkpoints, and W&B naming must be audited and migrated deliberately rather than silently supported through a long compatibility alias.
11. Evaluation and manifest loading may keep a narrow legacy read fallback for old `encoder` manifest fields. This fallback is for checkpoint migration only; it must not keep `--encoder` alive as a public CLI flag.
12. Use the full `observation_preparation` term for durable/public surfaces such as CLI, config, manifests, and registries. Use short names such as `obs_prep` only for internal/local implementation variables.
13. Model Observation Preparation as a small class hierarchy with concrete implementations for each input mode. Keep `EncoderInputContract` as a frozen value returned by each implementation, not as the behavior-owning object.
14. Expose sampled replay conversion and decoder target extraction as explicit methods rather than hiding both behind a single replay-preparation return object.
15. Represent decoder-target availability explicitly on the `EncoderInputContract`. Calling decoder-target extraction when the contract says no decoder target is available should fail with `ValueError` rather than returning `None`.
16. Include the Encoder Module class and its construction metadata in `EncoderInputContract`, alongside the prepared tensor shape, so module selection and prepared data shape cannot drift through a separate launcher registry.
17. Store the resolved `EncoderInputContract` on `R2DreamerConfig` so the agent config remains the runtime source of truth for the prepared tensor shape and Encoder Module construction.
18. Persist a serializable snapshot of the `EncoderInputContract` in manifests/checkpoints/W&B rather than relying on stringified Python class objects. Runtime config may hold class objects; durable metadata should use stable names, shapes, and booleans.
19. Store resolved Encoder Module constructor kwargs in `EncoderInputContract`. After launcher/CLI overrides are applied, the contract is the effective source used to instantiate the Encoder Module.
20. Use `ObservationFormContract` as the canonical form vocabulary for Observation Preparation. It should distinguish raw environment observation, replay observation, agent-ready observation, Encoder Module input, and decoder target.
21. Store `ObservationFormContract` in the `EncoderInputContract`.
22. Allow `agent_observation` to describe structured agent-ready observations. Keep `encoder_input` tensor-only because it describes the actual Encoder Module input after packing.
23. Represent `env_observation` as the raw environment observation structure, including at least `image` and `is_first`, rather than only the image tensor shape.
24. Use `ObservationField` as the shared metadata type for individual fields across environment observation, replay observation, agent-ready observation, Encoder Module input, and decoder target.

## Open Questions

- The final public API names for env-step preparation, replay-batch packing, and decoder target extraction.
- The exact field metadata representation on `ObservationFormContract`.
- How to migrate existing run configs, scripts, W&B naming, and evaluation workflows across the public launcher rename.
