# ADR 0007: Let the adapter declare its routing instead of dispatching on a config string

Status: implemented 2026-07-26 (PR #212)
Date: 2026-07-26
Supersedes: ADR 0001, "Use Observation Preparation as the public input-mode
boundary" (added 2026-06-15 in `03b36a3`, deleted here). Its problem statement
and its failed remedy are both carried forward in the Context below, so the file
itself held nothing but a pointer saying to ignore it.

## Context

ADR 0001 named the problem correctly: `encoder` was overloaded across launcher
choices, replay preparation, VGGT extraction and the Dreamer-side Flax module,
and behaviour was leaking through string conditionals. Its remedy was a
vocabulary split - **Observation Preparation** as the public boundary,
**Encoder Module** for the Flax side.

The vocabulary held. The dispatch did not. By mid-2026 a single
`cfg.encoder_type` string was resolved through six parallel tables, each
maintained by hand:

1. which encoder class to build,
2. which observation keys to write,
3. what schema the replay buffer preallocates,
4. what resolution the environment renders at,
5. what keyword arguments the VGGT extractor is constructed with,
6. which key the decoder probe reconstructs.

With 17 encoder-type values that is 102 cells that must agree. Nothing checked
that they did. Adding a variant meant editing six files; forgetting one gave a
silent misconfiguration rather than an error - a run that trains on the wrong
observation, or renders 518x518 and discards 98 percent of the pixels.

Renaming the boundary could not fix this, because the defect was not the name.
It was that one string had to be interpreted in six places.

## Decision

The adapter declares its own routing. Each adapter returns a list of

```python
AdapterField(key, encoder, buffer, value, decoder_target)
```

and that single declaration drives replay, encoder composition and the decoder
probe. `Encoder` is an opaque enum stored in the buffer schema; `buffer=False`
marks the one live field encoded per batch and broadcast over `(B, T)`;
`decoder_target=True` marks the field the reconstruction probe reads.

Everything else a run needs is a class constant on the adapter -
`RENDER_RESOLUTION`, `NEEDS_FEATURES`, `EXTRACTOR_KWARGS`, `ENCODER_OVERRIDES`,
and the optional `RUN_FLAGS` a variant claims from the CLI. The registry in
`src/adapters/__init__.py` maps a speaking name to a class and holds nothing
else.

Consequences of the shape, chosen deliberately:

- A variant differing only in a constant is a **subclass overriding that
  constant**, not a parametrized registry row, so the registry stays a flat
  list of names.
- The fusion `Dense` is applied exactly when a variant has more than one
  branch, so `fusion_dim` is implicit and a single-branch variant is its
  branch, unfused.
- A CLI flag a variant does not claim makes the run **refuse to start** rather
  than being ignored, because a diagnostic that silently never runs costs a
  cluster job to discover.

## Consequences

`src/r2dreamer/adapters/`, `src/r2dreamer/observation_preparation/`,
`encoder_types.py` and `observation_keys.py` are deleted; `src/main.py` becomes
a composition root; `src/r2dreamer/encoders/` holds only basic encoders.
17 encoder-type strings became 11 registry rows. Overall `src/` went from
20 543 to 15 499 lines.

ADR 0001's public term `--observation-preparation` is gone. The public knob is
`--adapter <registry key>`, and the key names what is observed: an `rgb_`
prefix means appearance is part of the observation, its absence means the arm
is deliberately appearance-blind.

The goal of ADR 0001 is met, by a different mechanism: string-based encoder
conditionals cannot leak behaviour because there is no string to condition on.

What this does not solve: the routing carried through replay
(`ReplayTransition.encoders`) has no consumer beyond the routing-changed guard
in `ReplayBuffer.add`. It is kept deliberately - that guard refuses to mix
incompatible rows when an adapter's routing changes mid-run, a failure mode
this ADR makes reachable for the first time.

## Notes

The migration surfaced five pre-existing defects that a string table had hidden,
including a `full_bf16` gate that had been silently dead since the config field
was dropped while its `getattr(cfg, "full_bf16", False)` reader stayed. That is
the recurring shape this ADR is meant to prevent: a name that resolves to
nothing produces no error, only an absence.
