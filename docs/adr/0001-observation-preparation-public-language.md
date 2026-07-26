# Use Observation Preparation as the public input-mode boundary

Status: superseded 2026-07-26 by
[ADR 0007](0007-adapter-declared-routing-contract.md). The problem named below
is real and unchanged; the remedy is not. Splitting the vocabulary did not stop
one config string from being interpreted in six places, so the routing now
lives on the adapter and there is no string left to condition on. The text
below is kept as the historical decision, not as current guidance.

The existing public and internal `encoder` language is overloaded: it refers to launcher choices, replay-buffer observation preparation, VGGT feature extraction, and the Dreamer-side Flax module that maps prepared tensors into embeddings. We will use **Observation Preparation** as the public input-mode boundary and reserve **Encoder Module** for the Dreamer-side Flax module, even though this creates a deliberate public break from `--encoder` to `--observation-preparation`. This makes the data-flow contract explicit and prevents future VGGT, hybrid, replay, and decoder-target behavior from leaking through string-based encoder conditionals.
