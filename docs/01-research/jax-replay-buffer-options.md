# JAX-only replay/ring-buffer options

Date: 2026-06-22

Goal: remove NumPy from the replay path where possible, ideally avoiding
`JAX -> NumPy -> RAM -> NumPy -> JAX` and moving toward `JAX -> RAM -> JAX`.

## Options

| Option | Mechanism | Pros | Cons | Fit |
|---|---|---|---|---|
| Pure functional JAX buffer | Keep replay arrays in a pytree state; update with `.at[idx].set(...)` or `jax.lax.dynamic_update_slice`; return a new state. | Idiomatic JAX, works under `jit`, used by Brax/Flashbax-style buffers. | Awkward with Python env loop because buffer state must be threaded everywhere; large updates may copy unless compiled well. | Good if trainer loop becomes more JAX-functional. |
| `jax.Ref` mutable buffer | Allocate mutable arrays with `jax.new_ref`; insert with `ref[idx] = value`; sample with indexed reads. | Closest to current mutable ring-buffer API; supports in-place writes. | JAX docs warn impure `jit` dispatch with `Ref` inputs is slower; local GPU smoke showed sampled CPU-Ref batches made tiny `train_step` much slower. | Good experiment only, not production yet. |
| Flashbax | Use existing JAX replay-buffer library. | Mature JAX-native replay buffers: flat, trajectory, prioritized variants. | New dependency; may need adaptation for this repo's sequence sampling, ring wrap, `is_first`, modal obs. | Worth reading before writing custom pure-JAX buffer. |
| Brax/dejax-style circular buffer | Copy/adapt small pure-JAX FIFO/circular replay design. | No dependency if copied minimally; known RL pattern. | Still requires custom sequence-window sampling and modal observation support. | Good source pattern. |
| Pinned-host JAX buffer | Store replay arrays in JAX host/pinned-host memory; explicitly `device_put` sampled batches to GPU. | Best match for `JAX -> RAM -> JAX`; may improve host-to-device transfer. | Newer/less-tested memory-placement path; must benchmark transfer and `train_step` together. | Best next experiment. |
| GPU-resident replay | Keep replay arrays on GPU/device and sample there. | Removes host-to-device batch transfer. | Capacity likely too large for Habitat/VGGT replay; competes with model memory. | Only for small-buffer ablations. |

## Existing evidence

- Buffer-only GPU profiling showed JAX approaches can improve sampling in some
  cases.
- End-to-end tiny `train_step` with CPU JAX `Ref` batches was much slower than
  NumPy batches in the local experiment.
- Likely reason: NumPy host arrays enter the GPU-compiled step through a better
  host-to-device placement path than CPU-device-committed JAX arrays produced by
  `Ref` sampling.
- Therefore: do not replace `src.buffer.replay_buffer.ReplayBuffer` yet.

## Recommended next experiment

Measure transfer explicitly:

```text
NumPy replay sample -> jax.device_put(batch, gpu) -> train_step
JAX Ref CPU sample  -> jax.device_put(batch, gpu) -> train_step
Pinned-host sample  -> jax.device_put(batch, gpu) -> train_step
```

Keep the production buffer unchanged unless the full sample+transfer+train path
is faster.

## References

- JAX `Ref` mutable arrays: https://docs.jax.dev/en/latest/array_refs.html
- `jax.ref` API: https://docs.jax.dev/en/latest/jax.ref.html
- Flashbax replay buffers: https://github.com/instadeepai/flashbax
- Brax replay buffers: https://github.com/google/brax/blob/main/brax/training/replay_buffers.py
- dejax circular/replay buffers: https://github.com/hr0nix/dejax
- JAX host offloading / pinned host memory: https://docs.jax.dev/en/latest/notebooks/host-offloading.html
- JAX `device_put`: https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html
