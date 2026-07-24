# house_context_multi_episode

## Goal

Run a multi-episode loop with a persistent house context and a simplified
adapter design, replacing the `ObsAdapter` base class with a plain function
protocol plus a routed-field struct.

## Adapter design decisions (2026-07-24)

1. **Output format: flat list of routed fields.** `transform` returns
   `list[AdapterField]` with `key: str`, `encoder: Encoder | None` (enum
   selecting which encoder branch consumes the field live: CONV, MLP,
   POINTNET, GNN; `None` = not a live encoder input),
   `buffer: bool` (store in replay), `value`. Defined in
   `adapter_contract.py` together with two adapter Protocols:
   `FrameAdapterFn` (frame only) and `FeatureAdapterFn` (frame + VGGT
   features). The adapter's signature declares which inputs it needs - no
   Optional parameter, no runtime check inside the call. When VGGT runs
   anyway, frame-only adapters are lifted once at wiring time via
   `ignore_features`; a driver that should *skip* VGGT for frame-only
   adapters instead dispatches once at construction (shape for a future
   CNN-baseline arm).
   `is_first` is not part of the transform output - the collector passes it
   explicitly when appending to the replay buffer.

2. **Shapes are inferred from the first observation**, not declared up front.
   Buffer allocation is deferred until the first frame. Known limitation:
   this only works for fixed-shape fields - the house-points field is
   variable-length (grows across steps/episodes), so it needs an explicit
   max-size + padding decision. That decision is deferred together with (3).

3. **Where the house context lives (replay buffer vs. own object) is
   deferred** until the rest of the loop works - it is the question this
   prototype exists to answer.

4. **No lifecycle hooks on the adapter** (`on_episode_reset`, `diagnostics`,
   `growth_history` are dropped). The prototype owns its episode loop, so
   VGGT cache reset / scene save-restore is called directly at the episode
   boundary in `run_multi_episode.py`. Reminder: for a multi-episode
   house-context experiment the episode boundary IS the experiment - do not
   forget to handle it explicitly.

## Open problems

- Dual representation of the image (uint8 for replay vs normalized float for
  live encoding): current struct has a single `value` slot; either accept
  recompute-on-sample or add an `encoder_value` slot later.
- Max-size + padding for the ragged house-points field (see decision 2).
