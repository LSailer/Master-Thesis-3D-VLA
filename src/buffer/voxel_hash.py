"""Pure JAX voxel-hash kernel for exact-dedup point accumulation.

This module holds the stateless, JIT-heavy algorithm used to fold a stream of
``(P, 3)`` XYZ/RGB point frames into a fixed-shape, exactly-deduplicated voxel
occupancy table with open addressing. It has no knowledge of scenes, dtypes
policy, or file I/O — see ``src.buffer.house_context_pose_buffer`` for the
stateful wrapper that adds that bookkeeping.

The functional API is:
    - ``VoxelContextConfig``: static sizing/threshold config for one table.
    - ``VoxelContextState``: fixed-shape device state (the "table").
    - ``empty_state``: build a zeroed ``VoxelContextState``.
    - ``add_frame_to_state``: fold one flattened frame into a state (jitted).
    - ``house_context_snapshot``: read back a fixed-size XYZRGB snapshot
      (jitted).
    - ``is_power_of_two``: host-side helper for validating ``hash_table_size``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class VoxelContextConfig:
    """Static JAX config for the fixed-shape voxel context state.

    Attributes:
        voxel_size_m: Voxel edge length in metres; XYZ points are floored by
            this size to derive the int32 voxel key used for deduplication.
        confidence_score: Minimum per-point confidence required to admit a
            point (points below it, or non-finite, are dropped).
        hash_table_size: Power-of-two number of slots in the open-addressed
            occupancy table (must be >= ``capacity``).
        capacity: Maximum number of representative voxels stored; novel voxels
            beyond it are counted as overflow rather than stored.
        max_probe_count: Maximum linear-probing rounds per frame before a key
            is abandoned and counted as a failed insert.
    """

    voxel_size_m: float
    confidence_score: float
    hash_table_size: int
    capacity: int
    max_probe_count: int


class VoxelContextState(NamedTuple):
    """Fixed-shape device state for exact voxel occupancy and point storage."""

    key_xyz: jax.Array  # (hash_table_size, 3) int32 voxel keys
    occupied: jax.Array  # (hash_table_size,) bool
    store_xyz: jax.Array  # (capacity, 3) bfloat16 representative points
    store_rgb: jax.Array  # (capacity, 3) uint8 representative colours
    size: jax.Array  # () int32 logical stored row count, capped at capacity
    overflow_count: jax.Array  # () int32 new voxels dropped after capacity fills
    failed_insert_count: jax.Array  # () int32 hash-table insert failures


class _UniqueFrameVoxels(NamedTuple):
    """Static-length unique voxel representatives for one flattened frame."""

    xyz: jax.Array  # (P, 3) float32 representative points, sorted by voxel key
    rgb: jax.Array  # (P, 3) uint8 representative colours
    key_xyz: jax.Array  # (P, 3) int32 voxel keys
    active: jax.Array  # (P,) bool, true only for valid first representatives
    first_slot: jax.Array  # (P,) int32 first hash-table slot per key


class _ProbeLoopState(NamedTuple):
    """Carry for vectorized open-addressing probe rounds."""

    probe_index: jax.Array  # () int32
    voxel_state: VoxelContextState
    active: jax.Array  # (P,) bool keys not inserted/found yet


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two.

    Args:
        value: Integer to test.

    Returns:
        True if ``value`` is a positive power of two.
    """
    return value > 0 and value & (value - 1) == 0


def empty_state(hash_table_size: int, capacity: int) -> VoxelContextState:
    """Create an empty fixed-shape voxel context state on the default device.

    Args:
        hash_table_size: Power-of-two number of slots in the occupancy table.
        capacity: Maximum number of representative voxels stored.

    Returns:
        A zeroed ``VoxelContextState`` sized for ``hash_table_size``/``capacity``.
    """
    return VoxelContextState(
        key_xyz=jnp.zeros((hash_table_size, 3), dtype=jnp.int32),
        occupied=jnp.zeros((hash_table_size,), dtype=jnp.bool_),
        store_xyz=jnp.zeros((capacity, 3), dtype=jnp.bfloat16),
        store_rgb=jnp.zeros((capacity, 3), dtype=jnp.uint8),
        size=jnp.asarray(0, dtype=jnp.int32),
        overflow_count=jnp.asarray(0, dtype=jnp.int32),
        failed_insert_count=jnp.asarray(0, dtype=jnp.int32),
    )


def _hash_voxel_keys(keys_xyz: jax.Array, hash_table_size: int) -> jax.Array:
    """Hash int32 ``(..., 3)`` voxel keys into a power-of-two table."""
    keys_u32 = keys_xyz.astype(jnp.uint32)
    hashed = (
        keys_u32[..., 0] * jnp.uint32(73_856_093)
        ^ keys_u32[..., 1] * jnp.uint32(19_349_663)
        ^ keys_u32[..., 2] * jnp.uint32(83_492_791)
    )
    return (hashed & jnp.uint32(hash_table_size - 1)).astype(jnp.int32)


def _quantize_points(flat_xyz: jax.Array, voxel_size_m: float) -> tuple[jax.Array, jax.Array]:
    """Return finite mask and int32 voxel keys for ``(P, 3)`` XYZ points."""
    finite_xyz = jnp.isfinite(flat_xyz).all(axis=1)
    safe_xyz = jnp.where(jnp.isfinite(flat_xyz), flat_xyz, jnp.float32(0.0))
    voxel_keys = jnp.floor(safe_xyz / voxel_size_m).astype(jnp.int32)
    return finite_xyz, voxel_keys


def _unique_frame_voxels(
    flat_xyz: jax.Array,
    flat_rgb: jax.Array,
    valid: jax.Array,
    voxel_keys: jax.Array,
    hash_table_size: int,
) -> _UniqueFrameVoxels:
    """Sort rows by voxel key and keep the first valid input row per voxel."""
    invalid_key = jnp.iinfo(jnp.int32).max
    sort_keys = jnp.where(valid[:, None], voxel_keys, invalid_key)
    # lexsort is stable, so equal keys keep input order and the first valid
    # input row per voxel wins without an explicit tie-break key.
    sort_order = jnp.lexsort((sort_keys[:, 2], sort_keys[:, 1], sort_keys[:, 0]))
    sorted_keys = voxel_keys[sort_order]
    sorted_valid = valid[sort_order]
    same_as_previous = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.bool_),
            jnp.all(sorted_keys[1:] == sorted_keys[:-1], axis=1),
        ]
    )
    active = sorted_valid & ~same_as_previous
    return _UniqueFrameVoxels(
        xyz=flat_xyz[sort_order],
        rgb=flat_rgb[sort_order],
        key_xyz=sorted_keys,
        active=active,
        first_slot=_hash_voxel_keys(sorted_keys, hash_table_size),
    )


def _store_winning_voxels(
    state: VoxelContextState,
    frame: _UniqueFrameVoxels,
    slots: jax.Array,
    wins: jax.Array,
    config: VoxelContextConfig,
) -> VoxelContextState:
    """Commit this probe round's winning new voxels to table and store."""
    table_slots = jnp.where(wins, slots, config.hash_table_size)
    offsets = jnp.cumsum(wins.astype(jnp.int32)) - jnp.int32(1)
    destinations = state.size + offsets
    can_store = wins & (destinations < config.capacity)
    store_slots = jnp.where(can_store, destinations, config.capacity)
    return VoxelContextState(
        key_xyz=state.key_xyz.at[table_slots].set(frame.key_xyz, mode="drop"),
        occupied=state.occupied.at[table_slots].set(True, mode="drop"),
        store_xyz=state.store_xyz.at[store_slots].set(
            frame.xyz.astype(jnp.bfloat16), mode="drop"
        ),
        store_rgb=state.store_rgb.at[store_slots].set(
            frame.rgb.astype(jnp.uint8), mode="drop"
        ),
        size=jnp.minimum(state.size + jnp.sum(wins), config.capacity).astype(jnp.int32),
        overflow_count=state.overflow_count + jnp.sum(wins & ~can_store),
        failed_insert_count=state.failed_insert_count,
    )


def _probe_round(
    loop_state: _ProbeLoopState,
    frame: _UniqueFrameVoxels,
    config: VoxelContextConfig,
) -> _ProbeLoopState:
    """Run one vectorized linear-probing round for all active frame keys."""
    table_mask = jnp.int32(config.hash_table_size - 1)
    slots = (frame.first_slot + loop_state.probe_index) & table_mask
    slot_occupied = loop_state.voxel_state.occupied[slots]
    same_key = (
        loop_state.active
        & slot_occupied
        & jnp.all(loop_state.voxel_state.key_xyz[slots] == frame.key_xyz, axis=1)
    )
    empty_candidate = loop_state.active & ~slot_occupied
    inactive_order = jnp.int32(frame.active.shape[0])
    contender_order = jnp.where(
        empty_candidate,
        jnp.arange(frame.active.shape[0], dtype=jnp.int32),
        inactive_order,
    )
    winner_by_slot = jnp.full(
        (config.hash_table_size,), inactive_order, dtype=jnp.int32
    ).at[slots].min(contender_order)
    wins = empty_candidate & (winner_by_slot[slots] == contender_order)
    voxel_state = _store_winning_voxels(
        loop_state.voxel_state,
        frame,
        slots,
        wins,
        config,
    )
    return _ProbeLoopState(
        probe_index=loop_state.probe_index + jnp.int32(1),
        voxel_state=voxel_state,
        active=loop_state.active & ~(same_key | wins),
    )


def _insert_unique_voxels(
    state: VoxelContextState,
    frame: _UniqueFrameVoxels,
    config: VoxelContextConfig,
) -> VoxelContextState:
    """Insert sorted unique frame voxels with bounded vectorized probing."""
    initial = _ProbeLoopState(
        probe_index=jnp.asarray(0, dtype=jnp.int32),
        voxel_state=state,
        active=frame.active,
    )

    def should_probe(loop_state: _ProbeLoopState) -> jax.Array:
        return (loop_state.probe_index < config.max_probe_count) & jnp.any(
            loop_state.active
        )

    def probe_once(loop_state: _ProbeLoopState) -> _ProbeLoopState:
        return _probe_round(loop_state, frame, config)

    result = jax.lax.while_loop(should_probe, probe_once, initial)
    return result.voxel_state._replace(
        failed_insert_count=result.voxel_state.failed_insert_count
        + jnp.sum(result.active.astype(jnp.int32))
    )


@functools.partial(jax.jit, static_argnums=(4,), donate_argnums=(0,))
def add_frame_to_state(
    state: VoxelContextState,
    flat_xyz: jax.Array,
    flat_rgb: jax.Array,
    confidence_flat: jax.Array,
    config: VoxelContextConfig,
) -> VoxelContextState:
    """Add one fixed-shape frame to the voxel context state.

    Per-frame representatives are selected by a static lexicographic sort of
    int32 voxel keys. Cross-frame novelty is resolved by exact key comparison in
    a vectorized open-addressed table; hash collisions only add probe rounds.

    Args:
        state: Current fixed-shape voxel context state.
        flat_xyz: ``(P, 3)`` XYZ points for one flattened frame.
        flat_rgb: ``(P, 3)`` uint8 RGB colours aligned with ``flat_xyz``.
        confidence_flat: ``(P,)`` per-point confidence scores.
        config: Static sizing/threshold config (voxel size, capacity, etc).

    Returns:
        The updated ``VoxelContextState`` after inserting this frame's novel
        voxels.
    """
    flat_xyz = jnp.asarray(flat_xyz, dtype=jnp.float32)
    flat_rgb = jnp.asarray(flat_rgb, dtype=jnp.uint8)
    confidence_flat = jnp.asarray(confidence_flat, dtype=jnp.float32)
    finite_xyz, voxel_keys = _quantize_points(flat_xyz, config.voxel_size_m)
    valid = (
        finite_xyz
        & jnp.isfinite(confidence_flat)
        & (confidence_flat >= config.confidence_score)
    )
    frame = _unique_frame_voxels(
        flat_xyz,
        flat_rgb,
        valid,
        voxel_keys,
        config.hash_table_size,
    )
    return _insert_unique_voxels(state, frame, config)


@functools.partial(jax.jit, static_argnums=(1, 2))
def house_context_snapshot(
    state: VoxelContextState,
    max_points: int,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(max_points, 6)`` XYZRGB rows in ``dtype`` plus the valid count.

    Rows ``[0, count)`` carry stored points and rows beyond are zeros, so
    consumers can mask padding exactly (masked pooling in the encoder). While
    more voxels are stored than ``max_points``, an even stride subsamples them
    and ``count == max_points``; below that the stored prefix is zero-padded.

    Args:
        state: Voxel context state to snapshot.
        max_points: Fixed number of output rows.
        dtype: Output dtype for the XYZRGB rows.

    Returns:
        A tuple of the ``(max_points, 6)`` snapshot array and the ``()`` int32
        valid row count.
    """
    safe_size = jnp.maximum(state.size, jnp.int32(1))
    # int32 ``arange * size`` overflows once size exceeds 2**31 / max_points
    # (~524k stored voxels at max_points=4096). float32 keeps the stride math
    # exact enough: size <= capacity <= 2**24 is exactly representable, the
    # per-index error stays below one row, and floor preserves monotonicity.
    stride_ratio = safe_size.astype(jnp.float32) / jnp.float32(max_points)
    strided = jnp.floor(
        jnp.arange(max_points, dtype=jnp.float32) * stride_ratio
    ).astype(jnp.int32)
    rows = jnp.arange(max_points, dtype=jnp.int32)
    indices = jnp.where(state.size > max_points, strided, rows)
    indices = jnp.clip(indices, jnp.int32(0), safe_size - jnp.int32(1))
    xyz = state.store_xyz[indices].astype(dtype)
    rgb = state.store_rgb[indices].astype(dtype) / jnp.asarray(255.0, dtype)
    snapshot = jnp.concatenate([xyz, rgb], axis=1)
    count = jnp.minimum(state.size, jnp.int32(max_points))
    return jnp.where(rows[:, None] < count, snapshot, jnp.asarray(0.0, dtype)), count
