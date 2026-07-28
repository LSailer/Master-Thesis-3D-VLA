"""Packing of sampled replay windows into training-aligned arrays.

Provides ``replay_batch_to_arrays``: turns either a struct ``ReplayBatch``
or a list-of-lists transition window (``ReplayTransitionBatch``) into a raw
array batch with ``(B, T)`` leading axes, ready for ``agent.train_step``.
Lives next to ``ReplayBuffer`` because it is replay-domain code — the
training loop consumes its output but owns none of its logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.replay_buffer import (
    HybridObservation,
    ReplayBatch,
    ReplayTransition,
    ReplayTransitionBatch,
)

ArrayObservation: TypeAlias = np.ndarray | jax.Array
ReplayObservation: TypeAlias = (
    ArrayObservation | Mapping[str, ArrayObservation] | HybridObservation
)
ReplayArrayBatch: TypeAlias = dict[str, Any]


def _stack_array_grid(values: list[list[ArrayObservation]]) -> jax.Array:
    """Stack ``(B, T)`` replay observation arrays into one JAX array."""
    return jnp.stack(
        [jnp.stack([jnp.asarray(value) for value in sequence]) for sequence in values]
    )


def _transition_observation(transition: ReplayTransition) -> ReplayObservation:
    """Return a transition observation, including structured adapter mappings."""
    return transition.obs


def _stack_hybrid_observations(
    obs_grid: list[list[ReplayObservation]],
) -> dict[str, jax.Array]:
    """Stack replay-domain hybrid observations as explicit replay fields."""
    images: list[list[ArrayObservation]] = []
    wp_cp_values: list[list[ArrayObservation]] = []
    for sequence in obs_grid:
        image_sequence: list[ArrayObservation] = []
        wp_cp_sequence: list[ArrayObservation] = []
        for obs in sequence:
            if not isinstance(obs, HybridObservation):
                raise TypeError("cannot mix hybrid and non-hybrid replay observations")
            image_sequence.append(obs.image)
            wp_cp_sequence.append(obs.wp_cp)
        images.append(image_sequence)
        wp_cp_values.append(wp_cp_sequence)
    return {
        "image": _stack_array_grid(images),
        "wp_cp": _stack_array_grid(wp_cp_values),
    }


def _stack_mapping_observations(
    obs_grid: list[list[ReplayObservation]],
    first_obs: Mapping[str, ArrayObservation],
) -> dict[str, jax.Array]:
    """Stack structured replay mappings after checking each window has same keys."""
    keys = tuple(first_obs.keys())
    expected_keys = set(keys)
    for sequence in obs_grid:
        for obs in sequence:
            if not isinstance(obs, Mapping):
                raise TypeError(
                    "cannot mix mapping and non-mapping replay observations"
                )
            if set(obs.keys()) != expected_keys:
                raise KeyError(
                    "replay observation keys changed inside sampled batch: "
                    f"expected={sorted(expected_keys)}, got={sorted(obs.keys())}"
                )
    return {
        key: _stack_array_grid(
            [
                [cast(Mapping[str, ArrayObservation], obs)[key] for obs in sequence]
                for sequence in obs_grid
            ]
        )
        for key in keys
    }


def _stack_replay_observations(
    batch: ReplayTransitionBatch,
) -> jax.Array | dict[str, jax.Array]:
    """Stack transition observations into the raw replay-batch observation form."""
    obs_grid = [
        [_transition_observation(transition) for transition in sequence]
        for sequence in batch
    ]
    first_obs = obs_grid[0][0]
    if isinstance(first_obs, HybridObservation):
        return _stack_hybrid_observations(obs_grid)
    if isinstance(first_obs, Mapping):
        return _stack_mapping_observations(obs_grid, first_obs)
    return _stack_array_grid(cast(list[list[ArrayObservation]], obs_grid))


def replay_batch_to_arrays(
    batch: ReplayBatch | ReplayTransitionBatch,
) -> ReplayArrayBatch:
    """Pack transition-object replay windows into arrays with ``(B, T)`` prefix.

    Args:
        batch: Non-empty sampled replay windows returned by ``ReplayBuffer.sample``.

    Returns:
        Raw replay arrays keyed by ``obs``, ``actions``, ``rewards``,
        ``is_first``, and ``is_episode_end``. Observation leaves preserve their
        stored dtype and have shape ``(batch_size, seq_len, *obs_shape)``.

    Raises:
        ValueError: If the batch is empty, contains zero-length sequences, or
            mixes sequence lengths.
    """
    if isinstance(batch, ReplayBatch):
        return {
            "obs": batch.obs,
            "actions": batch.actions,
            "rewards": batch.rewards,
            "is_first": batch.is_first,
            "is_episode_end": batch.is_episode_end,
        }

    if not batch:
        raise ValueError("cannot convert an empty replay batch")
    seq_len = len(batch[0])
    if seq_len == 0:
        raise ValueError("cannot convert replay sequences with length zero")
    if any(len(sequence) != seq_len for sequence in batch):
        raise ValueError("all replay sequences must have the same length")

    episode_ends = [
        [bool(transition.is_episode_end) for transition in sequence]
        for sequence in batch
    ]
    is_first = [
        [
            offset == 0
            or bool(transition.is_first)
            or (offset > 0 and episode_ends[batch_index][offset - 1])
            for offset, transition in enumerate(sequence)
        ]
        for batch_index, sequence in enumerate(batch)
    ]

    return {
        "obs": _stack_replay_observations(batch),
        "actions": jnp.asarray(
            [[int(transition.action) for transition in sequence] for sequence in batch],
            dtype=jnp.int32,
        ),
        "rewards": jnp.asarray(
            [
                [float(transition.reward) for transition in sequence]
                for sequence in batch
            ],
            dtype=jnp.float32,
        ),
        "is_first": jnp.asarray(is_first, dtype=jnp.bool_),
        "is_episode_end": jnp.asarray(episode_ends, dtype=jnp.bool_),
    }
