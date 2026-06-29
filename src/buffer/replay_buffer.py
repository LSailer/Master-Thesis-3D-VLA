"""Replay buffer that stores complete transition objects in RAM ring slots.

The public interface is batch-first: ``ReplayBuffer.add`` accepts one
``ReplayTransition`` and ``sample`` returns a ``ReplayBatch`` whose leaves have
``(batch, time)`` leading axes. Internally, the replay ring stores copied
transitions directly instead of parallel arrays for observations, actions,
rewards, and flags.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.typing import DTypeLike

from src.environments.observation import ObservationFrame

ReplayObservationBatch: TypeAlias = jax.Array | dict[str, jax.Array]
ObservationLeaf: TypeAlias = jnp.ndarray | np.ndarray


@dataclass
class HybridObservation:
    """Hybrid image plus compact world-point observation for one transition.

    Args:
        image: Image observation for one environment step.
        wp_cp: Compact world-point / camera-pose feature for one environment
            step.
    """

    image: ObservationLeaf
    wp_cp: ObservationLeaf


ObservationInput: TypeAlias = (
    ObservationLeaf | HybridObservation | Mapping[str, ObservationLeaf]
)


@struct.dataclass
class ReplayTransition:
    """One transition accepted by the replay buffer.

    Args:
        obs: Replay observation for one stored frame.
        action: Discrete action that produced ``obs``.
        reward: Reward returned with ``obs``.
        is_first: Whether this transition should reset sequence state. Sampled
            batches also mark the first transition in every sampled window as
            ``True``.
        is_episode_end: Whether this transition ended an episode.
    """

    obs: ObservationInput
    action: int | np.integer
    reward: float | np.floating
    is_first: bool | np.bool_
    is_episode_end: bool | np.bool_

    @classmethod
    def from_frame(
        cls,
        obs: ObservationInput,
        frame: ObservationFrame,
    ) -> "ReplayTransition":
        """Build one transition from prepared replay observation and env frame.

        Args:
            obs: Replay observation prepared for ``frame``.
            frame: Environment observation frame returned by ``env.step``.

        Returns:
            A replay transition whose scalar fields come from ``frame``.

        Raises:
            ValueError: If ``frame`` is a reset frame without a previous action.
        """
        if frame.previous_action is None:
            raise ValueError("cannot build ReplayTransition from reset frame")
        return cls(
            obs=obs,
            action=frame.previous_action,
            reward=frame.reward,
            is_first=frame.is_first,
            is_episode_end=frame.is_episode_end,
        )


@struct.dataclass
class ReplayBatch:
    """Sampled replay batch with ``(batch, time)`` leading axes.

    Args:
        obs: Observation leaves stacked to ``(B, T, *obs_shape)``. Structured
            observations are stored as ``dict[field, array]`` with the same
            ``(B, T)`` prefix on every field.
        actions: Actions that produced the stored observations, one-hot encoded
            with the configured floating dtype and shape ``(B, T, num_actions)``.
        rewards: Rewards returned with the stored observations as configured
            floating arrays of shape ``(B, T)``.
        is_first: RSSM reset mask as configured floating arrays of shape
            ``(B, T)``. This is true at each sampled sequence start and after
            episode ends.
        is_episode_end: Episode-end flags returned with the stored observations
            as configured floating arrays of shape ``(B, T)``.
    """

    obs: ReplayObservationBatch
    actions: jax.Array
    rewards: jax.Array
    is_first: jax.Array
    is_episode_end: jax.Array

    def __getitem__(self, key: str) -> Any:
        """Return a field by legacy mapping key.

        Args:
            key: One of ``obs``, ``actions``, ``rewards``, ``is_first``, or
                ``is_episode_end``.

        Returns:
            The corresponding replay-batch field.

        Raises:
            KeyError: If ``key`` is not a replay-batch field.
        """
        if key == "obs":
            value = self.obs
        elif key == "actions":
            value = self.actions
        elif key == "rewards":
            value = self.rewards
        elif key == "is_first":
            value = self.is_first
        elif key == "is_episode_end":
            value = self.is_episode_end
        else:
            raise KeyError(key)
        return value


ReplayTransitionBatch: TypeAlias = list[list[ReplayTransition]]


class ReplayBuffer:
    """Lazy ring buffer for complete ``ReplayTransition`` objects.

    Args:
        capacity: Maximum number of transitions to store.
        num_actions: Number of discrete actions for sampled one-hot batches.
        float_dtype: Floating dtype for sampled action, reward, and mask arrays.

    Internally, the ring slots contain copied ``ReplayTransition`` objects only;
    there are no parallel arrays for transition attributes.
    """

    def __init__(
        self,
        capacity: int,
        num_actions: int,
        *,
        float_dtype: DTypeLike = jnp.float32,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}")
        self.capacity = capacity
        self.num_actions = num_actions
        self.float_dtype = float_dtype
        self.transitions: list[ReplayTransition | None] = [None] * capacity
        self.idx = 0
        self.size = 0

    def add(self, replay_transition: ReplayTransition) -> None:
        """Append one replay transition.

        Args:
            replay_transition: Transition to copy into the current ring slot.
        """
        self.transitions[self.idx] = deepcopy(replay_transition)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch:
        """Sample fixed-length frame-aligned replay batches.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Number of consecutive transitions per window.

        Returns:
            A ``ReplayBatch`` with ``(batch_size, seq_len)`` leading axes.
        """
        sequences = self.sample_transitions(batch_size=batch_size, seq_len=seq_len)
        batch = self._pack_replay_batch(sequences)
        return batch

    def sample_transitions(
        self,
        batch_size: int,
        seq_len: int,
    ) -> ReplayTransitionBatch:
        """Sample copied transition windows for transition-level checks.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Number of consecutive transitions per window.

        Returns:
            A list with ``batch_size`` sampled transition sequences. Each
            sequence contains ``seq_len`` copied transitions.
        """
        self._validate_sample_request(batch_size, seq_len)
        n_valid = self.size - seq_len + 1
        starts = np.random.randint(0, n_valid, size=batch_size)
        sequences = [self._gather_sequence(int(start), seq_len) for start in starts]
        return sequences

    def _validate_sample_request(self, batch_size: int, seq_len: int) -> None:
        """Validate common sampling arguments before drawing windows."""
        if self.size == 0:
            raise RuntimeError("cannot sample before adding at least one transition")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if self.size < seq_len:
            raise ValueError(
                f"not enough data in replay buffer: size={self.size}, seq_len={seq_len}"
            )

    def _gather_sequence(self, start: int, seq_len: int) -> list[ReplayTransition]:
        """Copy one sampled sequence from chronological ring-buffer positions."""
        oldest_index = 0 if self.size < self.capacity else self.idx
        sequence: list[ReplayTransition] = []
        for offset in range(seq_len):
            ring_index = (oldest_index + start + offset) % self.capacity
            transition = deepcopy(self._transition_at(ring_index))
            if offset == 0:
                transition = transition.replace(is_first=True)
            sequence.append(transition)
        return sequence

    def _pack_replay_batch(
        self,
        sequences: Sequence[Sequence[ReplayTransition]],
    ) -> ReplayBatch:
        """Pack sampled transition windows into a model-ready replay batch."""
        first_obs = sequences[0][0].obs
        if isinstance(first_obs, HybridObservation):
            obs = self._stack_hybrid_observations(sequences)
        elif isinstance(first_obs, Mapping):
            obs = self._stack_mapping_observations(sequences, first_obs)
        else:
            obs = self._stack_array_observations(sequences)

        action_ids, rewards, episode_ends, is_first = self._stack_transition_labels(
            sequences
        )
        actions = jax.nn.one_hot(action_ids, self.num_actions, dtype=self.float_dtype)
        batch = ReplayBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            is_first=is_first,
            is_episode_end=episode_ends,
        )
        return batch

    def _stack_transition_labels(
        self,
        sequences: Sequence[Sequence[ReplayTransition]],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Stack frame-aligned action, reward, end, and reset labels."""
        rows: dict[str, list[list[int | float]]] = {
            "actions": [],
            "rewards": [],
            "episode_ends": [],
            "is_first": [],
        }

        for sequence in sequences:
            row: dict[str, list[int | float]] = {
                "actions": [],
                "rewards": [],
                "episode_ends": [],
                "is_first": [],
            }
            previous_episode_end = False

            for offset, transition in enumerate(sequence):
                episode_end = bool(transition.is_episode_end)
                reset = offset == 0 or bool(transition.is_first) or previous_episode_end
                row["actions"].append(int(transition.action))
                row["rewards"].append(float(transition.reward))
                row["episode_ends"].append(float(episode_end))
                row["is_first"].append(float(reset))
                previous_episode_end = episode_end

            for key, values in row.items():
                rows[key].append(values)

        arrays = (
            jnp.asarray(rows["actions"], dtype=jnp.int32),
            jnp.asarray(rows["rewards"], dtype=self.float_dtype),
            jnp.asarray(rows["episode_ends"], dtype=self.float_dtype),
            jnp.asarray(rows["is_first"], dtype=self.float_dtype),
        )
        return arrays

    def _stack_array_observations(
        self,
        sequences: Sequence[Sequence[ReplayTransition]],
    ) -> jax.Array:
        """Stack array observations with shape ``(B, T, *obs_shape)``."""
        for sequence in sequences:
            for transition in sequence:
                if isinstance(transition.obs, (Mapping, HybridObservation)):
                    raise TypeError("cannot mix structured and array observations")

        stacked_sequences = [
            jnp.stack([jnp.asarray(transition.obs) for transition in sequence])
            for sequence in sequences
        ]
        stacked_batch = jnp.stack(stacked_sequences)
        return stacked_batch

    def _stack_hybrid_observations(
        self,
        sequences: Sequence[Sequence[ReplayTransition]],
    ) -> dict[str, jax.Array]:
        """Stack legacy hybrid observations into explicit replay fields."""
        images: list[list[ObservationLeaf]] = []
        wp_cp_values: list[list[ObservationLeaf]] = []
        for sequence in sequences:
            image_sequence: list[ObservationLeaf] = []
            wp_cp_sequence: list[ObservationLeaf] = []
            for transition in sequence:
                if not isinstance(transition.obs, HybridObservation):
                    raise TypeError("cannot mix hybrid and non-hybrid observations")
                image_sequence.append(transition.obs.image)
                wp_cp_sequence.append(transition.obs.wp_cp)
            images.append(image_sequence)
            wp_cp_values.append(wp_cp_sequence)

        obs = {
            "image": self._stack_array_grid(images),
            "wp_cp": self._stack_array_grid(wp_cp_values),
        }
        return obs

    def _stack_mapping_observations(
        self,
        sequences: Sequence[Sequence[ReplayTransition]],
        first_obs: Mapping[str, ObservationLeaf],
    ) -> dict[str, jax.Array]:
        """Stack mapping observations while enforcing stable keys."""
        keys = tuple(first_obs.keys())
        expected_keys = set(keys)
        for sequence in sequences:
            for transition in sequence:
                if not isinstance(transition.obs, Mapping):
                    raise TypeError("cannot mix mapping and non-mapping observations")
                if set(transition.obs.keys()) != expected_keys:
                    raise KeyError(
                        "replay observation keys changed inside sampled batch: "
                        f"expected={sorted(expected_keys)}, "
                        f"got={sorted(transition.obs.keys())}"
                    )

        stacked_fields: dict[str, jax.Array] = {}
        for key in keys:
            stacked_sequences = []
            for sequence in sequences:
                stacked_steps = []
                for transition in sequence:
                    obs_mapping = cast(Mapping[str, ObservationLeaf], transition.obs)
                    stacked_steps.append(jnp.asarray(obs_mapping[key]))
                stacked_sequences.append(jnp.stack(stacked_steps))
            stacked_fields[key] = jnp.stack(stacked_sequences)

        return stacked_fields

    def _stack_array_grid(self, values: list[list[ObservationLeaf]]) -> jax.Array:
        """Stack a ``(B, T)`` grid of observation leaves."""
        stacked_sequences = [
            jnp.stack([jnp.asarray(value) for value in sequence])
            for sequence in values
        ]
        stacked_batch = jnp.stack(stacked_sequences)
        return stacked_batch

    def _transition_at(self, index: int) -> ReplayTransition:
        """Return the stored transition at a valid physical ring index."""
        transition = self.transitions[index]
        if transition is None:
            raise RuntimeError(f"ring slot {index} does not contain a transition")
        return transition
