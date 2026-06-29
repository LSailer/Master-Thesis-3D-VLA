"""Replay buffer that stores complete transition objects in RAM ring slots.

The public interface is transition-first: ``ReplayBuffer.add`` accepts one
``ReplayTransition`` and ``sample`` returns lists of ``ReplayTransition`` objects.
The replay ring stores copied transitions directly instead of parallel arrays for
observations, actions, rewards, and flags.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeAlias

import jax.numpy as jnp  # noqa: E402
import numpy as np
from flax import struct

from src.environments.observation import ObservationFrame

ReplayBatch = list[list["ReplayTransition"]]


@dataclass
class HybridObservation:
    """Hybrid image plus compact world-point observation for one transition.

    Args:
        image: Image observation for one environment step.
        wp_cp: Compact world-point / camera-pose feature for one environment
            step.
    """

    image: jnp.ndarray
    wp_cp: jnp.ndarray


ObservationLeaf: TypeAlias = jnp.ndarray | np.ndarray
ObservationInput: TypeAlias = (
    ObservationLeaf | HybridObservation | Mapping[str, ObservationLeaf]
)


@struct.dataclass
class ReplayTransition:
    """One transition accepted by and returned from the replay buffer.

    Args:
        obs: Typed observation dataclass for one environment step.
        action: Discrete scalar action stored for this transition.
        reward: Scalar reward stored for this transition.
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
            obs: Replay observation prepared for this frame.
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


class ReplayBuffer:
    """Lazy ring buffer for complete ``ReplayTransition`` objects.

    Args:
        capacity: Maximum number of transitions to store.

    Internally, the ring slots contain copied ``ReplayTransition`` objects only;
    there are no parallel arrays for transition attributes.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.transitions: list[ReplayTransition | None] = [None] * capacity
        self.idx = 0
        self.size = 0

    def add(self, replay_transition: ReplayTransition) -> None:
        """Append one replay transition.

        Args:
            replay_transition: Transition to copy into the current ring slot.
        """
        # Decision: Observation schema validation is intentionally omitted for now;
        # revisit this before packed array batches require stable shapes and dtypes.
        self.transitions[self.idx] = deepcopy(replay_transition)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch:
        """Sample fixed-length transition sequences from valid ring positions.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Number of consecutive transitions per window.

        Returns:
            A list with ``batch_size`` sampled sequences. Each sequence is a
            list of ``seq_len`` copied ``ReplayTransition`` objects.
        """
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
        n_valid = self.size - seq_len + 1
        starts = np.random.randint(0, n_valid, size=batch_size)
        return [self._gather_sequence(start, seq_len) for start in starts]

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

    def _transition_at(self, index: int) -> ReplayTransition:
        """Return the stored transition at a valid physical ring index."""
        transition = self.transitions[index]
        if transition is None:
            raise RuntimeError(f"ring slot {index} does not contain a transition")
        return transition
