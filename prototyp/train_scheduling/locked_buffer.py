"""Thread-safe wrapper around the real ``ReplayBuffer`` (prototype).

The production ``ReplayBuffer`` (src/buffer/replay_buffer.py) is a host-NumPy
structure-of-arrays ring buffer with unguarded ``idx``/``size`` mutation in
``add()`` and an unguarded ring gather in ``sample()`` — it is not safe to use
from an actor thread and a learner thread at once. This wrapper delegates
every operation to a real buffer instance under one ``threading.Lock`` so the
threaded scheduling mode can swap it into ``ExperienceCollector.buffer``
without touching ``src/``.
"""

from __future__ import annotations

import threading
from typing import Any

from src.buffer.replay_buffer import (
    ReplayBatch,
    ReplayBuffer,
    ReplayTransition,
    ReplayTransitionBatch,
)


class LockedReplayBuffer:
    """Serialize all access to a wrapped ``ReplayBuffer`` behind one lock.

    Duck-types the buffer surface ``ExperienceCollector`` and the scheduling
    loops use (``add``, ``sample``, ``sample_transitions``, ``size``,
    ``capacity``); any other attribute is delegated read-only to the inner
    buffer.

    Attributes:
        inner: The wrapped, single-thread ``ReplayBuffer``.
    """

    def __init__(self, inner: ReplayBuffer) -> None:
        """Wraps an existing buffer; the caller must stop using it directly.

        Args:
            inner: Real replay buffer that receives all delegated calls.
        """
        self.inner = inner
        self._lock = threading.Lock()

    def add(self, replay_transition: ReplayTransition) -> None:
        """Append one transition under the lock.

        Args:
            replay_transition: Transition forwarded to the inner buffer.
        """
        with self._lock:
            self.inner.add(replay_transition)

    def sample(self, batch_size: int, seq_len: int) -> ReplayBatch:
        """Sample a replay batch under the lock.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Number of consecutive transitions per window.

        Returns:
            A ``ReplayBatch`` with ``(batch_size, seq_len)`` leading axes.
        """
        with self._lock:
            return self.inner.sample(batch_size, seq_len)

    def sample_transitions(
        self, batch_size: int, seq_len: int
    ) -> ReplayTransitionBatch:
        """Sample copied transition windows under the lock.

        Args:
            batch_size: Number of windows to sample.
            seq_len: Number of consecutive transitions per window.

        Returns:
            ``batch_size`` sampled transition sequences of length ``seq_len``.
        """
        with self._lock:
            return self.inner.sample_transitions(batch_size, seq_len)

    @property
    def size(self) -> int:
        """Number of stored transitions (read under the lock)."""
        with self._lock:
            return self.inner.size

    @property
    def capacity(self) -> int:
        """Maximum number of transitions the inner buffer can hold."""
        return self.inner.capacity

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attribute read to the inner buffer.

        Args:
            name: Attribute name not found on the wrapper itself.

        Returns:
            The inner buffer's attribute value.

        Raises:
            AttributeError: If the inner buffer lacks the attribute too.
        """
        return getattr(self.inner, name)
