"""Behavior checks for the replay buffer implementation."""

from __future__ import annotations

import numpy as np
import pytest

from src.buffer.replay_buffer import HybridObservation, ReplayBuffer, ReplayTransition
from src.environments.observation import ObservationFrame


def _buffer(capacity: int) -> ReplayBuffer:
    """Create a replay buffer with enough actions for test transition ids."""
    return ReplayBuffer(capacity=capacity, num_actions=10)


def _transition(index: int, *, is_first: bool = False) -> ReplayTransition:
    """Create one scalar replay transition with identifiable values.

    Args:
        index: Value encoded into observation, action, and reward.
        is_first: Episode-start flag for the returned transition.

    Returns:
        A replay transition suitable for deterministic buffer checks.
    """
    return ReplayTransition(
        obs=np.array([index], dtype=np.float32),
        action=index,
        reward=float(index),
        is_first=is_first,
        is_episode_end=False,
    )


class TestReplayBufferUnitBehavior:
    """Unit-level replay-buffer contract checks."""

    def test_add_stores_deepcopy_of_transition(self) -> None:
        """Mutating caller-owned arrays after add must not change replay storage."""
        buffer = _buffer(capacity=2)
        source_obs = np.array([1.0], dtype=np.float32)
        transition = ReplayTransition(
            obs=source_obs,
            action=1,
            reward=1.0,
            is_first=True,
            is_episode_end=False,
        )

        buffer.add(transition)
        source_obs[0] = 99.0

        stored = buffer.sample_transitions(batch_size=1, seq_len=1)[0][0]
        assert isinstance(stored.obs, np.ndarray)
        assert stored.obs[0] == 1.0

    def test_sample_transitions_returns_deepcopy_not_storage_reference(self) -> None:
        """Mutating a sampled transition must not mutate replay storage."""
        buffer = _buffer(capacity=2)
        buffer.add(_transition(1))

        sampled = buffer.sample_transitions(batch_size=1, seq_len=1)[0][0]
        assert isinstance(sampled.obs, np.ndarray)
        sampled.obs[0] = 99.0

        sampled_again = buffer.sample_transitions(batch_size=1, seq_len=1)[0][0]
        assert isinstance(sampled_again.obs, np.ndarray)
        assert sampled_again.obs[0] == 1.0

    def test_hybrid_observation_is_deepcopied(self) -> None:
        """Hybrid image/wp_cp arrays must also be owned by replay storage."""
        buffer = _buffer(capacity=2)
        image = np.zeros((1, 2, 2), dtype=np.uint8)
        wp_cp = np.zeros((3,), dtype=np.float32)
        buffer.add(
            ReplayTransition(
                obs=HybridObservation(image=image, wp_cp=wp_cp),
                action=1,
                reward=1.0,
                is_first=False,
                is_episode_end=False,
            )
        )

        image[...] = 255
        wp_cp[...] = 7.0

        sampled = buffer.sample_transitions(batch_size=1, seq_len=1)[0][0]
        assert isinstance(sampled.obs, HybridObservation)
        assert np.all(sampled.obs.image == 0)
        assert np.allclose(sampled.obs.wp_cp, 0.0)

    def test_wraparound_sampling_keeps_chronological_sequences(self) -> None:
        """Modulo gathering must preserve time order after ring wraparound."""
        buffer = _buffer(capacity=5)
        for index in range(8):
            buffer.add(_transition(index))

        assert buffer.idx == 3
        assert buffer.size == 5

        np.random.seed(0)
        for _ in range(50):
            batch = buffer.sample_transitions(batch_size=4, seq_len=3)
            for sequence in batch:
                rewards = [transition.reward for transition in sequence]
                assert rewards[1] - rewards[0] == 1.0
                assert rewards[2] - rewards[1] == 1.0

    def test_sample_marks_window_start_as_sequence_start(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sampled windows mark their first transition as an RSSM reset point."""
        buffer = _buffer(capacity=4)
        buffer.add(_transition(0, is_first=True))
        buffer.add(_transition(1, is_first=False))
        buffer.add(_transition(2, is_first=False))

        def fixed_start(_low: int, _high: int, size: int) -> np.ndarray:
            return np.full(size, 1, dtype=np.int64)

        monkeypatch.setattr(np.random, "randint", fixed_start)

        sequence = buffer.sample_transitions(batch_size=1, seq_len=2)[0]
        assert [transition.reward for transition in sequence] == [1.0, 2.0]
        assert sequence[0].is_first is True

    def test_sample_returns_frame_aligned_replay_batch(self) -> None:
        """Model batches keep action/reward/end labels aligned with observations."""
        buffer = ReplayBuffer(capacity=4, num_actions=4)
        buffer.add(
            ReplayTransition(
                obs=np.array([10.0], dtype=np.float32),
                action=2,
                reward=3.0,
                is_first=False,
                is_episode_end=True,
            )
        )

        batch = buffer.sample(batch_size=1, seq_len=1)

        np.testing.assert_allclose(np.asarray(batch.obs), np.array([[[10.0]]]))
        np.testing.assert_allclose(np.asarray(batch.actions[0, 0]), np.eye(4)[2])
        np.testing.assert_allclose(np.asarray(batch.rewards), np.array([[3.0]]))
        np.testing.assert_allclose(np.asarray(batch.is_episode_end), np.array([[1.0]]))
        np.testing.assert_allclose(np.asarray(batch.is_first), np.array([[1.0]]))

    def test_invalid_capacity_raises(self) -> None:
        """Capacity must be positive."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=0, num_actions=10)

    def test_sample_before_add_raises(self) -> None:
        """Sampling an empty buffer is invalid."""
        buffer = _buffer(capacity=2)
        with pytest.raises(RuntimeError, match="before adding"):
            buffer.sample(batch_size=1, seq_len=1)

    def test_sample_longer_than_buffer_size_raises(self) -> None:
        """A sampled sequence cannot be longer than the stored data."""
        buffer = _buffer(capacity=3)
        buffer.add(_transition(0))
        with pytest.raises(ValueError, match="not enough data"):
            buffer.sample(batch_size=1, seq_len=2)


class TestReplayBufferFrameBehavior:
    """Checks at the seam with environment observation frames."""

    def test_transition_from_observation_frame_copies_scalar_fields(self) -> None:
        """ReplayTransition.from_frame should use the environment-frame contract."""
        frame = ObservationFrame(
            image=np.empty((0,), dtype=np.uint8),
            is_first=False,
            previous_action=3,
            reward=1.5,
            done=True,
        )

        transition = ReplayTransition.from_frame(
            obs=np.array([7.0], dtype=np.float32),
            frame=frame,
        )

        assert transition.action == 3
        assert transition.reward == 1.5
        assert transition.is_first is False
        assert transition.is_episode_end is True

    def test_transition_from_reset_frame_raises(self) -> None:
        """Reset frames have no previous action and are not replay transitions."""
        reset_frame = ObservationFrame(
            image=np.empty((0,), dtype=np.uint8), is_first=True
        )

        with pytest.raises(ValueError, match="reset frame"):
            ReplayTransition.from_frame(
                obs=np.array([0.0], dtype=np.float32),
                frame=reset_frame,
            )
