"""Tests for src/buffer/replay_arrays.py — replay window -> array packing."""

import numpy as np

from src.buffer.replay_arrays import replay_batch_to_arrays
from src.buffer.replay_buffer import ReplayTransition


class TestReplayBatchToArrays:
    """Replay transition windows become raw training-aligned arrays."""

    def test_marks_is_first_after_episode_end(self):
        batch = replay_batch_to_arrays(
            [
                [
                    ReplayTransition(
                        obs=np.array([0.0], dtype=np.float32),
                        action=0,
                        reward=0.0,
                        is_first=False,
                        is_episode_end=False,
                    ),
                    ReplayTransition(
                        obs=np.array([1.0], dtype=np.float32),
                        action=1,
                        reward=1.0,
                        is_first=False,
                        is_episode_end=True,
                    ),
                    ReplayTransition(
                        obs=np.array([2.0], dtype=np.float32),
                        action=2,
                        reward=2.0,
                        is_first=False,
                        is_episode_end=False,
                    ),
                ]
            ]
        )

        np.testing.assert_array_equal(
            np.asarray(batch["is_first"]), np.array([[True, False, True]])
        )
