"""Tests for the unified ReplayBuffer with BufferConfig."""

import numpy as np
import jax.numpy as jnp
import pytest

from modules.dreamerv3.replay_buffer import ReplayBuffer, BufferConfig


class TestBufferConfig:
    def test_uint8_defaults(self):
        cfg = BufferConfig(capacity=100, obs_shape=(3, 64, 64))
        assert cfg.obs_dtype == "uint8"
        assert cfg.normalize_obs is True

    def test_float32_no_normalize(self):
        cfg = BufferConfig(capacity=100, obs_shape=(4116,),
                           obs_dtype="float32", normalize_obs=False)
        assert cfg.obs_dtype == "float32"
        assert cfg.normalize_obs is False


class TestReplayBufferUint8:
    """Test ReplayBuffer with uint8 images (old ReplayBuffer behavior)."""

    @pytest.fixture
    def buf(self):
        cfg = BufferConfig(capacity=50, obs_shape=(3, 4, 4))
        return ReplayBuffer(cfg)

    def test_add_and_size(self, buf):
        obs = np.random.randint(0, 255, (3, 4, 4), dtype=np.uint8)
        buf.add(obs, action=1, reward=1.0, done=False)
        assert buf.size == 1

    def test_sample_shapes(self, buf):
        for i in range(20):
            obs = np.random.randint(0, 255, (3, 4, 4), dtype=np.uint8)
            buf.add(obs, i % 5, float(i), i == 9)
        batch = buf.sample(batch_size=4, seq_len=5)
        assert batch["obs"].shape == (4, 5, 3, 4, 4)
        assert batch["actions"].shape == (4, 5)
        assert batch["rewards"].shape == (4, 5)
        assert batch["dones"].shape == (4, 5)
        assert batch["terminals"].shape == (4, 5)
        assert batch["is_first"].shape == (4, 5)

    def test_sample_normalizes_uint8(self, buf):
        obs = np.full((3, 4, 4), 255, dtype=np.uint8)
        for _ in range(10):
            buf.add(obs, 0, 0.0, False)
        batch = buf.sample(2, 3)
        assert jnp.allclose(batch["obs"], 1.0, atol=1e-5)

    def test_sample_dtypes(self, buf):
        for _ in range(10):
            buf.add(np.zeros((3, 4, 4), dtype=np.uint8), 0, 0.0, False)
        batch = buf.sample(2, 3)
        assert batch["obs"].dtype == jnp.float32
        assert batch["actions"].dtype == jnp.int32
        assert batch["rewards"].dtype == jnp.float32
        assert batch["is_first"].dtype == jnp.float32

    def test_ring_buffer_wraps(self, buf):
        """Buffer wraps around when capacity is exceeded."""
        for i in range(60):  # capacity is 50
            buf.add(np.zeros((3, 4, 4), dtype=np.uint8), 0, float(i), False)
        assert buf.size == 50

    def test_is_first_after_done(self, buf):
        """is_first should be True at t=0 and after done flags."""
        for i in range(10):
            buf.add(np.zeros((3, 4, 4), dtype=np.uint8), 0, 0.0, done=(i == 4))
        batch = buf.sample(1, 8)
        # is_first[:, 0] is always True
        assert batch["is_first"][0, 0] == 1.0


class TestReplayBufferFloat32:
    """Test ReplayBuffer with float32 features (old VGGTReplayBuffer behavior)."""

    @pytest.fixture
    def buf(self):
        cfg = BufferConfig(capacity=50, obs_shape=(4116,),
                           obs_dtype="float32", normalize_obs=False)
        return ReplayBuffer(cfg)

    def test_add_and_size(self, buf):
        obs = np.random.randn(4116).astype(np.float32)
        buf.add(obs, action=1, reward=1.0, done=False)
        assert buf.size == 1

    def test_sample_shapes(self, buf):
        for i in range(20):
            obs = np.random.randn(4116).astype(np.float32)
            buf.add(obs, i % 4, float(i), i == 9)
        batch = buf.sample(batch_size=4, seq_len=5)
        assert batch["obs"].shape == (4, 5, 4116)
        assert batch["actions"].shape == (4, 5)

    def test_sample_does_not_normalize(self, buf):
        obs = np.full((4116,), 255.0, dtype=np.float32)
        for _ in range(10):
            buf.add(obs, 0, 0.0, False)
        batch = buf.sample(2, 3)
        # Should NOT divide by 255
        assert jnp.allclose(batch["obs"], 255.0, atol=1e-5)

    def test_terminal_flag(self, buf):
        for i in range(10):
            buf.add(np.zeros(4116, dtype=np.float32), 0, 0.0,
                    done=(i == 5), terminal=(i == 5))
        batch = buf.sample(1, 8)
        # terminals should be stored and returned
        assert batch["terminals"].dtype == jnp.float32


class TestWriteHeadSafety:
    """Verify sampled sequences never cross the write-head boundary."""

    def test_no_temporal_discontinuity_after_wrap(self):
        """After wrapping, reward sequences must be temporally contiguous.

        We write rewards 0,1,2,...,cap+extra with capacity=20, seq_len=5.
        After wrapping, positions [0..idx) have the newest rewards,
        positions [idx..cap) have older rewards. A valid sequence must
        have monotonically increasing rewards (no backward jump).
        """
        cap, seq_len = 20, 5
        cfg = BufferConfig(capacity=cap, obs_shape=(2,),
                           obs_dtype="float32", normalize_obs=False)
        buf = ReplayBuffer(cfg)

        # Write 30 items into a capacity-20 buffer (wraps at 20)
        for i in range(30):
            buf.add(np.array([float(i), 0.0], dtype=np.float32),
                    action=0, reward=float(i), done=False)

        assert buf.size == cap
        assert buf.idx == 10  # 30 % 20

        # Sample many batches — every sequence's rewards must be monotonic
        np.random.seed(42)
        for _ in range(200):
            batch = buf.sample(batch_size=8, seq_len=seq_len)
            rewards = np.array(batch["rewards"])  # (8, 5)
            for b in range(8):
                seq = rewards[b]
                diffs = np.diff(seq)
                assert np.all(diffs >= 0), (
                    f"Non-monotonic reward sequence {seq.tolist()} — "
                    f"likely crossed the write head"
                )

    def test_sample_works_when_idx_near_zero(self):
        """Edge case: buffer just wrapped, idx=1."""
        cap, seq_len = 20, 5
        cfg = BufferConfig(capacity=cap, obs_shape=(2,),
                           obs_dtype="float32", normalize_obs=False)
        buf = ReplayBuffer(cfg)

        # Write exactly cap+1 items so idx=1
        for i in range(cap + 1):
            buf.add(np.array([float(i), 0.0], dtype=np.float32),
                    action=0, reward=float(i), done=False)

        assert buf.idx == 1
        assert buf.size == cap
        # Should still be able to sample without error
        batch = buf.sample(batch_size=4, seq_len=seq_len)
        assert batch["obs"].shape == (4, seq_len, 2)

    def test_sample_works_when_idx_near_end(self):
        """Edge case: idx is close to capacity, small new region."""
        cap, seq_len = 20, 5
        cfg = BufferConfig(capacity=cap, obs_shape=(2,),
                           obs_dtype="float32", normalize_obs=False)
        buf = ReplayBuffer(cfg)

        # Write cap+2 items so idx=2
        for i in range(cap + cap - 2):
            buf.add(np.array([float(i), 0.0], dtype=np.float32),
                    action=0, reward=float(i), done=False)

        assert buf.idx == 18
        assert buf.size == cap
        batch = buf.sample(batch_size=4, seq_len=seq_len)
        assert batch["obs"].shape == (4, seq_len, 2)
