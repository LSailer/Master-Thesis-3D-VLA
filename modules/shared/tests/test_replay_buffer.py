"""Tests for the unified ReplayBuffer with BufferConfig."""

import tempfile

import numpy as np
import jax.numpy as jnp
import pytest

from modules.shared.replay_buffer import ReplayBuffer, BufferConfig, ValReplayDataset


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


class TestEpisodeBoundaries:
    """Document how ReplayBuffer exposes episode boundaries to the RSSM."""

    def test_done_boundary_sets_following_is_first(self):
        cfg = BufferConfig(capacity=20, obs_shape=(2,),
                           obs_dtype="float32", normalize_obs=False)
        buf = ReplayBuffer(cfg)
        for episode in range(3):
            for step in range(4):
                buf.add(
                    np.array([episode, step], dtype=np.float32),
                    action=episode,
                    reward=float(episode * 10 + step),
                    done=(step == 3),
                )

        np.random.seed(0)
        for _ in range(50):
            batch = buf.sample(batch_size=4, seq_len=5)
            obs = np.array(batch["obs"])
            dones = np.array(batch["dones"])
            is_first = np.array(batch["is_first"])

            assert np.all(is_first[:, 0] == 1.0)
            for b in range(obs.shape[0]):
                for t in range(1, obs.shape[1]):
                    episode_changed = obs[b, t, 0] != obs[b, t - 1, 0]
                    if episode_changed:
                        assert dones[b, t - 1] == 1.0
                        assert is_first[b, t] == 1.0

    def test_terminal_flag_survives_successful_episode_end(self):
        cfg = BufferConfig(capacity=12, obs_shape=(2,),
                           obs_dtype="float32", normalize_obs=False)
        buf = ReplayBuffer(cfg)
        for step in range(6):
            buf.add(
                np.array([0, step], dtype=np.float32),
                action=0,
                reward=float(step),
                done=(step == 2 or step == 5),
                terminal=(step == 2),
            )

        np.random.seed(1)
        batch = buf.sample(batch_size=8, seq_len=3)
        dones = np.array(batch["dones"])
        terminals = np.array(batch["terminals"])

        assert np.any(terminals == 1.0)
        assert np.all(terminals <= dones)
        assert np.any((dones == 1.0) & (terminals == 0.0))

    def test_wraparound_episode_change_has_reset_marker(self):
        cfg = BufferConfig(capacity=10, obs_shape=(2,),
                           obs_dtype="float32", normalize_obs=False)
        buf = ReplayBuffer(cfg)
        for i in range(16):
            episode, step = divmod(i, 4)
            buf.add(
                np.array([episode, step], dtype=np.float32),
                action=0,
                reward=float(i),
                done=(step == 3),
                terminal=(episode == 2 and step == 3),
            )

        assert buf.size == cfg.capacity
        np.random.seed(2)
        for _ in range(100):
            batch = buf.sample(batch_size=6, seq_len=4)
            obs = np.array(batch["obs"])
            dones = np.array(batch["dones"])
            is_first = np.array(batch["is_first"])
            for b in range(obs.shape[0]):
                for t in range(1, obs.shape[1]):
                    if obs[b, t, 0] != obs[b, t - 1, 0]:
                        assert dones[b, t - 1] == 1.0
                        assert is_first[b, t] == 1.0


class TestValReplayDataset:
    """Test ValReplayDataset normalization behavior."""

    @pytest.fixture
    def val_npz(self, tmp_path):
        """Create a minimal .npz file with 2 episodes of 10 steps each."""
        N = 20
        obs = np.full((N, 3, 4, 4), 200, dtype=np.uint8)
        actions = np.zeros(N, dtype=np.int32)
        rewards = np.ones(N, dtype=np.float32)
        dones = np.zeros(N, dtype=bool)
        dones[9] = True   # episode 1 ends at step 9
        dones[19] = True  # episode 2 ends at step 19
        terminals = np.zeros(N, dtype=bool)
        path = str(tmp_path / "val.npz")
        np.savez(path, obs=obs, actions=actions, rewards=rewards,
                 dones=dones, terminals=terminals)
        return path

    def test_default_normalizes(self, val_npz):
        ds = ValReplayDataset(val_npz)
        batch = ds.sample(batch_size=2, seq_len=5)
        # uint8 value 200 / 255 ≈ 0.784
        assert jnp.allclose(batch["obs"], 200.0 / 255.0, atol=1e-5)

    def test_normalize_false_skips_division(self, val_npz):
        ds = ValReplayDataset(val_npz, normalize=False)
        batch = ds.sample(batch_size=2, seq_len=5)
        # Should keep raw value 200.0 (as float32, no /255)
        assert jnp.allclose(batch["obs"], 200.0, atol=1e-5)
