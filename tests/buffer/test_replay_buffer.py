"""Tests for the lazy ReplayBuffer."""

from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from src.buffer.replay_buffer import ReplayBuffer, ValReplayDataset
from src.environments.observation import ObservationFrame


def _frame(action: int = 0, reward: float = 0.0, done: bool = False) -> ObservationFrame:
    return ObservationFrame(
        image=np.empty((0,), dtype=np.uint8),
        is_first=False,
        previous_action=action,
        reward=reward,
        done=done,
    )


def _add(
    buffer: ReplayBuffer,
    obs: np.ndarray | dict[str, np.ndarray],
    *,
    action: int = 0,
    reward: float = 0.0,
    done: bool = False,
) -> None:
    buffer.add(obs, _frame(action=action, reward=reward, done=done))


class TestReplayBufferPolicy:
    def test_capacity_only_initializes_on_first_add(self):
        buf = ReplayBuffer(capacity=3)
        assert buf.obs is None

        _add(buf, np.zeros((2,), dtype=np.float32))

        obs_storage = cast(np.ndarray, buf.obs)
        assert obs_storage.shape == (3, 2)
        assert obs_storage.dtype == np.float32

    def test_sample_before_first_add_raises(self):
        buf = ReplayBuffer(capacity=3)

        with pytest.raises(RuntimeError, match="before adding"):
            buf.sample(batch_size=1, seq_len=1)

    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=0)


class TestReplayBufferUint8:
    """ReplayBuffer preserves uint8 image observations."""

    @pytest.fixture
    def buf(self):
        return ReplayBuffer(capacity=50)

    def test_add_and_size(self, buf):
        obs = np.random.randint(0, 255, (3, 4, 4), dtype=np.uint8)
        _add(buf, obs, action=1, reward=1.0)
        assert buf.size == 1

    def test_sample_shapes(self, buf):
        for i in range(20):
            obs = np.random.randint(0, 255, (3, 4, 4), dtype=np.uint8)
            _add(buf, obs, action=i % 5, reward=float(i), done=i == 9)
        batch = buf.sample(batch_size=4, seq_len=5)
        assert batch["obs"].shape == (4, 5, 3, 4, 4)
        assert batch["actions"].shape == (4, 5)
        assert batch["rewards"].shape == (4, 5)
        assert batch["is_episode_end"].shape == (4, 5)
        assert batch["is_first"].shape == (4, 5)

    def test_sample_preserves_uint8(self, buf):
        obs = np.full((3, 4, 4), 255, dtype=np.uint8)
        for _ in range(10):
            _add(buf, obs)
        batch = buf.sample(2, 3)
        assert batch["obs"].dtype == jnp.uint8
        assert int(batch["obs"][0, 0, 0, 0, 0]) == 255

    def test_sample_dtypes(self, buf):
        for _ in range(10):
            _add(buf, np.zeros((3, 4, 4), dtype=np.uint8))
        batch = buf.sample(2, 3)
        assert batch["obs"].dtype == jnp.uint8
        assert batch["actions"].dtype == jnp.int32
        assert batch["rewards"].dtype == jnp.float32
        assert batch["is_first"].dtype == jnp.bool_
        assert batch["is_episode_end"].dtype == jnp.bool_

    def test_ring_buffer_wraps(self, buf):
        for i in range(60):
            _add(buf, np.zeros((3, 4, 4), dtype=np.uint8), reward=float(i))
        assert buf.size == 50

    def test_is_first_after_episode_end(self, buf):
        for i in range(10):
            _add(buf, np.zeros((3, 4, 4), dtype=np.uint8), done=(i == 4))
        batch = buf.sample(1, 8)
        assert batch["is_first"][0, 0]


class TestReplayBufferFloat32:
    """ReplayBuffer preserves float32 feature observations."""

    @pytest.fixture
    def buf(self):
        return ReplayBuffer(capacity=50)

    def test_add_and_size(self, buf):
        obs = np.random.randn(4116).astype(np.float32)
        _add(buf, obs, action=1, reward=1.0)
        assert buf.size == 1

    def test_sample_shapes(self, buf):
        for i in range(20):
            obs = np.random.randn(4116).astype(np.float32)
            _add(buf, obs, action=i % 4, reward=float(i), done=i == 9)
        batch = buf.sample(batch_size=4, seq_len=5)
        assert batch["obs"].shape == (4, 5, 4116)
        assert batch["actions"].shape == (4, 5)

    def test_sample_does_not_normalize(self, buf):
        obs = np.full((4116,), 255.0, dtype=np.float32)
        for _ in range(10):
            _add(buf, obs)
        batch = buf.sample(2, 3)
        assert jnp.allclose(batch["obs"], 255.0, atol=1e-5)

    def test_episode_end_flag(self, buf):
        for i in range(10):
            _add(buf, np.zeros(4116, dtype=np.float32), done=(i == 5))
        batch = buf.sample(1, 8)
        assert batch["is_episode_end"].dtype == jnp.bool_
        assert jnp.any(batch["is_episode_end"])


class TestReplayBufferMappingObs:
    """ReplayBuffer supports explicit multi-modal observation fields."""

    @pytest.fixture
    def buf(self):
        return ReplayBuffer(capacity=50)

    def test_add_and_sample_fields(self, buf):
        for i in range(20):
            obs = {
                "image": np.full((3, 4, 4), i, dtype=np.uint8),
                "wp_cp": np.full((6,), float(i), dtype=np.float32),
            }
            _add(buf, obs, action=i % 4, reward=float(i), done=i == 9)

        batch = buf.sample(batch_size=4, seq_len=5)
        assert set(batch["obs"]) == {"image", "wp_cp"}
        assert batch["obs"]["image"].shape == (4, 5, 3, 4, 4)
        assert batch["obs"]["image"].dtype == jnp.uint8
        assert batch["obs"]["wp_cp"].shape == (4, 5, 6)
        assert batch["obs"]["wp_cp"].dtype == jnp.float32
        assert batch["actions"].shape == (4, 5)
        assert batch["is_first"].shape == (4, 5)

    def test_missing_field_raises(self, buf):
        _add(
            buf,
            {
                "image": np.zeros((3, 4, 4), dtype=np.uint8),
                "wp_cp": np.zeros((6,), dtype=np.float32),
            },
        )
        obs = {"image": np.zeros((3, 4, 4), dtype=np.uint8)}
        with pytest.raises(KeyError, match="observation keys changed"):
            _add(buf, obs)

    def test_mapping_field_dtype_is_preserved(self):
        buf = ReplayBuffer(capacity=10)
        obs = {
            "image": np.array([255], dtype=np.uint8),
            "features": np.array([255.0], dtype=np.float32),
        }
        for _ in range(4):
            _add(buf, obs)

        batch = buf.sample(batch_size=1, seq_len=2)
        assert batch["obs"]["image"].dtype == jnp.uint8
        assert batch["obs"]["features"].dtype == jnp.float32
        assert jnp.allclose(batch["obs"]["features"], 255.0)


class TestWriteHeadSafety:
    """Verify sampled sequences never cross the write-head boundary."""

    def test_no_temporal_discontinuity_after_wrap(self):
        cap, seq_len = 20, 5
        buf = ReplayBuffer(capacity=cap)

        for i in range(30):
            _add(
                buf,
                np.array([float(i), 0.0], dtype=np.float32),
                reward=float(i),
            )

        assert buf.size == cap
        assert buf.idx == 10

        np.random.seed(42)
        for _ in range(200):
            batch = buf.sample(batch_size=8, seq_len=seq_len)
            rewards = np.array(batch["rewards"])
            for b in range(8):
                seq = rewards[b]
                diffs = np.diff(seq)
                assert np.all(diffs >= 0), (
                    f"Non-monotonic reward sequence {seq.tolist()} — "
                    f"likely crossed the write head"
                )

    def test_sample_works_when_idx_near_zero(self):
        cap, seq_len = 20, 5
        buf = ReplayBuffer(capacity=cap)

        for i in range(cap + 1):
            _add(buf, np.array([float(i), 0.0], dtype=np.float32), reward=float(i))

        assert buf.idx == 1
        assert buf.size == cap
        batch = buf.sample(batch_size=4, seq_len=seq_len)
        obs_batch = cast(jnp.ndarray, batch["obs"])
        assert obs_batch.shape == (4, seq_len, 2)

    def test_sample_works_when_idx_near_end(self):
        cap, seq_len = 20, 5
        buf = ReplayBuffer(capacity=cap)

        for i in range(cap + cap - 2):
            _add(buf, np.array([float(i), 0.0], dtype=np.float32), reward=float(i))

        assert buf.idx == 18
        assert buf.size == cap
        batch = buf.sample(batch_size=4, seq_len=seq_len)
        obs_batch = cast(jnp.ndarray, batch["obs"])
        assert obs_batch.shape == (4, seq_len, 2)


class TestEpisodeBoundaries:
    """Document how ReplayBuffer exposes episode boundaries to the RSSM."""

    def test_episode_end_boundary_sets_following_is_first(self):
        buf = ReplayBuffer(capacity=20)
        for episode in range(3):
            for step in range(4):
                _add(
                    buf,
                    np.array([episode, step], dtype=np.float32),
                    action=episode,
                    reward=float(episode * 10 + step),
                    done=(step == 3),
                )

        np.random.seed(0)
        for _ in range(50):
            batch = buf.sample(batch_size=4, seq_len=5)
            obs = np.array(batch["obs"])
            episode_end = np.array(batch["is_episode_end"])
            is_first = np.array(batch["is_first"])

            assert np.all(is_first[:, 0])
            for b in range(obs.shape[0]):
                for t in range(1, obs.shape[1]):
                    episode_changed = obs[b, t, 0] != obs[b, t - 1, 0]
                    if episode_changed:
                        assert episode_end[b, t - 1]
                        assert is_first[b, t]

    def test_wraparound_episode_change_has_reset_marker(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(16):
            episode, step = divmod(i, 4)
            _add(
                buf,
                np.array([episode, step], dtype=np.float32),
                reward=float(i),
                done=(step == 3),
            )

        assert buf.size == 10
        np.random.seed(2)
        for _ in range(100):
            batch = buf.sample(batch_size=6, seq_len=4)
            obs = np.array(batch["obs"])
            episode_end = np.array(batch["is_episode_end"])
            is_first = np.array(batch["is_first"])
            for b in range(obs.shape[0]):
                for t in range(1, obs.shape[1]):
                    if obs[b, t, 0] != obs[b, t - 1, 0]:
                        assert episode_end[b, t - 1]
                        assert is_first[b, t]


class TestValReplayDataset:
    """Test ValReplayDataset episode reconstruction behavior."""

    @pytest.fixture
    def val_npz(self, tmp_path):
        num_steps = 20
        obs = np.full((num_steps, 3, 4, 4), 200, dtype=np.uint8)
        actions = np.zeros(num_steps, dtype=np.int32)
        rewards = np.ones(num_steps, dtype=np.float32)
        episode_ends = np.zeros(num_steps, dtype=bool)
        episode_ends[9] = True
        episode_ends[19] = True
        path = str(tmp_path / "val.npz")
        np.savez(
            path,
            obs=obs,
            actions=actions,
            rewards=rewards,
            episode_ends=episode_ends,
        )
        return path

    def test_sample_preserves_obs_dtype(self, val_npz):
        ds = ValReplayDataset(val_npz)
        batch = ds.sample(batch_size=2, seq_len=5)
        assert batch["obs"].dtype == jnp.uint8
        assert int(batch["obs"][0, 0, 0, 0, 0]) == 200

    def test_episode_count_uses_episode_ends(self, val_npz):
        ds = ValReplayDataset(val_npz)
        assert ds.episode_count() == 2
