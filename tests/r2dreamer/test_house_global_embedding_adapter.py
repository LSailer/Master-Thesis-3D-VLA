"""CPU tests for the L1 house-global-embedding obs adapter.

Covers src/r2dreamer/adapters/token_adapters.py::VGGTHouseGlobalEmbeddingObsAdapter
— RGB-only replay, live global patch token injection, scene-aware reset, and
the optional PLY dump trigger (src/prototyp/house_global_embedding/IDEA.md).
"""

import numpy as np
import pytest

from src.buffer.replay_buffer import ReplayBatch
from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.token_adapters import (
    VGGTHouseGlobalEmbeddingObsAdapter,
)
from src.r2dreamer.observation_keys import (
    GLOBAL_PATCH_TOKENS_KEY,
    HYBRID_IMAGE_KEY,
)

N_TOKENS = 1374
TOKEN_DIM = 1024
PATCH_START = 5
N_PATCHES = N_TOKENS - PATCH_START  # 1369


class _FakeGlobalTokenExtractor:
    """Fake extractor returning a deterministic global_tokens (1374, 1024)."""

    def __init__(self, tokens: np.ndarray | None = None):
        self._tokens = tokens
        self.reset_for_scene_calls: list[str] = []
        self.ply_dumps: list[str] = []

    def reset_for_scene(self, scene_id: str) -> None:
        self.reset_for_scene_calls.append(scene_id)

    def reset(self) -> None:
        pass

    def extract(self, obs):
        import jax.numpy as jnp

        if self._tokens is None:
            tokens = np.arange(N_TOKENS * TOKEN_DIM, dtype=np.float32).reshape(
                N_TOKENS, TOKEN_DIM
            ) / 1000.0
        else:
            tokens = self._tokens
        return {"aggregator_features": jnp.asarray(tokens)}

    def write_point_cloud_ply(self, path: str) -> None:
        self.ply_dumps.append(path)


def _frame(seed: int, *, is_first=False, scene_id="houseA"):
    image = np.random.default_rng(seed).integers(
        0, 256, size=(3, 8, 8), dtype=np.uint8
    )
    return ObservationFrame(image=image, is_first=is_first, scene_id=scene_id)


def test_adapter_stores_rgb_only_in_replay_and_live_tokens_in_agent_obs():
    tokens = (
        np.arange(N_TOKENS * TOKEN_DIM, dtype=np.float32).reshape(
            N_TOKENS, TOKEN_DIM
        )
        / 1000.0
    )
    adapter = VGGTHouseGlobalEmbeddingObsAdapter(_FakeGlobalTokenExtractor(tokens))
    replay, agent_obs = adapter.transform(_frame(0, is_first=True))

    assert set(replay) == {HYBRID_IMAGE_KEY}
    assert replay[HYBRID_IMAGE_KEY].shape == (3, 64, 64)
    assert replay[HYBRID_IMAGE_KEY].dtype == np.uint8

    assert set(agent_obs) == {
        HYBRID_IMAGE_KEY,
        GLOBAL_PATCH_TOKENS_KEY,
        "is_first",
    }
    assert agent_obs[GLOBAL_PATCH_TOKENS_KEY].shape == (N_PATCHES, TOKEN_DIM)


def test_adapter_split_is_patches_only():
    tokens = (
        np.arange(N_TOKENS * TOKEN_DIM, dtype=np.float32).reshape(
            N_TOKENS, TOKEN_DIM
        )
        / 1000.0
    )
    adapter = VGGTHouseGlobalEmbeddingObsAdapter(_FakeGlobalTokenExtractor(tokens))
    _, agent_obs = adapter.transform(_frame(1, is_first=True))

    np.testing.assert_allclose(
        np.asarray(agent_obs[GLOBAL_PATCH_TOKENS_KEY]),
        tokens[PATCH_START:],
    )
    assert agent_obs[GLOBAL_PATCH_TOKENS_KEY].shape[0] == N_PATCHES


def test_augment_replay_batch_injects_latest_live_patch_tokens():
    tokens_a = (
        np.arange(N_TOKENS * TOKEN_DIM, dtype=np.float32).reshape(
            N_TOKENS, TOKEN_DIM
        )
        / 1000.0
    )
    tokens_b = tokens_a + 1.0
    extractor = _FakeGlobalTokenExtractor(tokens_a)
    adapter = VGGTHouseGlobalEmbeddingObsAdapter(extractor)
    adapter.transform(_frame(0, is_first=True))

    batch = ReplayBatch(
        obs={HYBRID_IMAGE_KEY: np.zeros((1, 2, 3, 64, 64), dtype=np.uint8)},
        actions=np.zeros((1, 2), dtype=np.int32),
        rewards=np.zeros((1, 2), dtype=np.float32),
        is_episode_end=np.zeros((1, 2), dtype=bool),
        is_first=np.zeros((1, 2), dtype=np.float32),
    )
    augmented = adapter.augment_replay_batch(batch)

    assert set(augmented.obs) == {HYBRID_IMAGE_KEY, GLOBAL_PATCH_TOKENS_KEY}
    assert augmented.obs[HYBRID_IMAGE_KEY].shape == (1, 2, 3, 64, 64)
    assert augmented.obs[GLOBAL_PATCH_TOKENS_KEY].shape == (N_PATCHES, TOKEN_DIM)
    np.testing.assert_allclose(
        np.asarray(augmented.obs[GLOBAL_PATCH_TOKENS_KEY]),
        tokens_a[PATCH_START:],
    )

    extractor._tokens = tokens_b
    adapter.transform(_frame(1, is_first=False))
    augmented = adapter.augment_replay_batch(batch)
    np.testing.assert_allclose(
        np.asarray(augmented.obs[GLOBAL_PATCH_TOKENS_KEY]),
        tokens_b[PATCH_START:],
    )


def test_on_episode_reset_is_scene_aware():
    extractor = _FakeGlobalTokenExtractor()
    adapter = VGGTHouseGlobalEmbeddingObsAdapter(extractor)
    assert adapter.on_episode_reset is not None

    adapter.on_episode_reset("house-7")
    assert extractor.reset_for_scene_calls == ["house-7"]
    adapter.on_episode_reset()
    assert extractor.reset_for_scene_calls == ["house-7", "scene"]


def test_dump_disabled_by_default():
    extractor = _FakeGlobalTokenExtractor()
    adapter = VGGTHouseGlobalEmbeddingObsAdapter(extractor)
    for s in range(6):
        adapter.transform(_frame(s, is_first=(s == 0)))
    assert extractor.ply_dumps == []
    assert adapter.diagnostics()["house_global_embedding/dump_count"] == 0.0


def test_dump_every_n_steps(tmp_path):
    extractor = _FakeGlobalTokenExtractor()
    adapter = VGGTHouseGlobalEmbeddingObsAdapter(
        extractor, pointcloud_dump_every=4, pointcloud_dump_dir=str(tmp_path)
    )
    for s in range(8):
        adapter.transform(_frame(s, is_first=(s == 0)))

    dumped_steps = sorted(p.split("step")[-1].removesuffix(".ply") for p in extractor.ply_dumps)
    assert dumped_steps == ["4", "8"]
    assert len(extractor.ply_dumps) == 2
    for path in extractor.ply_dumps:
        assert path.startswith(str(tmp_path))
    assert adapter.diagnostics()["house_global_embedding/dump_count"] == 2.0


def test_end_of_first_episode_dump(tmp_path):
    extractor = _FakeGlobalTokenExtractor()
    adapter = VGGTHouseGlobalEmbeddingObsAdapter(
        extractor, pointcloud_dump_every=10_000, pointcloud_dump_dir=str(tmp_path)
    )
    adapter.transform(_frame(0, is_first=True))
    for s in range(1, 4):
        adapter.transform(_frame(s, is_first=False))
    adapter.transform(_frame(4, is_first=True))

    end_dumps = [p for p in extractor.ply_dumps if "end_of_first_episode" in p]
    assert len(end_dumps) == 1
    assert len(extractor.ply_dumps) == 1
    adapter.transform(_frame(5, is_first=False))
    adapter.transform(_frame(6, is_first=True))
    assert len([p for p in extractor.ply_dumps if "end_of_first_episode" in p]) == 1


def test_diagnostics_reports_camera_head_cache_inactive():
    extractor = _FakeGlobalTokenExtractor()
    extractor._past_kvs_camera = None
    adapter = VGGTHouseGlobalEmbeddingObsAdapter(extractor)
    adapter.transform(_frame(0, is_first=True))
    stats = adapter.diagnostics()
    assert stats["house_global_embedding/camera_head_cache_active"] == 0.0
    assert stats["house_global_embedding/env_steps"] == 1.0


def test_adapter_rejects_wrong_token_count():
    bad = np.zeros((100, TOKEN_DIM), dtype=np.float32)
    adapter = VGGTHouseGlobalEmbeddingObsAdapter(_FakeGlobalTokenExtractor(bad))
    with pytest.raises(ValueError, match="global_patch_tokens"):
        adapter.transform(_frame(0, is_first=True))
