"""Fakes for driving the routed adapter pipeline on CPU.

The real gate for every adapter is a SLURM end-to-end run; these fakes exist so
the wiring (routing -> replay -> encoder -> train_step -> act) is verified in
seconds, without Habitat scenes or VGGT weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGTExtractOutput

PATCH_GRID = 37
AGG_TOKENS = 1374
AGG_HALF_DIM = 1024
# The real point head's confidence is an ``expp1`` activation (>= 1); adapters
# admit points above their own threshold (1.5 for the voxel buffer), so the fake
# must sit above it or the accumulation path never runs.
FAKE_CONFIDENCE = 2.0


@dataclass
class FakeEnv:
    """Deterministic env stub with fixed-length episodes.

    Emits the same frame contract as the Habitat wrapper: HWC uint8 renders at
    ``resolution``, ``is_first`` on resets, ``previous_action`` on steps, and
    ``done`` on the last step of an episode.
    """

    resolution: int = 64
    episode_len: int = 6
    num_actions: int = 4
    scene_id: str = "scene-a"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self._step = 0
        self.reset_count = 0
        self.closed = False

    def _image(self) -> jnp.ndarray:
        shape = (self.resolution, self.resolution, 3)
        return jnp.asarray(self._rng.integers(0, 256, shape, dtype=np.uint8))

    def reset(self) -> ObservationFrame:
        self._step = 0
        self.reset_count += 1
        return ObservationFrame(
            image=self._image(), is_first=True, scene_id=self.scene_id
        )

    def step(self, action: int) -> ObservationFrame:
        self._step += 1
        return ObservationFrame(
            image=self._image(),
            is_first=False,
            previous_action=action,
            reward=0.5,
            done=self._step >= self.episode_len,
            scene_id=self.scene_id,
        )

    def close(self) -> None:
        self.closed = True


class FakeExtractor:
    """Frozen-extractor stub: shape-correct VGGT output, no weights.

    Mirrors the real extractor's boundary contract - a frame with ``is_first``
    resets the stream before the frame is consumed - and records those resets so
    tests can assert it.
    """

    def __init__(self, patch_grid: int = PATCH_GRID) -> None:
        self.patch_grid = patch_grid
        self.scene_resets: list[str] = []
        self.extract_count = 0

    def extract(self, source: ObservationFrame) -> VGGTExtractOutput:
        """Return a shape-correct output derived from the frame's mean intensity.

        ``world_points`` matches the frame's spatial layout, as the real point
        head does - adapters zip the point map against the RGB frame.
        """
        if source.is_first:
            self.scene_resets.append(source.scene_id)
        self.extract_count += 1
        image = source.image
        height, width = jnp.shape(image)[:2]
        offset = jnp.mean(jnp.asarray(image, jnp.float32)) / 255.0
        # Spatially varying, like a real point map: a constant map would have
        # zero extent and hide extent-relative voxel sizing.
        rows = jnp.linspace(0.0, 1.0, height)[:, None, None]
        cols = jnp.linspace(0.0, 1.0, width)[None, :, None]
        world_points = jnp.broadcast_to(
            offset + rows + cols, (height, width, 3)
        ).astype(jnp.float32)
        return VGGTExtractOutput(
            world_points=world_points,
            confidence=jnp.full((height, width), FAKE_CONFIDENCE, dtype=jnp.float32),
            camera_pose=jnp.arange(9, dtype=jnp.float32) * offset,
            frame_tokens=jnp.zeros((AGG_TOKENS, AGG_HALF_DIM), dtype=jnp.float32),
            global_tokens=jnp.full(
                (AGG_TOKENS, AGG_HALF_DIM), offset, dtype=jnp.float32
            ),
        )


@pytest.fixture(name="fake_extractor")
def fake_extractor_fixture() -> FakeExtractor:
    """A fresh frozen-extractor stub."""
    return FakeExtractor()
