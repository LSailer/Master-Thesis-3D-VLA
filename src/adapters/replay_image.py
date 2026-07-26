"""Shared RGB helper for adapters whose env renders larger than replay stores."""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Replay/conv-branch image side length. VGGT variants render at 518 for the
# frozen extractor but only ever store and encode this 64x64 view.
REPLAY_IMAGE_SIZE = 64
REPLAY_IMAGE_SHAPE = (REPLAY_IMAGE_SIZE, REPLAY_IMAGE_SIZE, 3)


def replay_image(image: jnp.ndarray) -> jnp.ndarray:
    """Downsample an HWC frame to the ``(64, 64, 3)`` uint8 replay image.

    Resizes on device. Note this is NOT free: habitat hands the frame over as
    host numpy (``HabitatObjectNavEnv._obs_to_image`` slices the sim's rgb
    buffer), so the ``jnp.asarray`` below transfers the full render up, and
    ``transition_from_fields`` copies the 64x64 result back down for replay.
    Whether that beats a host-side resize is unmeasured - the legacy path used
    PIL on the host. Measure before assuming either way.

    Args:
        image: HWC RGB frame at the env's render resolution.

    Returns:
        The ``(64, 64, 3)`` uint8 view stored in replay and fed to the conv
        branch. Frames already at the target size pass through untouched.
    """
    image = jnp.asarray(image)
    if image.shape[:2] == REPLAY_IMAGE_SHAPE[:2]:
        return image.astype(jnp.uint8)
    resized = jax.image.resize(
        image.astype(jnp.float32), REPLAY_IMAGE_SHAPE, method="linear"
    )
    return jnp.clip(resized, 0, 255).astype(jnp.uint8)
