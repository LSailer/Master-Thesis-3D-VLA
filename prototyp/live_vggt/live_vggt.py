"""Live VGGT prototype for tracking per-scene point-cloud visibility."""

# Design note:
# For 2M environment steps, storing full VGGT world_points per step is too large:
#   world_points: (T, H, W, 3)
# At 518x518x3 this is several TB for 2M steps.
#
# Target design:
#   1. Maintain one static/global point cloud per scene:
#        scene_points_xyz: (P, 3)
#      New VGGT points are matched into this global cloud by voxel or nearest-neighbor
#      matching. Points should usually not be deleted; they become non-visible for
#      a step instead.
#
#   2. For each environment step, store only visibility into the global cloud:
#        visible_point_ids_grid: (H, W), int32
#      Each cell contains the global point id visible at that pixel, or -1 if invalid.
#      This preserves the H,W image layout without storing xyz every step.
#
#   3. To reconstruct the visible point cloud for one step:
#        ids = visible_point_ids_grid
#        mask = ids >= 0
#        visible_points_hwc = scene_points_xyz[ids]  # conceptually (H, W, 3)
#        visible_points_hwc[~mask] = NaN or 0
#
#   4. ReplayBuffer should not store full world_points. It should store compact CP
#      / camera pose plus a lookup key for the external scene cache, e.g.
#        scene_index, episode_id, env_step_id, camera_pose
#      Without such a key, sampled replay steps cannot be joined back to their
#      visible point IDs.
#
# Open representation choices:
#   - Full 518x518 point-id grid: preserves exact VGGT pixel layout but is still large.
#   - Pooled 37x37 or 64x64 point-id grid: much smaller and matches current WP/CP use.
#   - Sparse list of visible point ids: smallest, but loses H,W layout.
#   - Keyframe + delta visibility: useful if consecutive steps change little.
from pathlib import Path

import jax
import jax.numpy as jnp

# Video frames are host-side copies; NumPy marks the device -> host boundary.
import numpy as np

from src.baselines.random_agent import RandomAgent
from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.buffer.replay_buffer import (
    ObservationInput,
    ReplayBuffer,
    ReplayTransition,
)
from src.configs.config import R2DreamerConfig
from src.configs.trainer_config import TrainerConfig
from src.environments.habitat import ACTIONS, HabitatEnvConfig, HabitatObjectNavEnv
from prototyp.live_vggt.debug import Debugger
from src.shared.video_utils import write_frames_mp4
from prototyp.live_vggt.point_change_plot import (
    PointChangePlotter,
)
from src.r2dreamer.agent import R2DreamerAgent
from src.vggt.jax import JAXVGGTFeatureExtractor

ACTION_LIMIT = 10
RESET_EVERY_STEP = False
THRESHOLD = 0.005
NEAREST_NEIGHBOR_RADIUS_METERS = 0.001
debug = Debugger(__file__)
USE_ONLY_STOP = False


def prefill_replay_buffer(
    buffer: ReplayBuffer,
    environment: HabitatObjectNavEnv,
    extractor: JAXVGGTFeatureExtractor,
    seed: int,
    prefill_steps: int,
):
    """Warm up replay with random environment transitions.

    Args:
        buffer: Replay buffer that receives copied ``ReplayTransition`` objects.
        environment: Habitat ObjectNav environment used for random rollout.
        extractor: VGGT feature extractor used to convert frames to replay obs.
        rng_key: JAX PRNG key with shape ``(2,)`` and dtype ``uint32``.
        prefill_steps: Number of random transitions to append.
        num_actions: Number of discrete actions sampled from ``[0, num_actions)``.

    Side effects:
        Resets and steps ``environment``, resets ``extractor`` at episode starts,
        and mutates ``buffer`` by appending ``prefill_steps`` transitions.
    """
    if prefill_steps <= 0:
        raise ValueError(f"prefill_steps must be positive, got {prefill_steps}")

    current_frame = environment.reset()
    extractor.reset()
    random_agent = RandomAgent(environment, seed=seed)

    for _ in range(prefill_steps):
        current_obs = random_agent.act()

        current_features = extractor.extract(current_frame)
        current_buffer_obs: ObservationInput = {
            "LINEAR": jnp.asarray(current_features.camera_pose)
        }
        buffer.add(ReplayTransition.from_frame(current_buffer_obs, current_obs))

        if current_obs.done:
            current_frame = environment.reset()
            extractor.reset()


def main() -> None:
    """Run the prototype loop and emit observation/point-cloud diagnostics."""
    trainer_config = TrainerConfig()
    environment_config = HabitatEnvConfig(max_episode_steps=50, curriculum="L1")
    habitat_environment = HabitatObjectNavEnv(environment_config)
    extractor = JAXVGGTFeatureExtractor()

    video_path = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "prototype_live_vggt"
        / "observations.mp4"
    )
    video_frames: list[np.ndarray] = []
    change_plotter = PointChangePlotter()

    buffer = ReplayBuffer(
        trainer_config.buffer_capacity, habitat_environment.num_actions
    )

    r2dreamer_config = R2DreamerConfig()
    rng_key = jax.random.PRNGKey(trainer_config.seed)

    r2_agent = R2DreamerAgent(r2dreamer_config, rng_key)
    prefill_replay_buffer(
        buffer,
        habitat_environment,
        extractor,
        trainer_config.seed,
        prefill_steps=trainer_config.prefill_steps,
    )
    obs = habitat_environment.reset()
    house_buffer = HouseContextPoseBuffer(confidence_score=5, scene_id=obs.scene_id)
    extractor.reset()
    encoder_obs: ObservationInput = {"CNN": extractor.extract(obs).camera_pose}
    for step_id in range(ACTION_LIMIT):
        debug(f"step {step_id}/{ACTION_LIMIT}: start")

        if USE_ONLY_STOP:
            obs = habitat_environment.step(0)
        else:
            replay_batch = buffer.sample(
                seq_len=trainer_config.seq_len, batch_size=trainer_config.batch_size
            )
            rng_key, train_key = jax.random.split(rng_key)
            metrics = r2_agent.train_step(replay_batch, train_key)
            rng_key, act_key = jax.random.split(rng_key)
            action = r2_agent.act(encoder_obs, obs.is_first, act_key)
            obs = habitat_environment.step(action)

            debug(f"step {step_id}: env action={action} ({ACTIONS[action]})")

        video_frames.append(np.asarray(obs.image))
        debug(f"step {step_id}: extractor start")
        if RESET_EVERY_STEP:
            extractor.reset()
        features = extractor.extract(obs)
        debug(
            f"step {step_id}: extractor done world_points={features.world_points.shape}"
        )
        points = features.world_points.reshape(-1, 3)  # Reshape (H, W, 3) -> (N, 3).
        house_context = house_buffer.add(features, obs)
        obs_input: ObservationInput = {"CNN": features.camera_pose}
        replay = ReplayTransition.from_frame(obs_input, obs)
        buffer.add(replay)
        encoder_obs = obs_input

        # TODO Call the buffer
        #

        if obs.done:
            reset_obs = habitat_environment.reset()
            extractor.reset()
            print("new episode:", reset_obs.scene_id, reset_obs.episode_id)
        # Re-encode every step (pre-existing behavior): a crash mid-run still
        # leaves a playable partial video on disk.
        write_frames_mp4(video_frames, video_path, fps=10)


if __name__ == "__main__":
    main()
