"""Habitat ObjectNav env adapter for PyTorch r2dreamer's interface."""
import math

import numpy as np


_AGENT_STATE_DIM = 25  # 16 extrinsics + 9 intrinsics, row-major float32


def build_agent_state(
    position: np.ndarray,
    quat_xyzw: np.ndarray,
    hfov_deg: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Pack camera extrinsics + intrinsics into a flat 25-D float32 vector.

    Habitat-sim uses a Y-up, right-handed world frame and the OpenGL camera
    convention (camera looks down -Z). Downstream Plücker math must respect
    that or rays will point the wrong way.

    Layout (row-major):
        bytes  0..15: 4x4 camera-to-world extrinsics
        bytes 16..24: 3x3 intrinsics K (fx, fy, cx, cy from hfov + image size)
    """
    qx, qy, qz, qw = (float(quat_xyzw[0]), float(quat_xyzw[1]),
                      float(quat_xyzw[2]), float(quat_xyzw[3]))
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        raise ValueError("quaternion has zero norm")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)

    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = np.asarray(position, dtype=np.float32)

    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx  # square pixels in Habitat
    cx = width / 2.0
    cy = height / 2.0
    intrinsics = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return np.concatenate([extrinsics.reshape(-1), intrinsics.reshape(-1)]).astype(np.float32)


class HabitatR2DreamerEnv:
    def __init__(self, obs_size=64, split="train", max_episode_steps=500,
                 max_geodesic=None, reward_type="geodesic_delta"):
        from modules.shared.configs import DreamerConfig
        from modules.envs.habitat import HabitatObjectNavEnv
        config = DreamerConfig(
            obs_shape=(3, obs_size, obs_size),
            max_episode_steps=max_episode_steps,
            split=split, reward_type=reward_type)
        self._env = HabitatObjectNavEnv(config, max_geodesic=max_geodesic)
        self.num_actions = 4
        self._H = obs_size
        self._W = obs_size

    def _agent_state(self) -> np.ndarray:
        sim = self._env._env.sim
        agent_state = sim.get_agent_state()
        # The RGB sensor's pose is the camera pose; agent_state.position is
        # the agent root, but for ObjectNav the RGB sensor sits at the agent.
        sensor_state = agent_state.sensor_states.get("rgb", agent_state)
        position = np.asarray(sensor_state.position, dtype=np.float32)
        rotation = sensor_state.rotation
        # habitat_sim quaternion: .x .y .z .w
        quat_xyzw = np.array(
            [rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float32)
        # hfov is set per-sensor; default 90 deg if introspection fails
        try:
            hfov_deg = float(sim._sensors["rgb"].specification().hfov)
        except (AttributeError, KeyError):
            hfov_deg = 90.0
        return build_agent_state(position, quat_xyzw, hfov_deg, self._H, self._W)

    def reset(self):
        obs = self._env.reset()
        image = np.transpose(obs["image"], (1, 2, 0))  # CHW->HWC
        return {"image": image, "reward": np.float32(0.0),
                "is_first": True, "is_last": False, "is_terminal": False,
                "agent_state": self._agent_state()}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(np.argmax(action))
        obs = self._env.step(action)
        image = np.transpose(obs["image"], (1, 2, 0))
        done = obs["done"]
        success = obs.get("success", 0.0) > 0
        return {"image": image, "reward": np.float32(obs["reward"]),
                "is_first": False, "is_last": done, "is_terminal": success,
                "success": obs.get("success", 0.0),
                "spl": obs.get("spl", 0.0),
                "agent_state": self._agent_state()}

    def close(self):
        self._env.close()
