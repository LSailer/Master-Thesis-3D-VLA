
"""Tests for video frame composition and W&B episode logging."""
import numpy as np

from src.shared.video_utils import (
    MAX_VIDEO_FRAMES,
    compose_frame,
    log_episode_video,
    render_topdown_frame,
)


class MockHabitatEnv:
    def sample_navmesh(self, resolution=0.1):
        return {
            "grid": np.ones((8, 8), dtype=bool),
            "x_min": -1.0,
            "x_max": 1.0,
            "z_min": -1.0,
            "z_max": 1.0,
            "resolution": resolution,
        }


class MockVideo:
    def __init__(self, data, fps, format):
        self.data = data
        self.fps = fps
        self.format = format


class MockWandb:
    Video = MockVideo

    def __init__(self):
        self.logged = []

    def log(self, payload, step=None):
        self.logged.append((payload, step))


def test_compose_frame_returns_side_by_side_uint8():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    topdown = np.full((32, 32, 3), 255, dtype=np.uint8)

    frame = compose_frame(rgb, topdown)

    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[-1] == 3
    assert frame.shape[0] <= 256
    assert frame.shape[1] <= 512


def test_render_topdown_frame_returns_rgb_uint8():
    frame = render_topdown_frame(
        MockHabitatEnv(),
        trajectory_so_far=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.25]],
        goal_positions=[[0.75, 0.0, 0.75]],
    )

    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[-1] == 3


def test_log_episode_video_subsamples_and_logs_wandb_video():
    wandb = MockWandb()
    frames = [
        np.full((16, 16, 3), value % 256, dtype=np.uint8)
        for value in range(MAX_VIDEO_FRAMES + 25)
    ]

    video = log_episode_video(wandb, "eval/episode_video_0", frames, step=12, fps=5)

    assert isinstance(video, MockVideo)
    assert video.data.shape[0] <= MAX_VIDEO_FRAMES
    assert video.data.shape[1:] == (3, 16, 16)
    assert video.fps == 5
    assert video.format == "mp4"
    assert wandb.logged == [({"eval/episode_video_0": video}, 12)]
