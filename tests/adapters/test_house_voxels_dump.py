"""House-map PLY dumps: the schedule, the artifacts, and who may ask for them.

The dump is a diagnostic - the accumulated map is only judged by eye - so what
is worth pinning is that it writes when asked, stays silent when not, and that
asking the wrong variant fails loudly instead of producing nothing.
"""

from __future__ import annotations

import pytest

from src.adapters import ADAPTERS
from src.adapters.house_voxels import DUMP_SUBDIR
from src.adapters.rgb import RgbAdapter
from src.main import _adapter_kwargs, make_adapter
from src.launch.parser import build_parser

from tests.adapters.conftest import FakeEnv, FakeExtractor

RESOLUTION = 32
HOUSE_ADAPTERS = ("rgb_house_voxels", "rgb_house_voxels_gnn")
PLY = "step_00000_context.ply"


def small_buffer_variant(name: str) -> type:
    """Return the registered variant with a test-sized voxel store.

    The production store is a 2^24-slot hash table (~300 MB): correct for a
    whole house, wasteful for the handful of 32x32 frames below. Only the
    sizing constants change, so the dump path under test is the real one.
    """
    return type(
        f"Small{ADAPTERS[name].__name__}",
        (ADAPTERS[name],),
        {"BUFFER_CAPACITY": 1 << 14, "BUFFER_HASH_TABLE_SIZE": 1 << 15},
    )


def roll(adapter, env: FakeEnv, steps: int) -> None:
    """Drive ``steps`` env steps through the adapter the way the collector does.

    Reset frames go through the adapter too (that is where episode boundaries
    become visible to it), and a finished episode is followed by a reset.
    """
    adapter(env.reset())
    for _ in range(steps):
        frame = env.step(0)
        adapter(frame)
        if frame.done:
            adapter(env.reset())


def dumped(root) -> list[str]:
    """Return the snapshot labels written under ``root``, sorted."""
    dump_root = root / DUMP_SUBDIR
    if not dump_root.exists():
        return []
    return sorted(path.name for path in dump_root.iterdir())


@pytest.mark.parametrize("name", HOUSE_ADAPTERS)
def test_dumping_is_off_unless_a_schedule_is_given(name, fake_extractor, tmp_path):
    """A run directory alone must not turn a training run into a PLY writer."""
    adapter = small_buffer_variant(name)(fake_extractor, output_dir=str(tmp_path))

    roll(adapter, FakeEnv(resolution=RESOLUTION, episode_len=3), steps=8)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", HOUSE_ADAPTERS)
def test_scheduled_steps_write_one_ply_per_scene(name, fake_extractor, tmp_path):
    """Both cloud branches inherit the dump; the schedule counts adapter steps."""
    adapter = small_buffer_variant(name)(
        fake_extractor, pointcloud_dump_steps="2,4", output_dir=str(tmp_path)
    )

    # Long episode: no boundary, so only the schedule can produce output.
    roll(adapter, FakeEnv(resolution=RESOLUTION, episode_len=99), steps=6)

    assert dumped(tmp_path) == ["step_000000002", "step_000000004"]
    for label in dumped(tmp_path):
        assert (tmp_path / DUMP_SUBDIR / label / "scene-a" / PLY).is_file()


def test_the_first_episode_is_dumped_when_it_ends(fake_extractor, tmp_path):
    """The map after one episode is the reference for judging later growth."""
    adapter = small_buffer_variant("rgb_house_voxels")(
        fake_extractor,
        # Far beyond the rollout: whatever lands must come from the boundary.
        pointcloud_dump_steps="10000",
        output_dir=str(tmp_path),
    )

    roll(adapter, FakeEnv(resolution=RESOLUTION, episode_len=3), steps=8)

    assert dumped(tmp_path) == ["end_of_first_episode"]
    assert (tmp_path / DUMP_SUBDIR / "end_of_first_episode" / "scene-a" / PLY).is_file()


def test_a_schedule_without_a_run_directory_is_refused(fake_extractor):
    """Failing at construction beats discovering the empty directory later."""
    with pytest.raises(ValueError, match="run directory"):
        small_buffer_variant("rgb_house_voxels")(
            fake_extractor, pointcloud_dump_steps="2"
        )


def test_a_variant_that_claims_no_knob_still_builds():
    """The generic wiring must leave every other variant exactly as it was."""
    args = build_parser().parse_args([])

    adapter = make_adapter(RgbAdapter, args)

    assert isinstance(adapter, RgbAdapter)


def test_asking_an_unclaiming_variant_to_dump_is_an_error():
    """Silently ignoring the flag would waste the cluster job it was set for."""
    args = build_parser().parse_args(["--pointcloud_dump_steps", "2,4"])

    with pytest.raises(ValueError, match="pointcloud_dump_steps"):
        make_adapter(RgbAdapter, args)


def test_the_dump_uses_the_run_directory_not_the_raw_flag():
    """A preset launch names its run directory as a kwarg, never as --output_dir."""
    args = build_parser().parse_args(
        ["--pointcloud_dump_steps", "2,4", "--output_dir", "output/ignored"]
    )

    kwargs = _adapter_kwargs(
        ADAPTERS["rgb_house_voxels"], args, output_dir="output/runs/actual"
    )

    assert kwargs["output_dir"] == "output/runs/actual"


def test_a_rollout_that_owns_no_artifacts_gets_no_run_directory():
    """The validation collector must not write over the training run's dumps."""
    args = build_parser().parse_args(["--output_dir", "output/runs/actual"])

    kwargs = _adapter_kwargs(ADAPTERS["rgb_house_voxels"], args, output_dir=None)

    assert "output_dir" not in kwargs


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_claimed_flag_exists_on_the_train_cli(name):
    """A claim on a deleted flag is a job that dies at argparse on the cluster."""
    known = set(build_parser().parse_args([]).__dict__)

    assert not set(getattr(ADAPTERS[name], "RUN_FLAGS", ())) - known
