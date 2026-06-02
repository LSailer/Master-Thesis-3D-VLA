from src.r2dreamer.launch.evaluate import _find_manifest_for_checkpoint


def test_find_manifest_next_to_checkpoint(tmp_path):
    ckpt = tmp_path / "step_000000010.pkl"
    manifest = tmp_path / "MANIFEST.json"
    ckpt.touch()
    manifest.write_text('{"config": {}}')

    assert _find_manifest_for_checkpoint(ckpt) == manifest.resolve()


def test_find_manifest_in_run_dir_for_checkpoints_subdir(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "step_000000010.pkl"
    manifest = tmp_path / "MANIFEST.json"
    ckpt.touch()
    manifest.write_text('{"config": {}}')

    assert _find_manifest_for_checkpoint(ckpt) == manifest.resolve()
