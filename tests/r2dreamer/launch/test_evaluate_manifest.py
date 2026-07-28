"""Tests for evaluate-launcher manifest discovery, arch overrides, and the loop."""

import json
from dataclasses import replace

import jax

from src.adapters.contract import AdapterField, Encoder

from src.configs.agent_config import R2DreamerConfig
from src.r2dreamer.agent import encoder_overrides_from_config
from src.r2dreamer.checkpointing import config_snapshot
from src.r2dreamer.encoders.routed_composite import routed_encoder_from_fields
from src.launch.eval_artifacts import (
    arch_overrides_from_manifest,
    find_manifest_for_checkpoint,
)


def test_find_manifest_next_to_checkpoint(tmp_path):
    ckpt = tmp_path / "step_000000010.pkl"
    manifest = tmp_path / "MANIFEST.json"
    ckpt.touch()
    manifest.write_text('{"config": {}}')

    assert find_manifest_for_checkpoint(ckpt) == manifest.resolve()


def test_find_manifest_in_run_dir_for_checkpoints_subdir(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "step_000000010.pkl"
    manifest = tmp_path / "MANIFEST.json"
    ckpt.touch()
    manifest.write_text('{"config": {}}')

    assert find_manifest_for_checkpoint(ckpt) == manifest.resolve()


def _write_manifest(tmp_path, config: dict):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "step_000000010.pkl"
    ckpt.touch()
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"config": config}))
    return str(ckpt)


def test_arch_overrides_recovers_rssm_and_head_widths(tmp_path):
    """The routing comes from the adapter; widths must come from the manifest."""
    ckpt = _write_manifest(
        tmp_path,
        {
            "deter_size": 512,
            "encoder_mults": [2, 3, 4],
            "twohot_bins": 41,
            "adapter": "rgb",
            "buffer_capacity": 123,  # not an arch field
        },
    )

    overrides = arch_overrides_from_manifest(ckpt)

    assert overrides["deter_size"] == 512
    assert overrides["encoder_mults"] == (2, 3, 4)  # tuple, as Flax needs
    assert overrides["twohot_bins"] == 41
    assert "buffer_capacity" not in overrides
    assert "adapter" not in overrides


def test_mlp_branch_depth_round_trips_through_the_manifest(tmp_path):
    """A swept branch depth must survive to eval, or the run is unevaluatable.

    ``--mlp_layers`` used to reach the encoder as a bare CLI override that no
    artifact recorded, so evaluation rebuilt the branch at the default depth and
    ``_assert_params_match`` rejected the checkpoint with nothing on disk to
    recover the value from.
    """
    trained = R2DreamerConfig(mlp_layers=3)
    ckpt = _write_manifest(tmp_path, config_snapshot(trained))

    overrides = arch_overrides_from_manifest(ckpt)
    at_eval = replace(R2DreamerConfig(), **overrides)

    assert overrides["mlp_layers"] == 3
    assert at_eval.mlp_layers == 3
    assert encoder_overrides_from_config(at_eval) == encoder_overrides_from_config(
        trained
    )


def test_compute_dtype_gate_round_trips_through_the_manifest(tmp_path):
    """The precision gate changes compute, not shapes, so nothing else catches it.

    ``full_bf16`` leaves the param tree bit-identical, so ``_assert_params_match``
    passes whether or not it is recovered; a run trained under the gate would
    silently be evaluated in float32.
    """
    trained = R2DreamerConfig(full_bf16=True, compute_dtype="bfloat16")
    ckpt = _write_manifest(tmp_path, config_snapshot(trained))

    overrides = arch_overrides_from_manifest(ckpt)
    at_eval = replace(R2DreamerConfig(), **overrides)

    assert at_eval.full_bf16 is True
    assert encoder_overrides_from_config(at_eval) == encoder_overrides_from_config(
        trained
    )


def test_a_deeper_mlp_branch_really_builds_more_params(tmp_path):
    """Guards the test above: the depth must be observable in the param tree."""
    fields = [
        AdapterField(
            key="pose",
            encoder=Encoder.MLP,
            buffer=True,
            value=jax.numpy.zeros((8,)),
        )
    ]
    obs = {"pose": jax.numpy.zeros((1, 8))}

    def params_for(cfg):
        encoder = routed_encoder_from_fields(
            fields, **encoder_overrides_from_config(cfg)
        )
        return jax.tree_util.tree_structure(
            encoder.init(jax.random.PRNGKey(0), obs)
        )

    deep = params_for(R2DreamerConfig(mlp_layers=3))

    assert params_for(R2DreamerConfig(mlp_layers=1)) != deep
    assert params_for(replace(R2DreamerConfig(), mlp_layers=3)) == deep


def test_arch_overrides_without_manifest_is_empty(tmp_path):
    ckpt = tmp_path / "step_000000010.pkl"
    ckpt.touch()

    assert arch_overrides_from_manifest(str(ckpt)) == {}
    assert arch_overrides_from_manifest(None) == {}
