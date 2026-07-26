"""Tests for src/r2dreamer/checkpointing.py and the from_checkpoint guard.

A checkpoint carries parameters only - no architecture description. The encoder
is rebuilt from one live adapter call (``fields=``) and
``R2DreamerAgent._assert_params_match`` compares that fresh param tree against
the loaded one, so a config or adapter that drifted since the checkpoint was
written fails with a named path instead of a shape error inside a jitted apply.
"""

import json
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.adapters.contract import AdapterField, AdapterOutput, Encoder
from src.configs.config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.checkpointing import (
    config_snapshot,
    load_checkpoint,
    save_checkpoint,
)

IMAGE_SHAPE = (64, 64, 3)

# Architecture fields ``from_checkpoint`` recovers from the run manifest. The
# rebuilt agent must match the saved one on all of them or the param-tree guard
# fires - which is exactly what the mismatch tests below provoke.
ARCH_FIELDS = (
    "adapter",
    "deter_size",
    "hidden_size",
    "stoch_classes",
    "stoch_discrete",
    "blocks",
    "encoder_depth",
    "encoder_kernel",
    "encoder_mults",
    "mlp_units",
    "mlp_layers_reward",
    "mlp_layers_cont",
    "mlp_layers_actor",
    "mlp_layers_critic",
    "twohot_bins",
)


def _cfg(**overrides) -> R2DreamerConfig:
    params: dict[str, object] = dict(
        adapter="rgb",
        num_actions=4,
        deter_size=32,
        hidden_size=16,
        stoch_classes=4,
        stoch_discrete=4,
        blocks=4,
        encoder_depth=4,
        encoder_kernel=3,
        encoder_mults=(1, 1, 1, 1),
        mlp_units=16,
        mlp_layers_reward=1,
        mlp_layers_cont=1,
        mlp_layers_actor=1,
        mlp_layers_critic=1,
        twohot_bins=21,
        imagination_horizon=2,
        horizon=10,
        lr=1e-3,
        warmup_steps=0,
    )
    params.update(overrides)
    return R2DreamerConfig(**params)


def _arch_kwargs(**overrides) -> dict[str, object]:
    cfg = _cfg()
    kwargs = {name: getattr(cfg, name) for name in ARCH_FIELDS}
    kwargs.update(overrides)
    return kwargs


def _rgb_fields() -> AdapterOutput:
    return [
        AdapterField(
            key="image",
            encoder=Encoder.CONV,
            buffer=True,
            value=jnp.zeros(IMAGE_SHAPE, jnp.uint8),
            decoder_target=True,
        )
    ]


def _fields_with_extra_branch() -> AdapterOutput:
    return [
        *_rgb_fields(),
        AdapterField(
            key="camera_pose",
            encoder=Encoder.MLP,
            buffer=True,
            value=jnp.zeros((9,), jnp.float32),
        ),
    ]


def _agent(cfg: R2DreamerConfig, fields: AdapterOutput, *, seed: int = 42):
    return R2DreamerAgent(cfg, jax.random.PRNGKey(seed), fields=fields)


class TestCheckpointRoundTrip:
    """save_checkpoint and load_checkpoint round-trip agent state."""

    @pytest.fixture
    def agent(self):
        return _agent(_cfg(), _rgb_fields())

    def test_roundtrip_preserves_params(self, agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=100, output_dir=tmpdir)
            assert os.path.exists(path)

            data = load_checkpoint(path)
            assert data["step"] == 100
            for key in agent.params:
                jax.tree.map(
                    lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6),
                    agent.params[key],
                    data["params"][key],
                )

    def test_roundtrip_preserves_ema_state(self, agent):
        """The old _save_checkpoint missed ema_state — verify it's saved now."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=50, output_dir=tmpdir)
            data = load_checkpoint(path)
            assert "ema_state" in data
            np.testing.assert_allclose(
                data["ema_state"], np.array(agent.ema_state), atol=1e-6
            )

    def test_roundtrip_preserves_slow_critic(self, agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(agent, step=10, output_dir=tmpdir)
            data = load_checkpoint(path)
            jax.tree.map(
                lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6),
                agent.slow_critic_params,
                data["slow_critic_params"],
            )

    def test_checkpoint_carries_no_architecture_description(self, agent):
        # The architecture is rebuilt from a live adapter call at load time, so
        # nothing describing the encoder is written to disk.
        with tempfile.TemporaryDirectory() as tmpdir:
            data = load_checkpoint(save_checkpoint(agent, step=10, output_dir=tmpdir))

        assert set(data) == {
            "step",
            "params",
            "opt_state",
            "slow_critic_params",
            "ema_state",
        }


class TestFromCheckpoint:
    """The encoder comes from ``fields``; only the parameters come from disk."""

    def test_loads_params_and_step_for_a_matching_architecture(self, tmp_path):
        fields = _rgb_fields()
        original = _agent(_cfg(), fields, seed=1)
        path = save_checkpoint(original, step=4321, output_dir=str(tmp_path))

        # A different seed, so a failure to load shows up as differing weights.
        loaded = R2DreamerAgent.from_checkpoint(
            path, num_actions=4, seed=99, fields=fields, **_arch_kwargs()
        )

        assert loaded.checkpoint_step == 4321
        for tree_a, tree_b in (
            (loaded.params, original.params),
            (loaded.slow_critic_params, original.slow_critic_params),
        ):
            for a, b in zip(jax.tree.leaves(tree_a), jax.tree.leaves(tree_b)):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6)

    def test_extra_routed_branch_is_rejected_by_the_param_tree_guard(self, tmp_path):
        # An adapter that grew a field composes an extra branch, so the fresh
        # tree has params the checkpoint never stored.
        path = save_checkpoint(
            _agent(_cfg(), _rgb_fields()), step=10, output_dir=str(tmp_path)
        )

        with pytest.raises(ValueError, match="do not match the rebuilt architecture"):
            R2DreamerAgent.from_checkpoint(
                path,
                num_actions=4,
                seed=0,
                fields=_fields_with_extra_branch(),
                **_arch_kwargs(),
            )

    def test_changed_branch_width_is_rejected_by_the_param_tree_guard(self, tmp_path):
        # Same routing, wider conv branch: the tree structure still matches, so
        # only the per-leaf shape comparison catches it.
        path = save_checkpoint(
            _agent(_cfg(), _rgb_fields()), step=10, output_dir=str(tmp_path)
        )

        with pytest.raises(ValueError, match="param shape mismatch"):
            R2DreamerAgent.from_checkpoint(
                path,
                num_actions=4,
                seed=0,
                fields=_rgb_fields(),
                **_arch_kwargs(encoder_depth=8),
            )


class TestConfigSnapshot:
    def test_snapshot_is_a_json_serializable_dataclass_dump(self):
        cfg = _cfg()

        snapshot = config_snapshot(cfg)

        assert snapshot["adapter"] == "rgb"
        assert snapshot["encoder_depth"] == cfg.encoder_depth
        json.dumps(snapshot)

    def test_snapshot_carries_no_encoder_contract_keys(self):
        # Both the contract snapshot and the encoder-module path are gone; the
        # adapter name is the only variant provenance the manifest needs.
        snapshot = config_snapshot(_cfg())

        assert "encoder_input_contract" not in snapshot
        assert "encoder_module" not in snapshot
        assert "encoder_module_cls" not in snapshot
        assert "encoder_type" not in snapshot
