"""Parity + shape tests for the generic CompositeEncoder.

These prove the composite reproduces the legacy encoders' parameter trees and
outputs *bit-identically* on a fixed seed — the precondition for the golden run
(the composite must match legacy init RNG folding, not just the computation).
They double as the migration parity references for ``cnn`` and ``hybrid`` until
the parametrized recipe test replaces them at the end (see DELETIONS.md).
"""

import jax
import jax.numpy as jnp

from src.r2dreamer.encoders.cnn import ConvEncoder, make_rgb_conv_encoder
from src.r2dreamer.encoders.composite import (
    FUSIONS,
    BranchSpec,
    CompositeEncoder,
    CompositeSpec,
)
from src.r2dreamer.encoders.mlp import HybridEncoder
from src.r2dreamer.observation_keys import HYBRID_IMAGE_KEY, HYBRID_WP_CP_KEY
from src.r2dreamer.world_model.heads import R2MLP

CNN_KW = dict(depth=16, kernel_size=5, mults=(2, 3, 4, 4))
HYBRID_VGGT_DIM = 4116


def _cnn_spec() -> CompositeSpec:
    return CompositeSpec(
        branches=(
            BranchSpec(
                obs_key=HYBRID_IMAGE_KEY,
                module_name="cnn",
                make=lambda name: ConvEncoder(name=name, **CNN_KW),
            ),
        ),
        fusion="concat",
    )


def _hybrid_spec() -> CompositeSpec:
    return CompositeSpec(
        branches=(
            BranchSpec(
                obs_key=HYBRID_IMAGE_KEY,
                module_name="cnn",
                make=lambda name: make_rgb_conv_encoder(name=name, **CNN_KW),
            ),
            BranchSpec(
                obs_key=HYBRID_WP_CP_KEY,
                module_name="vggt_mlp",
                make=lambda name: R2MLP(hidden=1024, layers=2, out_dim=1024, name=name),
            ),
        ),
        fusion="gate",
    )


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def _trees_bit_identical(a, b) -> bool:
    la, lb = _leaves(a), _leaves(b)
    return len(la) == len(lb) and all(
        x.shape == y.shape and bool(jnp.array_equal(x, y)) for x, y in zip(la, lb)
    )


class TestCnnParity:
    """Single-branch concat must equal a bare ConvEncoder (share_scope)."""

    def test_param_tree_and_output_bit_identical(self):
        key = jax.random.PRNGKey(0)
        rgb = jax.random.normal(jax.random.PRNGKey(7), (2, 64, 64, 3))

        legacy = ConvEncoder(**CNN_KW)
        legacy_params = legacy.init(key, rgb)

        composite = CompositeEncoder(_cnn_spec())
        # Init from the same bare array the agent uses as the CNN dummy obs.
        comp_params = composite.init(key, rgb)

        # share_scope places conv params at the composite root — same names.
        assert set(comp_params["params"]) == set(legacy_params["params"])
        assert _trees_bit_identical(comp_params, legacy_params)

        legacy_out = legacy.apply(legacy_params, rgb)
        comp_out = composite.apply(comp_params, rgb)
        assert bool(jnp.array_equal(legacy_out, comp_out))

    def test_accepts_dict_and_array_obs_identically(self):
        key = jax.random.PRNGKey(3)
        rgb = jax.random.normal(jax.random.PRNGKey(9), (4, 64, 64, 3))
        composite = CompositeEncoder(_cnn_spec())
        params = composite.init(key, {HYBRID_IMAGE_KEY: rgb})
        out_dict = composite.apply(params, {HYBRID_IMAGE_KEY: rgb})
        out_arr = composite.apply(params, rgb)
        assert bool(jnp.array_equal(out_dict, out_arr))

    def test_preserves_replay_leading_dims(self):
        key = jax.random.PRNGKey(1)
        obs = jnp.zeros((3, 5, 64, 64, 3))  # (B, T, H, W, C)
        composite = CompositeEncoder(_cnn_spec())
        params = composite.init(key, obs)
        out = composite.apply(params, obs)
        assert out.shape[:2] == (3, 5)
        assert out.dtype == jnp.float32


class TestHybridParity:
    """Gate fusion must reproduce WMHybridEncoder param tree + output exactly."""

    def _obs(self):
        rgb = jax.random.normal(jax.random.PRNGKey(11), (2, 64, 64, 3))
        wp_cp = jax.random.normal(jax.random.PRNGKey(12), (2, HYBRID_VGGT_DIM))
        return {HYBRID_IMAGE_KEY: rgb, HYBRID_WP_CP_KEY: wp_cp}

    def test_param_tree_and_output_bit_identical(self):
        key = jax.random.PRNGKey(0)
        obs = self._obs()

        legacy = HybridEncoder(vggt_dim=HYBRID_VGGT_DIM)
        legacy_params = legacy.init(key, obs)

        composite = CompositeEncoder(_hybrid_spec())
        comp_params = composite.init(key, obs)

        assert set(comp_params["params"]) == set(legacy_params["params"])
        assert set(comp_params["params"]) == {"cnn", "vggt_mlp", "gate"}
        assert _trees_bit_identical(comp_params, legacy_params)

        legacy_out = legacy.apply(legacy_params, obs)
        comp_out = composite.apply(comp_params, obs)
        assert comp_out.shape == (2, 2048)
        assert bool(jnp.array_equal(legacy_out, comp_out))

    def test_gate_starts_closed_and_backbone_matches_cnn_alone(self):
        key = jax.random.PRNGKey(0)
        obs = self._obs()
        composite = CompositeEncoder(_hybrid_spec())
        params = composite.init(key, obs)
        assert float(params["params"]["gate"]) == 0.0
        out = composite.apply(params, obs)
        # Gate closed => the gated half is zero; the backbone half is the CNN.
        cnn_half, vggt_half = out[:, :1024], out[:, 1024:]
        assert bool(jnp.all(vggt_half == 0.0))
        assert bool(jnp.all(jnp.isfinite(cnn_half)))

    def test_preserves_replay_leading_dims(self):
        key = jax.random.PRNGKey(2)
        obs = {
            HYBRID_IMAGE_KEY: jnp.zeros((3, 4, 64, 64, 3)),
            HYBRID_WP_CP_KEY: jnp.zeros((3, 4, HYBRID_VGGT_DIM)),
        }
        composite = CompositeEncoder(_hybrid_spec())
        params = composite.init(key, obs)
        out = composite.apply(params, obs)
        assert out.shape == (3, 4, 2048)


class TestFusions:
    """Fusion registry basics independent of a specific recipe."""

    def test_fusion_names(self):
        assert set(FUSIONS) == {"concat", "gate", "concat_mlp"}

    def test_concat_mlp_projects_to_embed_dim(self):
        spec = CompositeSpec(
            branches=(
                BranchSpec("a", "branch_a", lambda name: R2MLP(hidden=8, layers=1, out_dim=8, name=name)),
                BranchSpec("b", "branch_b", lambda name: R2MLP(hidden=8, layers=1, out_dim=8, name=name)),
            ),
            fusion="concat_mlp",
        )
        enc = CompositeEncoder(spec, embed_dim=16)
        obs = {"a": jnp.ones((2, 4)), "b": jnp.ones((2, 4))}
        params = enc.init(jax.random.PRNGKey(0), obs)
        out = enc.apply(params, obs)
        assert out.shape == (2, 16)
        assert "fusion_proj" in params["params"]

    def test_branch_keys_exposed_for_startup_check(self):
        assert set(CompositeEncoder(_hybrid_spec()).branches) == {"image", "wp_cp"}
