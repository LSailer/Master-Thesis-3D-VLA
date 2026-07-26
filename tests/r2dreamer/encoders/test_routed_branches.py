"""CPU tests for the branches ``RoutedCompositeEncoder`` composes.

The routed encoder replaced the encoder-type string dispatch, so the branch
architectures are now the only encoder surface left: one branch per
:class:`~src.adapters.contract.Encoder` member plus the fusion rules that turn
several branch embeddings into one RSSM input.

This file keeps the branch-level math that outlived the deleted per-encoder-type
test modules: the PointNet and k-NN-GCN cloud branches (T-Net identity init,
projection rule, gradient flow, optimization stability), the token Transformer's
shape contract, ``ConvEncoder`` normalization, ``MLPEncoder`` depth, and the
``ConvDecoder`` probe's output range. Everything runs on CPU against small
arrays; the per-variant end-to-end wiring lives in
``tests/adapters/test_routed_pipeline.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from src.adapters.contract import AdapterField, Encoder
from src.r2dreamer.encoders.cnn import ConvEncoder
from src.r2dreamer.encoders.decoder import ConvDecoder
from src.r2dreamer.encoders.gnn import GnnCloudEncoder
from src.r2dreamer.encoders.mlp import MLPEncoder
from src.r2dreamer.encoders.pointnet import PointNetCloudEncoder
from src.r2dreamer.encoders.routed_composite import (
    RoutedCompositeEncoder,
    routed_encoder_from_fields,
    routes_from_fields,
)
from src.r2dreamer.encoders.transformer import (
    TokenSequenceEncoder,
    TokenTransformerEncoder,
)

POINT_DIM = 6
# Small everywhere: the branch math under test is size-independent, and a CPU
# run of the full production widths (16384 points, 4096 graph nodes) is minutes.
CLOUD_ROWS = 128
EMBED_DIM = 16
NUM_TOKENS = 8
TOKEN_DIM = 8


@pytest.fixture(name="rng")
def rng_fixture() -> jax.Array:
    """A fixed PRNG key, so every assertion below is deterministic."""
    return jax.random.PRNGKey(0)


def _cloud(rows: int = CLOUD_ROWS, *, key: int = 1) -> jnp.ndarray:
    """A spatially varying ``(rows, 6)`` XYZRGB cloud in float16, as adapters emit."""
    return jax.random.uniform(
        jax.random.PRNGKey(key), (rows, POINT_DIM), dtype=jnp.float32
    ).astype(jnp.float16)


def _pointnet(**overrides) -> PointNetCloudEncoder:
    kwargs: dict[str, object] = {
        "num_points": 64,
        "embed_dim": EMBED_DIM,
        "tnet_mlp": (8, 16, 32),
        "tnet_fc": (16, 8),
        "mlp1": (8, 8),
        "mlp2": (8, 16, EMBED_DIM),
    }
    kwargs.update(overrides)
    return PointNetCloudEncoder(**kwargs)


def _gnn(**overrides) -> GnnCloudEncoder:
    kwargs: dict[str, object] = {
        "num_graph_nodes": 64,
        "knn_k": 4,
        "gcn_hidden": 16,
        "gcn_layers": 2,
        "embed_dim": EMBED_DIM,
    }
    kwargs.update(overrides)
    return GnnCloudEncoder(**kwargs)


# --------------------------------------------------------------------------- #
# PointNet cloud branch                                                        #
# --------------------------------------------------------------------------- #


class TestPointNetCloudEncoder:
    """Classic PointNet over one unbatched live cloud."""

    def test_encodes_one_cloud_to_one_vector(self, rng):
        branch = _pointnet()
        cloud = _cloud()
        params = branch.init(rng, cloud)

        embed = branch.apply(params, cloud)

        assert embed.shape == (EMBED_DIM,)
        assert bool(jnp.isfinite(jnp.asarray(embed, jnp.float32)).all())

    def test_subsample_budget_above_the_cloud_size_is_clamped(self, rng):
        # A run's cloud can hold fewer rows than the branch's budget; the
        # even-stride index must stay inside the array either way.
        branch = _pointnet(num_points=CLOUD_ROWS * 4)
        cloud = _cloud()
        params = branch.init(rng, cloud)

        embed = branch.apply(params, cloud)

        assert embed.shape == (EMBED_DIM,)
        assert bool(jnp.isfinite(jnp.asarray(embed, jnp.float32)).all())

    def test_tnets_start_as_identity(self, rng):
        # The T-Nets must be no-ops at init (arXiv:1612.00593): zero kernel plus
        # an identity bias, so training starts from the untransformed cloud.
        branch = _pointnet()
        params = branch.init(rng, _cloud())

        for name, k in (("input_tnet", 3), ("feature_tnet", branch.mlp1[-1])):
            transform = params["params"][name]["transform"]
            assert bool((transform["kernel"] == 0.0).all())
            assert bool((transform["bias"].reshape(k, k) == jnp.eye(k)).all())

    def test_projects_only_when_the_pooled_width_differs(self, rng):
        cloud = _cloud()
        matched = _pointnet(embed_dim=EMBED_DIM, mlp2=(8, 16, EMBED_DIM))
        widened = _pointnet(embed_dim=EMBED_DIM * 2, mlp2=(8, 16, EMBED_DIM))

        matched_params = matched.init(rng, cloud)
        widened_params = widened.init(rng, cloud)

        # mlp2[-1] == embed_dim: the max-pooled global feature *is* the branch
        # embedding, with no extra projection parameters.
        assert "proj" not in matched_params["params"]
        assert "proj" in widened_params["params"]
        assert widened.apply(widened_params, cloud).shape == (EMBED_DIM * 2,)

    def test_rejects_a_cloud_with_the_wrong_channel_count(self, rng):
        branch = _pointnet()
        with pytest.raises(ValueError, match=r"expected \(N, 6\) cloud"):
            branch.init(rng, jnp.zeros((CLOUD_ROWS, 4), jnp.float16))

    def test_shared_mlps_and_tnet_heads_receive_gradient(self, rng):
        branch = _pointnet()
        cloud = _cloud()
        params = branch.init(rng, cloud)

        @jax.jit
        def loss_fn(p):
            return jnp.sum(jnp.asarray(branch.apply(p, cloud), jnp.float32) ** 2)

        grads = jax.grad(loss_fn)(params)

        leaves = jax.tree_util.tree_leaves(grads)
        assert leaves
        assert all(bool(jnp.isfinite(g).all()) for g in leaves)
        # The zero T-Net kernel blocks the T-Nets' *internal* layers at init, but
        # the transform kernel itself must get signal or they never unblock.
        for name in ("mlp1_0", "mlp2_0"):
            assert bool((jnp.abs(grads["params"][name]["kernel"]) > 0).any())
        for name in ("input_tnet", "feature_tnet"):
            kernel = grads["params"][name]["transform"]["kernel"]
            assert bool((jnp.abs(kernel) > 0).any())


# --------------------------------------------------------------------------- #
# GNN cloud branch                                                             #
# --------------------------------------------------------------------------- #


class TestGnnCloudEncoder:
    """k-NN graph + weighted-mean GCN over one unbatched live cloud."""

    def test_encodes_one_cloud_to_one_vector(self, rng):
        branch = _gnn()
        cloud = _cloud()
        params = branch.init(rng, cloud)

        embed = branch.apply(params, cloud)

        assert embed.shape == (EMBED_DIM,)
        assert bool(jnp.isfinite(jnp.asarray(embed, jnp.float32)).all())

    def test_defaults_are_the_validated_sage_baseline(self):
        # Guards the 50k-step-validated baseline (jobs 5736062/5736907): the
        # default config must stay the plain "sage" path with no residuals.
        branch = GnnCloudEncoder()
        assert branch.message_mode == "sage"
        assert branch.residual is False
        assert (
            branch.num_graph_nodes,
            branch.knn_k,
            branch.gcn_hidden,
            branch.gcn_layers,
        ) == (4096, 8, 128, 2)

    def test_sage_mode_has_no_per_edge_dense(self, rng):
        params = _gnn().init(rng, _cloud())
        assert not any(k.startswith("gnn_edge") for k in params["params"])

    def test_edgeconv_mode_edge_dense_receives_gradient(self, rng):
        branch = _gnn(message_mode="edgeconv")
        cloud = _cloud()
        params = branch.init(rng, cloud)

        @jax.jit
        def loss_fn(p):
            return jnp.sum(jnp.asarray(branch.apply(p, cloud), jnp.float32) ** 2)

        grads = jax.grad(loss_fn)(params)

        for i in range(branch.gcn_layers):
            kernel = grads["params"][f"gnn_edge{i}"]["kernel"]
            assert bool((jnp.abs(kernel) > 0).any())

    def test_gcn_layers_receive_gradient(self, rng):
        branch = _gnn()
        cloud = _cloud()
        params = branch.init(rng, cloud)

        grads = jax.grad(
            lambda p: jnp.sum(jnp.asarray(branch.apply(p, cloud), jnp.float32) ** 2)
        )(params)

        assert bool((jnp.abs(grads["params"]["gnn_hidden0"]["kernel"]) > 0).any())

    def test_rejects_an_unknown_message_mode(self, rng):
        with pytest.raises(ValueError, match="message_mode"):
            _gnn(message_mode="nonsense").init(rng, _cloud())

    def test_rejects_a_cloud_with_the_wrong_channel_count(self, rng):
        with pytest.raises(ValueError, match=r"expected \(N, 6\) cloud"):
            _gnn().init(rng, jnp.zeros((CLOUD_ROWS, 4), jnp.float16))


@pytest.mark.parametrize("make_branch", [_pointnet, _gnn], ids=["pointnet", "gnn"])
def test_cloud_branch_optimizes_without_going_non_finite(make_branch):
    """A short optimization of a cloud branch must converge, not explode.

    The unit-scale stand-in for the smoke runs: both branches mix float32
    normalization with reduced-precision compute, and a regression there shows up
    as a NaN loss or grad norm within a handful of steps.
    """
    branch = make_branch()
    cloud = _cloud()
    target = jax.random.normal(jax.random.PRNGKey(3), (EMBED_DIM,), dtype=jnp.float32)
    params = branch.init(jax.random.PRNGKey(1), cloud)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            embed = jnp.asarray(branch.apply(p, cloud), jnp.float32)
            return jnp.mean((embed - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, grad_norm

    losses = []
    for _ in range(12):
        params, opt_state, loss, grad_norm = step(params, opt_state)
        assert bool(jnp.isfinite(loss)), "loss went non-finite"
        assert bool(jnp.isfinite(grad_norm)), "grad norm went non-finite"
        losses.append(float(loss))

    assert losses[-1] < losses[0]


# --------------------------------------------------------------------------- #
# Token branch                                                                 #
# --------------------------------------------------------------------------- #


def _token_branch(**overrides) -> TokenSequenceEncoder:
    kwargs: dict[str, object] = {
        "num_tokens": NUM_TOKENS,
        "token_dim": TOKEN_DIM,
        "embed_dim": EMBED_DIM,
        "layers": 1,
        "heads": 2,
    }
    kwargs.update(overrides)
    return TokenSequenceEncoder(**kwargs)


class TestTokenSequenceEncoder:
    """The Transformer branch over per-step ``(..., num_tokens, token_dim)`` fields."""

    def test_restores_replay_leading_dims(self, rng):
        # ``TokenTransformerEncoder`` takes at most one batch axis, so the branch
        # folds (B, T) into the batch and must put it back.
        branch = _token_branch()
        tokens = jnp.zeros((2, 3, NUM_TOKENS, TOKEN_DIM), jnp.float16)
        params = branch.init(rng, tokens)

        embed = branch.apply(params, tokens)

        assert embed.shape == (2, 3, EMBED_DIM)

    def test_unbatched_event_returns_one_vector(self, rng):
        branch = _token_branch()
        tokens = jnp.zeros((NUM_TOKENS, TOKEN_DIM), jnp.float16)
        params = branch.init(rng, tokens)

        assert branch.apply(params, tokens).shape == (EMBED_DIM,)

    def test_forwards_the_compute_dtype_to_the_transformer(self, rng):
        branch = _token_branch(compute_dtype=jnp.bfloat16)
        tokens = jnp.zeros((2, NUM_TOKENS, TOKEN_DIM), jnp.float32)
        params = branch.init(rng, tokens)

        assert branch.apply(params, tokens).dtype == jnp.bfloat16


class TestTokenTransformerEncoder:
    """Shape contract of the Transformer the token branch wraps."""

    def _encoder(self, **overrides) -> TokenTransformerEncoder:
        kwargs: dict[str, object] = {
            "embed_dim": EMBED_DIM,
            "token_dim": TOKEN_DIM,
            "num_tokens": NUM_TOKENS,
            "layers": 1,
            "heads": 2,
        }
        kwargs.update(overrides)
        return TokenTransformerEncoder(**kwargs)

    def test_encodes_a_token_sequence_to_the_embed_width(self, rng):
        encoder = self._encoder()
        tokens = jnp.zeros((2, NUM_TOKENS, TOKEN_DIM), jnp.float32)
        params = encoder.init(rng, tokens)

        embed = encoder.apply(params, tokens)

        assert embed.shape == (2, EMBED_DIM)
        assert bool(jnp.isfinite(jnp.asarray(embed, jnp.float32)).all())

    def test_one_positional_embedding_per_kept_token(self, rng):
        encoder = self._encoder()
        params = encoder.init(rng, jnp.zeros((1, NUM_TOKENS, TOKEN_DIM), jnp.float32))

        assert params["params"]["pos_embed"].shape == (1, NUM_TOKENS, TOKEN_DIM)

    def test_rejects_a_token_shape_it_cannot_reinterpret(self, rng):
        encoder = self._encoder()
        with pytest.raises(ValueError, match="expected tokens"):
            encoder.init(rng, jnp.zeros((1, NUM_TOKENS, TOKEN_DIM - 1), jnp.float32))

    def test_rejects_a_head_count_that_does_not_divide_the_model_dim(self, rng):
        encoder = self._encoder(heads=3)
        with pytest.raises(ValueError, match="divisible by heads"):
            encoder.init(rng, jnp.zeros((1, NUM_TOKENS, TOKEN_DIM), jnp.float32))


# --------------------------------------------------------------------------- #
# Conv and MLP branches, decoder probe                                         #
# --------------------------------------------------------------------------- #


CONV_KWARGS: dict[str, object] = {"depth": 4, "kernel_size": 3, "mults": (1, 1, 1, 1)}
# Four stride-2 stages take 64x64 down to 4x4, flattened at the last mult width.
CONV_EMBED_DIM = 4 * 1 * 4 * 4


class TestConvEncoder:
    """The conv branch every variant routes its replay RGB frame to."""

    def test_preserves_replay_leading_dims(self, rng):
        encoder = ConvEncoder(**CONV_KWARGS)
        images = jnp.zeros((2, 3, 64, 64, 3), jnp.uint8)
        params = encoder.init(rng, images)

        embed = encoder.apply(params, images)

        assert embed.shape == (2, 3, CONV_EMBED_DIM)

    def test_unbatched_event_returns_one_vector(self, rng):
        encoder = ConvEncoder(**CONV_KWARGS)
        image = jnp.zeros((64, 64, 3), jnp.uint8)
        params = encoder.init(rng, image)

        assert encoder.apply(params, image).shape == (CONV_EMBED_DIM,)

    def test_uint8_and_normalized_float_images_encode_identically(self, rng):
        # Replay stores uint8; ``act`` may hand over already-normalized floats.
        # The branch normalizes to [0, 1] and centers, so both must agree.
        encoder = ConvEncoder(**CONV_KWARGS)
        pixels = jax.random.randint(rng, (1, 64, 64, 3), 0, 256, dtype=jnp.int32)
        as_uint8 = pixels.astype(jnp.uint8)
        as_float = pixels.astype(jnp.float32) / 255.0
        params = encoder.init(rng, as_uint8)

        assert jnp.allclose(
            encoder.apply(params, as_uint8),
            encoder.apply(params, as_float),
            atol=1e-5,
        )

    def test_centers_images_around_zero(self, rng):
        # Dreamer's ``obs - 0.5`` centering: a mid-gray frame is the zero input,
        # so its embedding is the bias-only response of the stack.
        encoder = ConvEncoder(**CONV_KWARGS)
        gray = jnp.full((1, 64, 64, 3), 0.5, jnp.float32)
        zeros = jnp.zeros((1, 64, 64, 3), jnp.float32)
        params = encoder.init(rng, gray)

        gray_embed = encoder.apply(params, gray)
        black_embed = encoder.apply(params, zeros)

        assert not jnp.allclose(gray_embed, black_embed)
        assert bool(jnp.isfinite(gray_embed).all())

    def test_world_point_mode_symlogs_the_metric_range(self, rng):
        # ``Encoder.CONV_POINTS`` shares this module but feeds unbounded metric
        # XYZ: without symlog, metre-scale coordinates saturate the stack.
        encoder = ConvEncoder(
            input_kind="world_points", embed_dim=EMBED_DIM, **CONV_KWARGS
        )
        far = jnp.full((1, 64, 64, 3), 1e3, jnp.float32)
        params = encoder.init(rng, far)

        embed = encoder.apply(params, far)

        assert embed.shape == (1, EMBED_DIM)
        assert bool(jnp.isfinite(embed).all())

    def test_rejects_an_unknown_input_kind(self, rng):
        encoder = ConvEncoder(input_kind="depth", **CONV_KWARGS)
        with pytest.raises(ValueError, match="input_kind"):
            encoder.init(rng, jnp.zeros((1, 64, 64, 3), jnp.float32))


class TestMLPEncoder:
    """The dense branch for flat fields (camera pose)."""

    def test_preserves_replay_leading_dims(self, rng):
        encoder = MLPEncoder(embed_dim=EMBED_DIM, hidden=8, num_layers=1)
        features = jnp.zeros((2, 3, 9), jnp.float32)
        params = encoder.init(rng, features)

        assert encoder.apply(params, features).shape == (2, 3, EMBED_DIM)

    @pytest.mark.parametrize(
        "num_layers, expected",
        [
            (0, {"proj"}),
            (1, {"hidden0", "norm0", "proj"}),
            (3, {f"{p}{i}" for p in ("hidden", "norm") for i in range(3)} | {"proj"}),
        ],
    )
    def test_num_layers_controls_the_block_count(self, rng, num_layers, expected):
        # num_layers=0 is the escape hatch back to a bare linear projection.
        encoder = MLPEncoder(embed_dim=EMBED_DIM, hidden=8, num_layers=num_layers)
        params = encoder.init(rng, jnp.zeros((1, 9), jnp.float32))

        assert set(params["params"]) == expected


class TestConvDecoder:
    """The stop-gradient reconstruction probe behind ``--decoder``."""

    def test_decodes_features_to_rgb_in_the_unit_range(self, rng):
        decoder = ConvDecoder(**CONV_KWARGS)
        feat = jax.random.normal(rng, (4, 64), dtype=jnp.float32)
        params = decoder.init(rng, feat)

        image = decoder.apply(params, feat)

        assert image.shape == (4, 64, 64, 3)
        assert float(jnp.min(image)) >= 0.0
        assert float(jnp.max(image)) <= 1.0


# --------------------------------------------------------------------------- #
# Composite fusion                                                             #
# --------------------------------------------------------------------------- #


def _field(key: str, encoder: Encoder, value: jnp.ndarray, *, buffer: bool = True):
    return AdapterField(key=key, encoder=encoder, buffer=buffer, value=value)


def _image_field(key: str = "image") -> AdapterField:
    return _field(key, Encoder.CONV, jnp.zeros((64, 64, 3), jnp.uint8))


def _pose_field(key: str = "camera_pose") -> AdapterField:
    return _field(key, Encoder.MLP, jnp.zeros((9,), jnp.float32))


SMALL_BRANCHES: dict[str, object] = {
    "conv_depth": 4,
    "conv_kernel": 3,
    "conv_mults": (1, 1, 1, 1),
    "branch_embed_dim": EMBED_DIM,
    "mlp_hidden": 8,
    "pointnet_num_points": 32,
    "gnn_num_nodes": 32,
    "fusion_dim": 24,
}


# One representative field per ``Encoder`` member: how to build its value, and
# whether the branch consumes it per step. The cloud branches take a single
# unbatched cloud, so they are only ever routed as the live field.
BRANCH_CASES: dict[Encoder, tuple[object, bool]] = {
    Encoder.CONV: (lambda: jnp.zeros((64, 64, 3), jnp.uint8), True),
    Encoder.CONV_POINTS: (lambda: jnp.zeros((64, 64, 3), jnp.float32), True),
    Encoder.MLP: (lambda: jnp.zeros((9,), jnp.float32), True),
    Encoder.POINTNET: (_cloud, False),
    Encoder.GNN: (_cloud, False),
    Encoder.TRANSFORMER: (
        lambda: jnp.zeros((NUM_TOKENS, TOKEN_DIM), jnp.float16),
        True,
    ),
}


def _obs(fields, *, leading=(2, 3)) -> dict[str, jnp.ndarray]:
    """Batch the replayed fields to ``leading``; live fields stay single events."""
    return {
        f.key: (
            jnp.broadcast_to(f.value, (*leading, *f.value.shape))
            if f.buffer
            else f.value
        )
        for f in fields
    }


class TestRoutedCompositeEncoder:
    """Routing metadata, not a config string, decides the branch layout."""

    def test_routes_are_sorted_by_key(self):
        # Concatenation order must be deterministic across runs or the params
        # pytree (and every checkpoint written from it) shifts.
        fields = [_pose_field(), _image_field(), _pose_field("zzz")]
        assert [r.key for r in routes_from_fields(fields)] == [
            "camera_pose",
            "image",
            "zzz",
        ]

    def test_a_single_branch_variant_is_its_branch_unfused(self, rng):
        fields = [_image_field()]
        encoder = routed_encoder_from_fields(fields, **SMALL_BRANCHES)
        obs = _obs(fields)
        params = encoder.init(rng, obs)

        embed = encoder.apply(params, obs)

        assert "fusion" not in params["params"]
        assert embed.shape == (2, 3, CONV_EMBED_DIM)

    def test_multi_branch_variants_fuse_to_a_fixed_width(self, rng):
        # The RSSM input width must not depend on how many modalities a variant
        # happens to compose, so anything past one branch gets a fusion Dense.
        fields = [_image_field(), _pose_field()]
        encoder = routed_encoder_from_fields(fields, **SMALL_BRANCHES)
        obs = _obs(fields)
        params = encoder.init(rng, obs)

        embed = encoder.apply(params, obs)

        assert "fusion" in params["params"]
        assert embed.shape == (2, 3, SMALL_BRANCHES["fusion_dim"])

    def test_a_live_field_is_encoded_once_and_broadcast(self, rng):
        # The live cloud has no (B, T) prefix: it is one global event whose
        # embedding is broadcast over the per-step leading dims.
        fields = [
            _image_field(),
            _field("house_context", Encoder.POINTNET, _cloud(), buffer=False),
        ]
        encoder = routed_encoder_from_fields(fields, **SMALL_BRANCHES)
        obs = _obs(fields)
        params = encoder.init(rng, obs)

        embed = encoder.apply(params, obs)

        assert encoder.global_keys == ("house_context",)
        assert embed.shape == (2, 3, SMALL_BRANCHES["fusion_dim"])
        assert bool(jnp.isfinite(jnp.asarray(embed, jnp.float32)).all())

    def test_routing_without_a_per_step_field_is_rejected(self, rng):
        # Leading dims come from the replayed fields; a live-only routing has
        # nothing to broadcast against.
        fields = [_field("house_context", Encoder.POINTNET, _cloud(), buffer=False)]
        encoder = routed_encoder_from_fields(fields, **SMALL_BRANCHES)

        with pytest.raises(ValueError, match="at least one per-step key"):
            encoder.init(rng, _obs(fields))

    def test_every_encoder_member_is_exercised_below(self):
        # The ``Encoder`` docstring promises every member has a branch here. A
        # missing one would otherwise surface as NotImplementedError inside a
        # jitted apply, on whichever variant first routes to it — so a new
        # member has to come with a row in BRANCH_CASES.
        assert set(BRANCH_CASES) == set(Encoder)

    @pytest.mark.parametrize("member", list(BRANCH_CASES), ids=lambda m: m.name)
    def test_each_branch_encodes_its_field_through_the_composite(self, rng, member):
        make_value, per_step = BRANCH_CASES[member]
        fields = [_field(member.name.lower(), member, make_value(), buffer=per_step)]
        if not per_step:
            # Leading dims come from a per-step field; pair the live branch with one.
            fields.append(_image_field())
        encoder = routed_encoder_from_fields(
            fields, transformer_heads=2, **SMALL_BRANCHES
        )
        obs = _obs(fields, leading=(1,))

        embed = encoder.apply(encoder.init(rng, obs), obs)

        assert embed.shape[:-1] == (1,)
        assert bool(jnp.isfinite(jnp.asarray(embed, jnp.float32)).all())

    def test_branch_hyperparameters_default_to_the_module_attributes(self):
        # ``routed_encoder_from_fields`` must not invent its own defaults: the
        # agent translates the run config into overrides, everything else comes
        # from the module.
        encoder = routed_encoder_from_fields([_image_field()])
        defaults = RoutedCompositeEncoder(routes=())

        assert encoder.conv_depth == defaults.conv_depth
        assert encoder.branch_embed_dim == defaults.branch_embed_dim
        assert encoder.fusion_dim == defaults.fusion_dim
