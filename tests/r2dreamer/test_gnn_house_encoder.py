"""CPU tests for the GNN house encoders (src/r2dreamer/encoders/gnn_house.py)."""

import jax
import jax.numpy as jnp
import pytest

from src.r2dreamer.encoders.gnn_house import (
    GnnEdgeHousePointsCameraEncoder,
    GnnHousePointsCameraEncoder,
)
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
)

VARIANTS = [GnnHousePointsCameraEncoder, GnnEdgeHousePointsCameraEncoder]


def _make_obs(batch: int, n_points: int, size: int, key=0):
    rng = jax.random.PRNGKey(key)
    k_pose, k_pts = jax.random.split(rng)
    points = jax.random.uniform(k_pts, (1, n_points, 6), dtype=jnp.float32)
    mask = (jnp.arange(n_points) < size)[None, :, None]
    return {
        CAMERA_POSE_KEY: jax.random.normal(k_pose, (batch, 9), dtype=jnp.float32),
        HOUSE_CONTEXT_KEY: jnp.where(mask, points, 0.0).astype(jnp.float16),
        HOUSE_CONTEXT_SIZE_KEY: jnp.asarray(size, dtype=jnp.int32),
    }


def _make_encoder(cls=GnnHousePointsCameraEncoder, **overrides):
    kwargs = dict(
        embed_dim=32,
        camera_hidden=16,
        point_hidden=16,
        num_graph_nodes=64,
        knn_k=4,
        gcn_hidden=16,
        gcn_layers=2,
    )
    kwargs.update(overrides)
    return cls(**kwargs)


@pytest.mark.parametrize("cls", VARIANTS)
def test_forward_shape_and_finite(cls):
    enc = _make_encoder(cls)
    obs = _make_obs(batch=5, n_points=512, size=300)
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert out.shape == (5, 64)
    assert bool(jnp.isfinite(out).all())


@pytest.mark.parametrize("cls", VARIANTS)
def test_empty_house_zeroes_house_branch(cls):
    enc = _make_encoder(cls)
    obs = _make_obs(batch=3, n_points=512, size=0)
    params = enc.init(jax.random.PRNGKey(1), obs)
    camera_embed, house_embed = enc.apply(params, obs, method="branches")
    assert bool(jnp.isfinite(camera_embed).all())
    assert bool((house_embed == 0.0).all())


@pytest.mark.parametrize("cls", VARIANTS)
def test_size_below_node_count_is_finite(cls):
    enc = _make_encoder(cls)
    obs = _make_obs(batch=2, n_points=512, size=7)
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert bool(jnp.isfinite(out).all())


@pytest.mark.parametrize("cls", VARIANTS)
def test_jit_and_grad(cls):
    enc = _make_encoder(cls)
    obs = _make_obs(batch=2, n_points=256, size=200)
    params = enc.init(jax.random.PRNGKey(1), obs)

    @jax.jit
    def loss_fn(p):
        return jnp.sum(enc.apply(p, obs) ** 2)

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    # The GCN layers must receive gradient signal.
    gnn_grads = grads["params"]["gnn_hidden0"]["kernel"]
    assert bool((jnp.abs(gnn_grads) > 0).any())


def test_rejects_multi_house_batch():
    enc = _make_encoder()
    obs = _make_obs(batch=2, n_points=256, size=100)
    obs[HOUSE_CONTEXT_KEY] = jnp.concatenate(
        [obs[HOUSE_CONTEXT_KEY], obs[HOUSE_CONTEXT_KEY]], axis=0
    )
    with pytest.raises(ValueError, match="singleton"):
        enc.init(jax.random.PRNGKey(1), obs)


def test_baseline_defaults_unchanged():
    # Guards the 50k-validated baseline (jobs 5736062/5736907): the default
    # config must stay bit-for-bit the "sage" path with no residuals.
    enc = GnnHousePointsCameraEncoder(embed_dim=32)
    assert enc.message_mode == "sage"
    assert enc.residual is False
    assert (enc.num_graph_nodes, enc.knn_k, enc.gcn_hidden, enc.gcn_layers) == (
        4096, 8, 128, 2,
    )


def test_sage_params_have_no_edge_dense():
    enc = _make_encoder()
    obs = _make_obs(batch=2, n_points=256, size=200)
    params = enc.init(jax.random.PRNGKey(1), obs)
    assert not any(k.startswith("gnn_edge") for k in params["params"])


def test_edgeconv_edge_dense_gets_gradient():
    enc = _make_encoder(GnnEdgeHousePointsCameraEncoder)
    obs = _make_obs(batch=2, n_points=256, size=200)
    params = enc.init(jax.random.PRNGKey(1), obs)

    @jax.jit
    def loss_fn(p):
        return jnp.sum(enc.apply(p, obs) ** 2)

    grads = jax.grad(loss_fn)(params)
    for i in range(enc.gcn_layers):
        g = grads["params"][f"gnn_edge{i}"]["kernel"]
        assert bool((jnp.abs(g) > 0).any())


def test_invalid_message_mode_raises():
    enc = _make_encoder(message_mode="nonsense")
    obs = _make_obs(batch=2, n_points=256, size=200)
    with pytest.raises(ValueError, match="message_mode"):
        enc.init(jax.random.PRNGKey(1), obs)


@pytest.mark.parametrize("cls", VARIANTS)
def test_short_training_loop_is_stable(cls):
    # Locks in the smoke-run behavior at unit scale: a short end-to-end
    # optimization of the full encoder must keep loss/grads finite and
    # actually reduce the loss (no explosion, no collapse to NaN).
    import optax

    enc = _make_encoder(cls)
    obs = _make_obs(batch=4, n_points=512, size=400)
    target = jax.random.normal(jax.random.PRNGKey(3), (4, 64), dtype=jnp.float32)
    params = enc.init(jax.random.PRNGKey(1), obs)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            return jnp.mean((enc.apply(p, obs) - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        gnorm = optax.global_norm(grads)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, gnorm

    losses = []
    for _ in range(15):
        params, opt_state, loss, gnorm = step(params, opt_state)
        assert bool(jnp.isfinite(loss)), "loss went non-finite"
        assert bool(jnp.isfinite(gnorm)), "grad norm went non-finite"
        losses.append(float(loss))
    assert losses[-1] < losses[0]
