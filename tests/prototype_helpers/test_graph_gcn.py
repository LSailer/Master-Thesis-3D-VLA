"""Learning-behavior checks for ``src.prototype_helpers.graph_gcn``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.prototype_helpers.graph_gcn import RgbGcnAutoencoder
from src.prototype_helpers.knn_graph import build_knn_graph


def _smooth_colored_cloud(num_points: int = 200) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Random blob whose RGB varies smoothly with position (graph-smooth signal)."""
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(num_points, 3)).astype(np.float32)
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    rgb01 = (xyz - low) / (high - low)
    return jnp.asarray(xyz), jnp.asarray(rgb01, dtype=jnp.float32)


def test_masked_rgb_reconstruction_loss_halves() -> None:
    xyz, rgb01 = _smooth_colored_cloud()
    graph = build_knn_graph(xyz, k=6)
    num_nodes = graph.num_nodes

    mask = jax.random.bernoulli(jax.random.PRNGKey(1), 0.3, (num_nodes,))
    corrupted_rgb = jnp.where(mask[:, None], 0.0, rgb01)
    node_feats = jnp.concatenate(
        [xyz, corrupted_rgb, mask[:, None].astype(jnp.float32)], axis=-1
    )

    model = RgbGcnAutoencoder(hidden=32, num_layers=2)
    params = model.init(
        jax.random.PRNGKey(0),
        node_feats,
        graph.senders,
        graph.receivers,
        graph.weights,
        num_nodes,
    )
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        def loss_fn(params):
            prediction = model.apply(
                params,
                node_feats,
                graph.senders,
                graph.receivers,
                graph.weights,
                num_nodes,
            )
            per_node = jnp.sum((prediction - rgb01) ** 2, axis=-1)
            return jnp.sum(per_node * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    params, opt_state, initial_loss = train_step(params, opt_state)
    final_loss = initial_loss
    for _ in range(49):
        params, opt_state, final_loss = train_step(params, opt_state)

    prediction = model.apply(
        params,
        node_feats,
        graph.senders,
        graph.receivers,
        graph.weights,
        num_nodes,
    )
    assert prediction.shape == (num_nodes, 3)
    assert float(prediction.min()) >= 0.0 and float(prediction.max()) <= 1.0
    assert float(final_loss) < 0.5 * float(initial_loss)
