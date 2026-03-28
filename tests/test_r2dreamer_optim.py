"""Tests for LaProp optimizer and AGC gradient clipping."""

import jax
import jax.numpy as jnp
import optax
import pytest

from src.dreamerv3.optim import laprop, agc


class TestLaProp:
    def test_basic_optimization(self):
        """LaProp should minimize x^2."""
        tx = laprop(lr=4e-2)
        params = jnp.array([5.0, -3.0])
        state = tx.init(params)
        for _ in range(500):
            grads = 2 * params
            updates, state = tx.update(grads, state, params)
            params = optax.apply_updates(params, updates)
        assert jnp.allclose(params, jnp.zeros(2), atol=0.1)

    def test_pytree_params(self):
        """LaProp works with nested pytree params."""
        tx = laprop(lr=1e-2)
        params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
        state = tx.init(params)
        grads = jax.tree.map(lambda p: 2 * p, params)
        updates, state = tx.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        # Should move toward zero
        assert jnp.all(jnp.abs(new_params["a"]) < jnp.abs(params["a"]))


class TestAGC:
    def test_clips_large_gradients(self):
        params = jnp.array([1.0, 1.0])
        grads = jnp.array([100.0, 100.0])
        clipped = agc(grads, params, clip=0.3, pmin=1e-3)
        assert jnp.all(jnp.abs(clipped) < jnp.abs(grads))

    def test_preserves_small_gradients(self):
        params = jnp.array([10.0, 10.0])
        grads = jnp.array([0.01, 0.01])
        clipped = agc(grads, params, clip=0.3, pmin=1e-3)
        assert jnp.allclose(clipped, grads)

    def test_pmin_floor(self):
        params = jnp.array([0.0, 0.0])
        grads = jnp.array([10.0, 10.0])
        clipped = agc(grads, params, clip=0.3, pmin=1e-3)
        assert jnp.all(jnp.isfinite(clipped))

    def test_pytree(self):
        params = {"a": jnp.array([1.0]), "b": jnp.array([1.0, 1.0])}
        grads = {"a": jnp.array([100.0]), "b": jnp.array([100.0, 100.0])}
        clipped = agc(grads, params, clip=0.3, pmin=1e-3)
        assert jnp.all(jnp.abs(clipped["a"]) < jnp.abs(grads["a"]))
