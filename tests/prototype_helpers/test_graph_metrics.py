"""Checks for ``src.prototype_helpers.graph_metrics``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.prototype_helpers.graph_metrics import (
    chamfer_distances,
    rgb_psnr,
    table_bytes,
)


def test_chamfer_zero_for_identical_clouds() -> None:
    cloud = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=jnp.float32)
    forward, backward = chamfer_distances(cloud, cloud)
    assert forward == pytest.approx(0.0, abs=1e-6)
    assert backward == pytest.approx(0.0, abs=1e-6)


def test_chamfer_unit_shift() -> None:
    a = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
    b = a + jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    forward, backward = chamfer_distances(a, b)
    assert forward == pytest.approx(1.0, abs=1e-5)
    assert backward == pytest.approx(1.0, abs=1e-5)


def test_psnr_identical_is_infinite() -> None:
    rgb = jnp.array([[10, 20, 30]], dtype=jnp.uint8)
    assert rgb_psnr(rgb, rgb) == float("inf")


def test_psnr_matches_formula() -> None:
    true = jnp.zeros((4, 3), dtype=jnp.float32)
    pred = jnp.full((4, 3), 10.0, dtype=jnp.float32)
    expected = 10.0 * np.log10(255.0**2 / 100.0)
    assert rgb_psnr(true, pred) == pytest.approx(expected, abs=1e-3)


def test_table_bytes_breakdown() -> None:
    sizes = table_bytes(num_points=1000, k=16)
    assert sizes["node_table_bytes"] == 1000 * 9
    assert sizes["num_edges"] == 2 * 1000 * 16
    assert sizes["edge_table_coo_bytes"] == 2 * 1000 * 16 * 10
    assert sizes["edge_table_implicit_senders_bytes"] == 1000 * 16 * 6
    assert (
        sizes["total_implicit_bytes"]
        == sizes["node_table_bytes"] + sizes["edge_table_implicit_senders_bytes"]
    )
