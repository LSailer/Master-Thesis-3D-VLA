"""Minimal DreamerV3 agent stub with checkpoint save/load."""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import struct


class WMState(struct.PyTreeNode):
    params: dict

    def replace(self, **kwargs):
        return self.__class__(**{**self.__dict__, **kwargs})


class DreamerAgent:
    """Minimal DreamerV3 agent with world model state and checkpoint I/O."""

    def __init__(self, cfg: dict, rng: jax.Array):
        key1, key2 = jax.random.split(rng)
        params = {
            "encoder": {
                "kernel": jax.random.normal(key1, (cfg.get("obs_dim", 4), cfg.get("hidden", 8))),
                "bias": jax.random.normal(key2, (cfg.get("hidden", 8),)),
            }
        }
        self.wm_state = WMState(params=params)

    def save(self, path: str) -> None:
        dest = Path(path) / "checkpoint.pkl"
        params = jax.tree.map(lambda x: x, self.wm_state.params)
        with open(dest, "wb") as f:
            pickle.dump(params, f)

    def load(self, path: str) -> None:
        src = Path(path) / "checkpoint.pkl"
        with open(src, "rb") as f:
            params = pickle.load(f)
        self.wm_state = self.wm_state.replace(params=params)
