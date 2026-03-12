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

    def __init__(self, cfg, rng: jax.Array):
        self.cfg = cfg
        key1, key2, key3 = jax.random.split(rng, 3)
        obs_dim = cfg.obs_dim if hasattr(cfg, "obs_dim") else cfg.get("obs_dim", 4)
        hidden = cfg.hidden if hasattr(cfg, "hidden") else cfg.get("hidden", 8)
        num_actions = cfg.num_actions if hasattr(cfg, "num_actions") else 6
        params = {
            "encoder": {
                "kernel": jax.random.normal(key1, (obs_dim, hidden)),
                "bias": jax.random.normal(key2, (hidden,)),
            },
            "policy": {
                "kernel": jax.random.normal(key3, (hidden, num_actions)),
            },
        }
        self.wm_state = WMState(params=params)

    def act(self, obs, rng: jax.Array, training: bool = True):
        return self._act_forward(obs, rng, training)

    def _act_forward(self, obs, rng: jax.Array, training: bool = True):
        logits = self.wm_state.params["policy"]["kernel"].mean(axis=0)
        if training:
            action = jax.random.categorical(rng, logits)
        else:
            action = jnp.argmax(logits, axis=-1)
        return action

    def train_step(self, batch, rng: jax.Array) -> dict:
        return self._train_behavior(batch, rng)

    def _train_behavior(self, batch, rng: jax.Array) -> dict:
        imag_return = jnp.mean(batch["rewards"]).item()
        return {"imag_return": float(imag_return)}

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
