"""Checkpoint and serializable config snapshot helpers for R2-Dreamer."""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np


class CheckpointAgentLike(Protocol):
    """Agent state required by checkpoint serialization."""

    params: Any
    opt_state: Any
    slow_critic_params: Any
    ema_state: Any


def _missing_pickle_class(module: str, name: str) -> type:
    class MissingPickleClass:
        """Placeholder for optimizer-state classes missing from the current tree."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.state: Any = None

        def __setstate__(self, state):
            self.state = state

    MissingPickleClass.__name__ = name
    MissingPickleClass.__module__ = module
    return MissingPickleClass


class _CheckpointUnpickler(pickle.Unpickler):
    """Load old checkpoints even when unused optimizer-state classes moved."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return _missing_pickle_class(module, name)


def config_snapshot(config: Any) -> dict[str, Any]:
    """Return a JSON-serializable run config snapshot for manifests/W&B."""
    return (
        asdict(cast(Any, config))
        if is_dataclass(config)
        else dict(vars(config))
        if hasattr(config, "__dict__")
        else {}
    )


def save_checkpoint(agent: CheckpointAgentLike, step: int, output_dir: str) -> str:
    """Save full agent state including ema_state. Returns path."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:09d}.pkl")
    data = {
        "step": step,
        "params": jax.tree.map(np.array, agent.params),
        "opt_state": jax.tree.map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
            agent.opt_state,
        ),
        "slow_critic_params": jax.tree.map(np.array, agent.slow_critic_params),
        "ema_state": jax.tree.map(np.array, agent.ema_state),
    }
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, path)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load checkpoint dict from disk. Returns raw dict — caller restores."""
    with open(path, "rb") as f:
        return _CheckpointUnpickler(f).load()
