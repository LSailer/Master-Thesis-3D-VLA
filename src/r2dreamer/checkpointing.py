"""Checkpoint and serializable config snapshot helpers for R2-Dreamer."""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from src.r2dreamer.observation_preparation import (
    CNNObservationPreparation,
    encoder_module_kwargs_from_config,
    module_class_path,
    recover_encoder_input_contract,
)


class CheckpointAgentLike(Protocol):
    """Agent state required by checkpoint serialization."""

    cfg: Any
    params: Any
    opt_state: Any
    slow_critic_params: Any
    ema_state: Any


def config_snapshot(config: Any) -> dict[str, Any]:
    """Return a JSON-serializable run config snapshot for manifests/W&B."""
    snapshot = (
        asdict(config)
        if is_dataclass(config)
        else dict(vars(config))
        if hasattr(config, "__dict__")
        else {}
    )
    encoder_module_cls = snapshot.pop("encoder_module_cls", None)
    runtime_cls = getattr(config, "encoder_module_cls", None)
    if runtime_cls is not None:
        snapshot["encoder_module"] = module_class_path(runtime_cls)
    elif encoder_module_cls is not None:
        snapshot["encoder_module"] = str(encoder_module_cls)
    snapshot["encoder_input_contract"] = encoder_input_contract_snapshot(config)
    return snapshot


def encoder_input_contract_snapshot(config: Any) -> dict[str, Any] | None:
    """Extract or derive the durable Encoder Input Contract snapshot from config."""
    snapshot = getattr(config, "encoder_input_contract", None)
    if snapshot is None:
        if getattr(config, "encoder_type", None) != "cnn":
            return None
        snapshot = CNNObservationPreparation().contract.to_snapshot()
    snapshot = dict(snapshot)
    contract = recover_encoder_input_contract(snapshot)
    snapshot["encoder_module_kwargs"] = encoder_module_kwargs_from_config(
        config,
        contract.encoder_module_cls,
    )
    return snapshot


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
    contract_snapshot = encoder_input_contract_snapshot(agent.cfg)
    if contract_snapshot is not None:
        data["encoder_input_contract"] = contract_snapshot
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load checkpoint dict from disk. Returns raw dict — caller restores."""
    with open(path, "rb") as f:
        return pickle.load(f)
