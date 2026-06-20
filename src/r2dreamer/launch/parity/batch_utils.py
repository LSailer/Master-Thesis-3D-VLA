"""Shared batch builders and config helpers for parity and benchmark scripts.

Both _convert_batch (JAX) and make_batch_torch (PyTorch) are extracted from
run_parity_training.py and run_benchmark.py where they were duplicated verbatim.
make_pytorch_config and make_crafter_spaces were also duplicated across both
source scripts; they live here to keep both callers DRY.
"""

import numpy as np

SEED = 42
WARMUP_STEPS = 3
BATCH_SIZE = 16
SEQ_LEN = 64
NUM_ACTIONS = 17
OBS_SHAPE_CHW = (3, 64, 64)
OBS_SHAPE_HWC = (64, 64, 3)


def _convert_batch(transitions, starts):
    """Build a JAX-format batch from a list of transitions and start indices.

    Returns a dict of jnp arrays with keys:
        obs (B, T, C, H, W), actions (B, T, A), rewards (B, T),
        is_first (B, T), is_last (B, T), is_terminal (B, T).
    """
    import jax.numpy as jnp

    B, T = len(starts), SEQ_LEN
    obs = np.zeros((B, T, *OBS_SHAPE_CHW), dtype=np.float32)
    actions = np.zeros((B, T, NUM_ACTIONS), dtype=np.float32)
    rewards = np.zeros((B, T), dtype=np.float32)
    is_first = np.zeros((B, T), dtype=np.float32)
    is_last = np.zeros((B, T), dtype=np.float32)
    is_terminal = np.zeros((B, T), dtype=np.float32)
    for i, s in enumerate(starts):
        for t in range(T):
            tr = transitions[s + t]
            obs[i, t] = tr["image_chw"].astype(np.float32) / 255.0
            actions[i, t, tr["action"]] = 1.0
            rewards[i, t] = tr["reward"]
            is_first[i, t] = float(tr["is_first"])
            is_last[i, t] = float(tr["is_last"])
            is_terminal[i, t] = float(tr["is_terminal"])
    return {
        "obs": jnp.array(obs),
        "actions": jnp.array(actions),
        "rewards": jnp.array(rewards),
        "is_first": jnp.array(is_first),
        "is_last": jnp.array(is_last),
        "is_terminal": jnp.array(is_terminal),
    }


def make_batch_torch(transitions, starts, device="cuda"):
    """Build a PyTorch TensorDict batch from a list of transitions and start indices.

    Returns a TensorDict with keys:
        image (B, T, H, W, C) uint8, action (B, T, A), reward (B, T, 1),
        is_first (B, T, 1) bool, is_last (B, T, 1) bool, is_terminal (B, T, 1) bool.
    """
    import torch
    from tensordict import TensorDict

    B, T = len(starts), SEQ_LEN
    obs = np.zeros((B, T, *OBS_SHAPE_HWC), dtype=np.uint8)
    actions = np.zeros((B, T, NUM_ACTIONS), dtype=np.float32)
    rewards = np.zeros((B, T, 1), dtype=np.float32)
    is_first = np.zeros((B, T, 1), dtype=np.float32)
    is_last = np.zeros((B, T, 1), dtype=np.float32)
    is_terminal = np.zeros((B, T, 1), dtype=np.float32)
    for i, s in enumerate(starts):
        for t in range(T):
            tr = transitions[s + t]
            obs[i, t] = tr["image_hwc"]
            actions[i, t, tr["action"]] = 1.0
            rewards[i, t, 0] = tr["reward"]
            is_first[i, t, 0] = float(tr["is_first"])
            is_last[i, t, 0] = float(tr["is_last"])
            is_terminal[i, t, 0] = float(tr["is_terminal"])
    return TensorDict(
        {
            "image": torch.tensor(obs, device=device),
            "action": torch.tensor(actions, device=device),
            "reward": torch.tensor(rewards, device=device),
            "is_first": torch.tensor(is_first, dtype=torch.bool, device=device),
            "is_last": torch.tensor(is_last, dtype=torch.bool, device=device),
            "is_terminal": torch.tensor(is_terminal, dtype=torch.bool, device=device),
        },
        batch_size=(B, T),
    )


def make_pytorch_config(device, rep_loss="r2dreamer"):
    """Return an OmegaConf config for the PyTorch r2dreamer / DreamerV3 agent."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "act_entropy": 3e-4,
            "kl_free": 1.0,
            "imag_horizon": 15,
            "horizon": 333,
            "lamb": 0.95,
            "compile": False,
            "log_grads": False,
            "device": device,
            "rep_loss": rep_loss,
            "lr": 4e-5,
            "agc": 0.3,
            "pmin": 1e-3,
            "eps": 1e-20,
            "beta1": 0.9,
            "beta2": 0.999,
            "warmup": 1000,
            "slow_target_update": 1,
            "slow_target_fraction": 0.02,
            "loss_scales": {
                "barlow": 0.05,
                "infonce": 1.0,
                "recon": 1.0,
                "rew": 1.0,
                "con": 1.0,
                "dyn": 1.0,
                "rep": 0.1,
                "policy": 1.0,
                "value": 1.0,
                "repval": 0.3,
                "swav": 1.0,
                "temp": 1.0,
                "norm": 1.0,
            },
            "r2dreamer": {"lambd": 5e-4},
            "rssm": {
                "stoch": 32,
                "deter": 2048,
                "hidden": 256,
                "discrete": 16,
                "img_layers": 2,
                "obs_layers": 1,
                "dyn_layers": 1,
                "blocks": 8,
                "act": "SiLU",
                "norm": True,
                "unimix_ratio": 0.01,
                "initial": "learned",
                "device": device,
            },
            "encoder": {
                "mlp_keys": "$^",
                "cnn_keys": "image",
                "mlp": {
                    "shape": None,
                    "layers": 3,
                    "units": 256,
                    "act": "SiLU",
                    "norm": True,
                    "device": device,
                    "outscale": None,
                    "symlog_inputs": True,
                    "name": "mlp_encoder",
                },
                "cnn": {
                    "act": "SiLU",
                    "norm": True,
                    "kernel_size": 5,
                    "minres": 4,
                    "depth": 16,
                    "mults": [2, 3, 4, 4],
                },
            },
            "decoder": {
                "mlp_keys": "$^",
                "cnn_keys": "image",
                "mlp_dist": {"name": "symlog_mse"},
                "cnn_dist": {"name": "mse"},
                "mlp": {
                    "shape": None,
                    "layers": 3,
                    "units": 256,
                    "act": "SiLU",
                    "norm": True,
                    "dist": {"name": "identity"},
                    "device": device,
                    "outscale": 1.0,
                    "symlog_inputs": False,
                    "name": "mlp_decoder",
                },
                "cnn": {
                    "depth": 16,
                    "units": 256,
                    "bspace": 8,
                    "mults": [2, 3, 4, 4],
                    "act": "SiLU",
                    "norm": True,
                    "kernel_size": 5,
                    "minres": 4,
                    "outscale": 1.0,
                },
            },
            "reward": {
                "shape": [255],
                "layers": 1,
                "units": 256,
                "act": "SiLU",
                "norm": True,
                "dist": {"name": "symexp_twohot", "bin_num": 255},
                "outscale": 0.0,
                "device": device,
                "symlog_inputs": False,
                "name": "reward",
            },
            "cont": {
                "shape": [1],
                "layers": 1,
                "units": 256,
                "act": "SiLU",
                "norm": True,
                "dist": {"name": "binary"},
                "outscale": 1.0,
                "device": device,
                "symlog_inputs": False,
                "name": "cont",
            },
            "actor": {
                "shape": None,
                "layers": 3,
                "units": 256,
                "act": "SiLU",
                "norm": True,
                "device": device,
                "dist": {
                    "cont": {"name": "bounded_normal", "min_std": 0.1, "max_std": 1.0},
                    "disc": {"name": "onehot", "unimix_ratio": 0.01},
                    "multi_disc": {"name": "multi_onehot", "unimix_ratio": 0.01},
                },
                "outscale": 0.01,
                "symlog_inputs": False,
                "name": "actor",
            },
            "critic": {
                "shape": [255],
                "layers": 3,
                "units": 256,
                "act": "SiLU",
                "norm": True,
                "device": device,
                "dist": {"name": "symexp_twohot", "bin_num": 255},
                "outscale": 0.0,
                "symlog_inputs": False,
                "name": "value",
            },
        }
    )


def make_crafter_spaces():
    """Return (obs_space, act_space) for the Crafter environment."""
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(0, 255, OBS_SHAPE_HWC, dtype=np.uint8),
        }
    )
    act_space = gym.spaces.Box(low=0, high=1, shape=(NUM_ACTIONS,), dtype=np.float32)
    act_space.discrete = True
    return obs_space, act_space


def collect_crafter_data(num_steps, seed=42):
    """Collect random-policy transitions from CrafterEnv."""
    from src.environments.crafter import CrafterEnv

    env = CrafterEnv(size=(64, 64), seed=seed)
    transitions = []
    obs = env.reset()
    for _ in range(num_steps):
        action = np.random.randint(0, NUM_ACTIONS)
        next_obs = env.step(action)
        transitions.append(
            {
                "image_chw": obs.image.copy(),
                "image_hwc": obs.image.transpose(1, 2, 0).copy(),
                "action": action,
                "reward": next_obs.reward,
                "is_first": obs.is_first,
                "is_last": next_obs.done,
                "is_terminal": next_obs.done,
            }
        )
        obs = env.reset() if next_obs.done else next_obs
    env.close()
    return transitions


def precompute_batch_starts(num_steps, transitions, seed):
    """Compute reproducible batch start indices for num_steps batches."""
    rng = np.random.RandomState(seed)
    max_start = len(transitions) - SEQ_LEN
    return [rng.randint(0, max_start, size=BATCH_SIZE) for _ in range(num_steps)]
