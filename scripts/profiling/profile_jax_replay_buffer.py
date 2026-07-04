"""Probe a JAX-only replay buffer path against the current NumPy replay buffer.

The experiment keeps replay storage in CPU JAX ``Ref`` arrays: adapter output can
be a JAX array, replay lives in host RAM, and samples leave replay as JAX arrays.
Run it before replacing ``src.buffer.replay_buffer.ReplayBuffer``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.replay_buffer import ReplayBatch, ReplayBuffer, ReplayTransition
from src.r2dreamer.agent import R2DreamerAgent
from src.configs.config import R2DreamerConfig

ObsShape: TypeAlias = tuple[int, ...] | Mapping[str, tuple[int, ...]]
ObsDType: TypeAlias = str | Mapping[str, str]
ObsNormalize: TypeAlias = bool | Mapping[str, bool]
ObsTree: TypeAlias = jnp.ndarray | dict[str, jnp.ndarray]


@dataclass(frozen=True)
class ReplaySpec:
    """Explicit replay shape needed by the experimental JAX Ref buffer."""

    capacity: int
    obs_shape: ObsShape
    obs_dtype: ObsDType = "uint8"
    normalize_obs: ObsNormalize = True


@dataclass(frozen=True)
class ExperimentInputs:
    """Synthetic transitions shared by both replay implementations."""

    config: ReplaySpec
    obs: ObsTree
    actions: list[int]
    rewards: list[float]
    episode_ends: list[bool]


@dataclass(frozen=True)
class FieldSpec:
    """One replay field."""

    shape: tuple[int, ...]
    dtype: jnp.dtype
    normalize: bool
    keep_uint8: bool


def _pct(values: list[float], p_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p_value / 100.0) * (len(ordered) - 1)))
    return ordered[min(len(ordered) - 1, max(0, idx))]


def _summ(values: list[float]) -> dict[str, float]:
    return {
        "n": float(len(values)),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p90_ms": _pct(values, 90),
        "p95_ms": _pct(values, 95),
        "min_ms": min(values) if values else 0.0,
        "max_ms": max(values) if values else 0.0,
    }


def _block_tree(tree: object) -> None:
    for value in jax.tree.leaves(tree):
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()


def _jax_dtype(name: str) -> jnp.dtype:
    if name == "uint8":
        return jnp.uint8
    if name == "float16":
        return jnp.float16
    return jnp.float32


def _mapping_value(value: object, key: str, default: object) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return value


def _field_specs(config: ReplaySpec) -> dict[str, FieldSpec] | None:
    if not isinstance(config.obs_shape, Mapping):
        if not isinstance(config.obs_dtype, str):
            raise TypeError("single-field config needs string obs_dtype")
        if not isinstance(config.normalize_obs, bool):
            raise TypeError("single-field config needs bool normalize_obs")
        return None

    specs: dict[str, FieldSpec] = {}
    for key, shape in config.obs_shape.items():
        dtype_name = str(_mapping_value(config.obs_dtype, key, "float32"))
        normalize = bool(_mapping_value(config.normalize_obs, key, False))
        specs[key] = FieldSpec(
            shape=tuple(shape),
            dtype=_jax_dtype(dtype_name),
            normalize=normalize and dtype_name == "uint8",
            keep_uint8=dtype_name == "uint8" and not normalize,
        )
    return specs


def _single_spec(config: ReplaySpec) -> FieldSpec:
    if not isinstance(config.obs_shape, tuple):
        raise TypeError("single-field config needs tuple obs_shape")
    if not isinstance(config.obs_dtype, str):
        raise TypeError("single-field config needs string obs_dtype")
    if not isinstance(config.normalize_obs, bool):
        raise TypeError("single-field config needs bool normalize_obs")
    return FieldSpec(
        shape=config.obs_shape,
        dtype=_jax_dtype(config.obs_dtype),
        normalize=config.normalize_obs and config.obs_dtype == "uint8",
        keep_uint8=False,
    )


def _convert_obs_field(
    obs: jnp.ndarray,
    normalize: bool,
    keep_uint8: bool,
) -> jnp.ndarray:
    if normalize:
        return obs.astype(jnp.float32) / 255.0
    if keep_uint8:
        return obs
    return obs.astype(jnp.float32)


@dataclass(frozen=True)
class ReplayRefs:
    """Mutable JAX replay arrays."""

    obs: jax.Ref | dict[str, jax.Ref]
    actions: jax.Ref
    rewards: jax.Ref
    episode_ends: jax.Ref


class JaxRefReplayBuffer:
    """Experimental JAX Ref replay storage on CPU RAM."""

    def __init__(self, config: ReplaySpec) -> None:
        if not hasattr(jax, "new_ref"):
            raise RuntimeError("JAX Ref API missing; run with uv's newer JAX environment")
        self.capacity = config.capacity
        self.idx = 0
        self.size = 0
        device = jax.devices("cpu")[0]

        def zeros(shape: tuple[int, ...], dtype: jnp.dtype) -> jnp.ndarray:
            return jax.device_put(jnp.zeros(shape, dtype=dtype), device)

        field_specs = _field_specs(config)
        if field_specs is None:
            spec = _single_spec(config)
            self._obs_specs: FieldSpec | dict[str, FieldSpec] = spec
            obs = jax.new_ref(zeros((self.capacity, *spec.shape), spec.dtype))
        else:
            self._obs_specs = field_specs
            obs = {
                key: jax.new_ref(zeros((self.capacity, *spec.shape), spec.dtype))
                for key, spec in field_specs.items()
            }
        self._refs = ReplayRefs(
            obs=obs,
            actions=jax.new_ref(zeros((self.capacity,), jnp.int32)),
            rewards=jax.new_ref(zeros((self.capacity,), jnp.float32)),
            episode_ends=jax.new_ref(zeros((self.capacity,), jnp.bool_)),
        )
        self._add_jit = jax.jit(self._add_one)
        self._sample_at_jit = jax.jit(self._sample_at, static_argnames=("seq_len",))

    def add(
        self,
        obs: ObsTree,
        action: int,
        reward: float,
        episode_end: bool,
    ) -> None:
        """Append one transition."""
        transition = {
            "obs": obs,
            "action": jnp.asarray(action, dtype=jnp.int32),
            "reward": jnp.asarray(reward, dtype=jnp.float32),
            "episode_end": jnp.asarray(episode_end, dtype=jnp.bool_),
        }
        self._add_jit(jnp.asarray(self.idx, dtype=jnp.int32), transition)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _add_one(self, idx: jnp.ndarray, transition: dict[str, object]) -> None:
        """JIT body for one in-place write."""
        obs = transition["obs"]
        refs = self._refs
        if isinstance(refs.obs, Mapping):
            if not isinstance(obs, Mapping):
                raise TypeError("mapping replay buffer requires mapping obs")
            for key, storage in refs.obs.items():
                specs = self._obs_specs
                if not isinstance(specs, Mapping):
                    raise TypeError("mapping replay specs missing")
                storage[idx] = jnp.asarray(obs[key], dtype=specs[key].dtype)
        else:
            if isinstance(obs, Mapping):
                raise TypeError("single replay buffer requires array obs")
            spec = self._obs_specs
            if not isinstance(spec, FieldSpec):
                raise TypeError("single replay spec missing")
            refs.obs[idx] = jnp.asarray(obs, dtype=spec.dtype)
        refs.actions[idx] = transition["action"]
        refs.rewards[idx] = transition["reward"]
        refs.episode_ends[idx] = transition["episode_end"]

    def sample(self, batch_size: int, seq_len: int, key: jnp.ndarray) -> dict[str, object]:
        """Sample sequence starts with JAX RNG and gather them from Ref storage."""
        starts = self._sample_starts(batch_size, seq_len, key)
        return self._sample_at_jit(starts, seq_len=seq_len)

    def _sample_starts(
        self,
        batch_size: int,
        seq_len: int,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.size < self.capacity:
            n_valid = self.size - seq_len + 1
            if n_valid <= 0:
                raise AssertionError("Not enough data in buffer")
            return jax.random.randint(key, (batch_size,), 0, n_valid, dtype=jnp.int32)
        n_new = max(0, self.idx - seq_len + 1)
        n_old = max(0, self.capacity - seq_len - self.idx + 1)
        n_valid = n_new + n_old
        if n_valid <= 0:
            raise AssertionError("Not enough contiguous data in buffer")
        raw = jax.random.randint(key, (batch_size,), 0, n_valid, dtype=jnp.int32)
        return jnp.where(raw < n_new, raw, raw - n_new + self.idx)

    def _sample_at(self, starts: jnp.ndarray, seq_len: int) -> dict[str, object]:
        indices = starts[:, None] + jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        refs = self._refs
        episode_ends_b = refs.episode_ends[indices]
        is_first = jnp.zeros(episode_ends_b.shape, dtype=jnp.float32)
        is_first = is_first.at[:, 0].set(1.0)
        is_first = is_first.at[:, 1:].set(
            episode_ends_b[:, :-1].astype(jnp.float32)
        )
        return {
            "obs": self._gather_obs(indices),
            "actions": refs.actions[indices].astype(jnp.int32),
            "rewards": refs.rewards[indices].astype(jnp.float32),
            "is_episode_end": episode_ends_b.astype(jnp.float32),
            "is_first": is_first,
        }

    def _gather_obs(self, indices: jnp.ndarray) -> ObsTree:
        refs = self._refs
        if isinstance(refs.obs, Mapping):
            specs = self._obs_specs
            if not isinstance(specs, Mapping):
                raise TypeError("mapping replay specs missing")
            return {
                key: _convert_obs_field(
                    storage[indices], specs[key].normalize, specs[key].keep_uint8
                )
                for key, storage in refs.obs.items()
            }
        spec = self._obs_specs
        if not isinstance(spec, FieldSpec):
            raise TypeError("single replay spec missing")
        return _convert_obs_field(refs.obs[indices], spec.normalize, spec.keep_uint8)


def _config(layout: str, capacity: int) -> ReplaySpec:
    if layout == "cnn":
        return ReplaySpec(capacity, (3, 64, 64), "uint8", True)
    if layout == "vggt":
        return ReplaySpec(capacity, (4116,), "float32", False)
    if layout == "hybrid":
        return ReplaySpec(
            capacity,
            {"image": (3, 64, 64), "wp_cp": (4116,)},
            {"image": "uint8", "wp_cp": "float32"},
            {"image": True, "wp_cp": False},
        )
    raise ValueError(layout)


def _make_obs(layout: str, capacity: int, key: jnp.ndarray) -> ObsTree:
    if layout == "cnn":
        return jax.random.randint(key, (capacity, 3, 64, 64), 0, 256, dtype=jnp.uint8)
    if layout == "vggt":
        return jax.random.normal(key, (capacity, 4116), dtype=jnp.float32)
    if layout == "hybrid":
        key_image, key_wpcp = jax.random.split(key)
        return {
            "image": jax.random.randint(
                key_image, (capacity, 3, 64, 64), 0, 256, dtype=jnp.uint8
            ),
            "wp_cp": jax.random.normal(key_wpcp, (capacity, 4116), dtype=jnp.float32),
        }
    raise ValueError(layout)


def _tree_index(tree: ObsTree, idx: int) -> ObsTree:
    if isinstance(tree, Mapping):
        return {key: value[idx] for key, value in tree.items()}
    return tree[idx]


def _to_numpy_obs(tree: ObsTree) -> np.ndarray | dict[str, np.ndarray]:
    if isinstance(tree, Mapping):
        return {key: np.asarray(value) for key, value in tree.items()}
    return np.asarray(tree)


def _fill_numpy_buffer(
    buffer: ReplayBuffer,
    obs: ObsTree,
    actions: list[int],
    rewards: list[float],
    episode_ends: list[bool],
) -> dict[str, float]:
    from src.environments.observation import ObservationFrame

    times: list[float] = []
    for idx, action in enumerate(actions):
        frame = ObservationFrame(
            image=np.empty((0,), dtype=np.uint8),
            is_first=False,
            previous_action=action,
            reward=rewards[idx],
            done=episode_ends[idx],
        )
        t0 = time.perf_counter()
        buffer.add(
            ReplayTransition.from_frame(_to_numpy_obs(_tree_index(obs, idx)), frame)
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    return _summ(times)


def _fill_jax_buffer(
    buffer: JaxRefReplayBuffer,
    obs: ObsTree,
    actions: list[int],
    rewards: list[float],
    episode_ends: list[bool],
) -> dict[str, float]:
    times: list[float] = []
    for idx, action in enumerate(actions):
        t0 = time.perf_counter()
        buffer.add(_tree_index(obs, idx), action, rewards[idx], episode_ends[idx])
        times.append((time.perf_counter() - t0) * 1000.0)
    _block_tree(buffer.sample(1, 1, jax.random.PRNGKey(0)))
    return _summ(times)


def _convert_legacy_batch(batch: dict[str, object], num_actions: int) -> ReplayBatch:
    """Pack a raw JAX-ref replay dict into an agent-ready ``ReplayBatch``.

    ``JaxRefReplayBuffer.sample`` returns a raw dict whose ``actions`` are int32
    action ids, whereas the agent expects one-hot float actions. This mirrors
    the field layout produced by ``ReplayBuffer.sample``.

    Args:
        batch: Raw replay dict with keys ``obs``, ``actions`` (int32,
            ``(B, T)``), ``rewards``, ``is_first``, and ``is_episode_end``.
        num_actions: Number of discrete actions for one-hot encoding.

    Returns:
        A ``ReplayBatch`` with ``(B, T, num_actions)`` float one-hot actions and
        float rewards/masks.
    """
    actions = jax.nn.one_hot(
        jnp.asarray(batch["actions"]), num_actions, dtype=jnp.float32
    )
    return ReplayBatch(
        obs=batch["obs"],
        actions=actions,
        rewards=jnp.asarray(batch["rewards"], dtype=jnp.float32),
        is_first=jnp.asarray(batch["is_first"], dtype=jnp.float32),
        is_episode_end=jnp.asarray(batch["is_episode_end"], dtype=jnp.float32),
    )


def _measure_numpy_sample(
    buffer: ReplayBuffer,
    batch_size: int,
    seq_len: int,
    iters: int,
    warmup: int,
) -> dict[str, float]:
    times: list[float] = []
    for idx in range(iters + warmup):
        t0 = time.perf_counter()
        batch = buffer.sample(batch_size, seq_len)
        _block_tree(batch)
        if idx >= warmup:
            times.append((time.perf_counter() - t0) * 1000.0)
    return _summ(times)


def _measure_jax_sample(
    buffer: JaxRefReplayBuffer,
    batch_size: int,
    seq_len: int,
    iters: int,
    warmup: int,
) -> dict[str, float]:
    times: list[float] = []
    for idx in range(iters + warmup):
        t0 = time.perf_counter()
        batch = _convert_legacy_batch(
            buffer.sample(batch_size, seq_len, jax.random.PRNGKey(1000 + idx)), 4
        )
        _block_tree(batch)
        if idx >= warmup:
            times.append((time.perf_counter() - t0) * 1000.0)
    return _summ(times)


def _tiny_agent() -> R2DreamerAgent:
    """Build a small CNN agent to prove sampled batches train."""
    cfg = R2DreamerConfig(
        obs_shape=(3, 64, 64),
        num_actions=4,
        batch_size=2,
        seq_len=4,
        deter_size=32,
        hidden_size=32,
        stoch_classes=4,
        stoch_discrete=4,
        blocks=2,
        encoder_depth=4,
        encoder_mults=(2, 2),
        mlp_units=32,
        mlp_layers_actor=1,
        mlp_layers_critic=1,
        twohot_bins=31,
        imagination_horizon=3,
        compute_dtype="float32",
    )
    return R2DreamerAgent(cfg, jax.random.PRNGKey(7))


def _measure_train_step(batch: dict[str, object], iters: int, warmup: int) -> dict[str, float]:
    agent = _tiny_agent()
    times: list[float] = []
    for idx in range(iters + warmup):
        t0 = time.perf_counter()
        metrics = agent.train_step(batch, jax.random.PRNGKey(2000 + idx))
        if not np.isfinite(metrics["total_loss"]):
            raise RuntimeError(f"non-finite loss: {metrics['total_loss']}")
        if idx >= warmup:
            times.append((time.perf_counter() - t0) * 1000.0)
    return _summ(times)


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=("cnn", "vggt", "hybrid"), default="cnn")
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--sample-iters", type=int, default=30)
    parser.add_argument("--train-iters", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--output",
        default="output/profiles/jax_ref_replay_profile.json",
    )
    return parser.parse_args()


def _make_inputs(args: argparse.Namespace) -> ExperimentInputs:
    """Create deterministic synthetic replay transitions."""
    if args.capacity < args.seq_len:
        raise ValueError("capacity must be >= seq_len")
    key_obs, key_actions, key_rewards = jax.random.split(jax.random.PRNGKey(0), 3)
    return ExperimentInputs(
        config=_config(args.layout, args.capacity),
        obs=_make_obs(args.layout, args.capacity, key_obs),
        actions=np.asarray(
            jax.random.randint(key_actions, (args.capacity,), 0, 4, dtype=jnp.int32)
        ).tolist(),
        rewards=np.asarray(jax.random.normal(key_rewards, (args.capacity,))).tolist(),
        episode_ends=[False] * args.capacity,
    )


def _measure_replay(args: argparse.Namespace, inputs: ExperimentInputs) -> dict[str, object]:
    """Run replay timing and optional tiny train-step check."""
    numpy_buffer = ReplayBuffer(capacity=inputs.config.capacity, num_actions=4)
    jax_buffer = JaxRefReplayBuffer(inputs.config)
    runs: dict[str, dict[str, object]] = {
        "numpy_replay": {
            "add_from_jax_adapter_output": _fill_numpy_buffer(
                numpy_buffer,
                inputs.obs,
                inputs.actions,
                inputs.rewards,
                inputs.episode_ends,
            ),
            "sample_convert": _measure_numpy_sample(
                numpy_buffer,
                args.batch_size,
                args.seq_len,
                args.sample_iters,
                args.warmup,
            ),
        },
        "jax_ref_replay": {
            "add_from_jax_adapter_output": _fill_jax_buffer(
                jax_buffer,
                inputs.obs,
                inputs.actions,
                inputs.rewards,
                inputs.episode_ends,
            ),
            "sample_convert": _measure_jax_sample(
                jax_buffer,
                args.batch_size,
                args.seq_len,
                args.sample_iters,
                args.warmup,
            ),
        },
    }
    if args.layout == "cnn" and not args.skip_train:
        runs["numpy_replay"]["tiny_train_step"] = _measure_train_step(
            numpy_buffer.sample(2, 4), args.train_iters, args.warmup
        )
        runs["jax_ref_replay"]["tiny_train_step"] = _measure_train_step(
            _convert_legacy_batch(jax_buffer.sample(2, 4, jax.random.PRNGKey(99)), 4),
            args.train_iters,
            args.warmup,
        )
    return {
        "meta": {
            "layout": args.layout,
            "capacity": args.capacity,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_ref_note": "Refs are mutable, but impure jit dispatch is documented slower.",
        },
        "runs": runs,
    }


def main() -> None:
    """Run the profiling experiment."""
    args = _parse_args()
    results = _measure_replay(args, _make_inputs(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
