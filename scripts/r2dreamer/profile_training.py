"""Fine-grained phase-timing diagnostic for the R2-Dreamer training loop.

Covers CNN and VGGT encoders back-to-back. Emits per-phase timing
distributions (7 phases) + KV-cache audit for the VGGT path.
See docs/plans/l4-profiling.md and issue #74.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import jax
import jax.numpy as jnp
import numpy as np

from src.shared.profiling import timed
from src.buffer.replay_buffer import BufferConfig, ReplayBuffer
from src.environments.habitat import build_habitat_env
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.adapters import ObsAdapter
from src.r2dreamer.trainer import convert_batch

VGGT_FEATURE_DIM = 4116  # 37*37*3 + 9 — matches the vggt encoder (run id habitat-l1-vggt)


# 7-phase list locked in PRD #74.
# Slice 1 populates only: env_step, wm_inference, buffer_add, wm_training.
# Slice 2 adds: vggt_forward, vggt_wrapper, jax_upload.
ALL_PHASES = (
    "env_step",
    "vggt_forward",
    "vggt_wrapper",
    "jax_upload",
    "wm_inference",
    "buffer_add",
    "wm_training",
)


def init_phase_times() -> dict[str, list[float]]:
    return {p: [] for p in ALL_PHASES}


@dataclass
class RunResult:
    encoder: str
    phase_times: dict[str, list[float]]
    kv_audit: dict[str, int]
    episodes: dict[str, Any]
    config: dict[str, Any]


def _build_cnn(
    seed: int, curriculum_path: str | None,
) -> tuple[Any, Any, Any, ObsAdapter, R2DreamerConfig, Any]:
    from src.r2dreamer.world_model.encoders import ConvEncoder
    agent_cfg = R2DreamerConfig(
        encoder_type="cnn",
        encoder_module_cls=ConvEncoder,
        obs_shape=(3, 64, 64),
        num_actions=4,
        batch_size=16,
        seq_len=64,
        buffer_capacity=100_000,
        seed=seed,
    )
    env = build_habitat_env(
        obs_shape=(3, 64, 64),
        curriculum_path=curriculum_path,
        curriculum_mode="train",
    )
    buffer = ReplayBuffer(agent_cfg)
    init_key = jax.random.PRNGKey(seed)
    agent = R2DreamerAgent(agent_cfg, init_key)
    obs_adapter = ObsAdapter()
    return env, agent, buffer, obs_adapter, agent_cfg, None


def _flatten_vggt(out: dict) -> jnp.ndarray:
    """Concatenate VGGT output into a single (4116,) float32 feature vector.

    Duplicated from the vggt encoder's flatten readout to keep this self-contained.
    """
    wp = out["world_points"].reshape(-1)  # (4107,)
    cp = out["camera_pose"]              # (9,)
    return jnp.concatenate([wp, cp]).astype(jnp.float32)


def _build_vggt(
    seed: int, curriculum_path: str | None, render_resolution: int,
    compile: bool = False,
    compile_mode: str | None = None,
) -> tuple[Any, Any, Any, ObsAdapter, R2DreamerConfig, Any]:
    """Construct env (518 res), agent (VGGT encoder), buffer (float32 features),
    obs_adapter with on_episode_reset hook wired to extractor.reset, and the
    raw VGGTFeatureExtractor (needed by the loop for instrumented extract calls).
    """
    from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor
    from src.r2dreamer.world_model.encoders import VGGTEncoder as VGGTEncoderModule

    agent_cfg = R2DreamerConfig(
        encoder_type="vggt",
        encoder_module_cls=VGGTEncoderModule,
        obs_shape=(VGGT_FEATURE_DIM,),
        num_actions=4,
        batch_size=16,
        seq_len=64,
        buffer_capacity=100_000,
        seed=seed,
    )
    env = build_habitat_env(
        obs_shape=(3, render_resolution, render_resolution),
        curriculum_path=curriculum_path,
        curriculum_mode="train",
    )
    print(f"Loading InfiniteVGGT model (compile={compile}, compile_mode={compile_mode!r})...")
    extractor = VGGTFeatureExtractor(
        device="cuda", compile=compile, compile_mode=compile_mode,
    )
    print("InfiniteVGGT loaded.")

    # ObsAdapter wired so Trainer-style reset hook fires extractor.reset.
    # transform() is NOT used — the loop inlines extract(..., phase_times=...).
    obs_adapter = ObsAdapter(
        buffer_dtype="float32",
        buffer_shape=(VGGT_FEATURE_DIM,),
        normalize_on_sample=False,
        on_episode_reset=extractor.reset,
    )
    buffer = ReplayBuffer(BufferConfig(
        capacity=agent_cfg.buffer_capacity,
        obs_shape=(VGGT_FEATURE_DIM,),
        obs_dtype="float32",
        normalize_obs=False,
    ))
    init_key = jax.random.PRNGKey(seed)
    agent = R2DreamerAgent(agent_cfg, init_key)
    return env, agent, buffer, obs_adapter, agent_cfg, extractor


def run_loop(
    encoder: str,
    prefill_steps: int,
    acting_steps: int,
    seed: int = 0,
    curriculum_path: str | None = None,
    render_resolution: int = 518,
    compile: bool = False,
    compile_mode: str | None = None,
) -> RunResult:
    if encoder == "cnn":
        env, agent, buffer, obs_adapter, acfg, extractor = _build_cnn(
            seed, curriculum_path,
        )
    elif encoder == "vggt":
        env, agent, buffer, obs_adapter, acfg, extractor = _build_vggt(
            seed, curriculum_path, render_resolution, compile=compile,
            compile_mode=compile_mode,
        )
    else:
        raise ValueError(f"Unknown Observation Preparation mode: {encoder!r}")

    phase_times = init_phase_times()
    reset_count = 0
    boundary_count = 0
    total_steps = 0
    total_reward = 0.0
    episode_count = 0

    rng_key = jax.random.PRNGKey(seed + 1)

    def transform(obs_dict: dict) -> tuple[np.ndarray, dict]:
        """Build (buffer_obs, agent_obs). For VGGT, time VGGT phases inline."""
        if encoder != "vggt":
            return obs_adapter.transform(obs_dict)

        # VGGT path: extract() records vggt_forward + vggt_wrapper via kwarg.
        # _flatten_vggt is tiny CPU work — fold into vggt_wrapper's last entry
        # so vggt_wrapper represents the full "output → numpy feature" cost.
        out = extractor.extract(obs_dict["image"], phase_times=phase_times)
        t0 = time.perf_counter()
        features_jax = _flatten_vggt(out)
        features_np = np.asarray(features_jax)
        phase_times["vggt_wrapper"][-1] += (time.perf_counter() - t0) * 1000.0
        agent_obs = {"features": features_jax, "is_first": obs_dict.get("is_first", False)}
        return features_np, agent_obs

    def do_reset() -> tuple[dict, np.ndarray, dict]:
        nonlocal reset_count, boundary_count
        obs = env.reset()
        boundary_count += 1
        if obs_adapter.on_episode_reset is not None:
            obs_adapter.on_episode_reset()
            reset_count += 1
        buffer_obs, agent_obs = transform(obs)
        return obs, buffer_obs, agent_obs

    def probe_jax_upload(features: np.ndarray) -> None:
        """Measure numpy → JAX GPU transfer cost (proxy for what happens
        inside agent.act for VGGT features). Only makes sense for VGGT."""
        if encoder != "vggt":
            return
        with timed(phase_times, "jax_upload"):
            probe = jnp.asarray(features[None])
            probe.block_until_ready()

    print(f"[{encoder}] Prefilling {prefill_steps} steps (random actions)...")
    obs, buffer_obs, agent_obs = do_reset()
    for _ in range(prefill_steps):
        action = int(np.random.randint(0, acfg.num_actions))

        with timed(phase_times, "env_step"):
            next_obs = env.step(action)
        next_buffer_obs, next_agent_obs = transform(next_obs)

        with timed(phase_times, "buffer_add"):
            buffer.add(
                buffer_obs, action, next_obs["reward"],
                next_obs["done"], terminal=(next_obs.get("success", 0.0) > 0),
            )
        total_steps += 1
        total_reward += next_obs["reward"]

        if next_obs["done"]:
            episode_count += 1
            obs, buffer_obs, agent_obs = do_reset()
        else:
            obs, buffer_obs, agent_obs = next_obs, next_buffer_obs, next_agent_obs

    print(f"[{encoder}] Acting {acting_steps} steps with interleaved training...")
    batch_steps = acfg.batch_size * acfg.seq_len
    train_credit = 0.0
    for _ in range(acting_steps):
        rng_key, act_key = jax.random.split(rng_key)

        # Probe the numpy→JAX upload cost (VGGT only; no-op for CNN).
        if encoder == "vggt":
            probe_jax_upload(buffer_obs)

        with timed(phase_times, "wm_inference"):
            action = agent.act(agent_obs, act_key)

        with timed(phase_times, "env_step"):
            next_obs = env.step(action)
        next_buffer_obs, next_agent_obs = transform(next_obs)

        with timed(phase_times, "buffer_add"):
            buffer.add(
                buffer_obs, action, next_obs["reward"],
                next_obs["done"], terminal=(next_obs.get("success", 0.0) > 0),
            )
        total_steps += 1
        total_reward += next_obs["reward"]

        if next_obs["done"]:
            episode_count += 1
            obs, buffer_obs, agent_obs = do_reset()
        else:
            obs, buffer_obs, agent_obs = next_obs, next_buffer_obs, next_agent_obs

        if buffer.size >= batch_steps:
            train_credit += acfg.train_ratio / batch_steps
            while train_credit >= 1.0:
                rng_key, train_key = jax.random.split(rng_key)
                with timed(phase_times, "wm_training"):
                    batch = buffer.sample(acfg.batch_size, acfg.seq_len)
                    batch = convert_batch(batch, acfg.num_actions)
                    _ = agent.train_step(batch, train_key)
                train_credit -= 1.0

    env.close()

    # KV-cache audit: for VGGT, the reset hook must fire exactly once per
    # env.reset(). For CNN there is no hook so reset_count stays 0 — skip.
    if encoder == "vggt":
        assert reset_count == boundary_count, (
            f"KV-cache audit FAILED: reset_count={reset_count} "
            f"!= boundary_count={boundary_count}. Every env.reset() must "
            f"trigger VGGTFeatureExtractor.reset()."
        )

    return RunResult(
        encoder=encoder,
        phase_times=phase_times,
        kv_audit={"reset_count": reset_count, "boundary_count": boundary_count},
        episodes={
            "count": episode_count,
            "total_steps": total_steps,
            "avg_reward": total_reward / max(1, total_steps),
        },
        config={
            "prefill_steps": prefill_steps,
            "acting_steps": acting_steps,
            "seed": seed,
            "curriculum_path": curriculum_path,
            "render_resolution": render_resolution if encoder == "vggt" else None,
        },
    )


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def aggregate(result: RunResult) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for phase, values in result.phase_times.items():
        if values:
            out[phase] = {
                "mean_ms": mean(values),
                "p50_ms": _pct(values, 0.50),
                "p95_ms": _pct(values, 0.95),
                "n_calls": float(len(values)),
            }
        else:
            out[phase] = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "n_calls": 0.0}
    return out


def format_report(results_by_encoder: dict[str, RunResult]) -> str:
    aggs = {enc: aggregate(r) for enc, r in results_by_encoder.items()}
    has_cnn = "cnn" in aggs
    has_vggt = "vggt" in aggs

    header = ["phase"]
    if has_cnn:
        header += ["cnn_mean_ms", "cnn_p50_ms", "cnn_p95_ms"]
    if has_vggt:
        header += ["vggt_mean_ms", "vggt_p50_ms", "vggt_p95_ms"]
    if has_cnn and has_vggt:
        header.append("delta_ms")

    col_w = 16
    lines = [
        " | ".join(f"{h:>{col_w}}" for h in header),
        "-+-".join("-" * col_w for _ in header),
    ]
    for phase in ALL_PHASES:
        row = [phase]
        if has_cnn:
            a = aggs["cnn"][phase]
            row += [f"{a['mean_ms']:.3f}", f"{a['p50_ms']:.3f}", f"{a['p95_ms']:.3f}"]
        if has_vggt:
            a = aggs["vggt"][phase]
            row += [f"{a['mean_ms']:.3f}", f"{a['p50_ms']:.3f}", f"{a['p95_ms']:.3f}"]
        if has_cnn and has_vggt:
            d = aggs["vggt"][phase]["mean_ms"] - aggs["cnn"][phase]["mean_ms"]
            row.append(f"{d:+.3f}")
        lines.append(" | ".join(f"{c:>{col_w}}" for c in row))

    lines.append("")
    for enc, r in results_by_encoder.items():
        lines.append(
            f"[{enc}] episodes={r.episodes['count']} "
            f"total_steps={r.episodes['total_steps']} "
            f"avg_reward={r.episodes['avg_reward']:.3f}"
        )
        lines.append(
            f"[{enc}] kv_audit: reset_count={r.kv_audit['reset_count']} "
            f"boundary_count={r.kv_audit['boundary_count']}"
        )
    return "\n".join(lines)


def save_json(results_by_encoder: dict[str, RunResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"vggt_vs_cnn_{ts}.json"
    payload = {
        enc: {
            "phase_times": r.phase_times,
            "aggregate": aggregate(r),
            "kv_audit": r.kv_audit,
            "episodes": r.episodes,
            "config": r.config,
        }
        for enc, r in results_by_encoder.items()
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="R2-Dreamer training-loop profiler")
    parser.add_argument(
        "--observation-preparation",
        choices=("cnn", "vggt"),
        required=True,
        dest="observation_preparation",
        help="Observation Preparation input mode to profile.",
    )
    parser.add_argument("--prefill_steps", type=int, default=2000)
    parser.add_argument("--acting_steps", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="output/methods/profiling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum_path", type=str, default=None)
    parser.add_argument(
        "--render_resolution", type=int, default=518,
        help="Habitat render resolution (VGGT only; ignored for CNN).",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Apply torch.compile to VGGT sub-modules (VGGT only).",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default=None,
        help=(
            "torch.compile mode forwarded to all three sub-module compiles "
            "(VGGT only; ignored unless --compile is set). Omit to use "
            "torch's default mode (current shipped behaviour)."
        ),
    )
    args = parser.parse_args()

    result = run_loop(
        encoder=args.observation_preparation,
        prefill_steps=args.prefill_steps,
        acting_steps=args.acting_steps,
        seed=args.seed,
        curriculum_path=args.curriculum_path,
        render_resolution=args.render_resolution,
        compile=args.compile,
        compile_mode=args.compile_mode,
    )

    json_path = save_json({args.observation_preparation: result}, Path(args.output_dir))
    print()
    print(format_report({args.observation_preparation: result}))
    print()
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
