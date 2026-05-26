#!/usr/bin/env python
"""Offline training of the *external* (PyTorch) R2Dreamer — 3D-45 adapter.

Trains the ORIGINAL PyTorch R2Dreamer (``external/r2dreamer/``) offline from the
canonical replay buffer (``data/offline_buffer/``) on the WP/CP vector readout
(``z_wp_cp.npz``, 4116-d), with no live env and no VGGT forward pass. This is the
infra that the 3D-46 baseline runs depend on: it produces a PyTorch number that
is apples-to-apples with the JAX 3D-26 offline results.

Why a wrapper instead of ``external/r2dreamer/train.py``
-------------------------------------------------------
The stock external entry point is env-driven (``make_envs`` -> ``OnlineTrainer``,
torchrl ``LazyTensorStorage`` filled by live rollouts) and logs only to
TensorBoard/jsonl. This wrapper instead:

  * feeds the 4116-d WP/CP vector through the MLP encoder branch by declaring an
    obs space ``{"vector": Box(4116,)}`` and setting ``encoder.mlp_keys='vector'``
    / ``cnn_keys='$^'`` (the match-nothing regex that disables the CNN);
  * replays the fixed offline buffer with the SAME sampling semantics as the JAX
    ``OfflineBufferDataset`` — contiguous windows that may straddle episode
    boundaries, ``is_first`` reset wherever ``done`` flipped, ``is_terminal=done``,
    and a zeroed initial latent each batch (the external R2-Dreamer latent
    write-back is a no-op offline, matching the JAX path which carries no
    cross-batch latent state);
  * reuses ``agent.update()`` verbatim, so the optimizer / GradScaler / AGC /
    scheduler behaviour is identical to the online external trainer;
  * optionally mirrors scalars to W&B in the same project as the JAX runs.

The default ``rep_loss="r2dreamer"`` (Barlow-Twins redundancy reduction) carries
NO decoder — matching the decoder-free JAX R2Dreamer — so there is no
reconstruction NLL on either side. See 3D-45 / 3D-46.

IMPORTANT — run under the external venv (it has torchrl/tensordict/gymnasium):

    external/r2dreamer/.venv/bin/python scripts/r2dreamer/train_external_offline.py ...

Smoke (NaN-free 100 steps on CPU, no W&B):

    external/r2dreamer/.venv/bin/python scripts/r2dreamer/train_external_offline.py \\
        --encoder wp_cp --seed 0 --steps 100 \\
        --buffer-dir data/offline_buffer_smoke --output-dir /tmp/ext_smoke \\
        --device cpu --batch-size 4 --seq-len 16

Real run (GPU, one of seeds {0,1,2} for 3D-46):

    external/r2dreamer/.venv/bin/python scripts/r2dreamer/train_external_offline.py \\
        --encoder wp_cp --seed 0 --steps 500000 \\
        --buffer-dir data/offline_buffer --output-dir output/3d46/ext-wp_cp-seed0 \\
        --device cuda:0 --wandb --wandb-name ext-wp_cp-seed0
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXT = REPO_ROOT / "external" / "r2dreamer"

# Fairness-contract defaults — must match 3D-26 (the JAX offline runs).
SHARED = {
    "batch_size": 16,
    "seq_len": 64,
    "imag_horizon": 15,
}

# z_*.npz file + flat feature dim per readout. Aggregator is wired but out of
# scope for 3D-45/3D-46 (WP/CP only); kept so a future readout is a one-liner.
ENCODER_SPECS = {
    "wp_cp": {"z_file": "z_wp_cp.npz", "obs_dim": 4116},
    "aggregator": {"z_file": "z_aggregator.npz", "obs_dim": 3072},
}

NUM_ACTIONS = 4  # Habitat objectnav: stop / forward / left / right


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--encoder", choices=list(ENCODER_SPECS), default="wp_cp")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=500_000, help="Total grad steps.")
    p.add_argument("--buffer-dir", default="data/offline_buffer")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-size", default="size12M")
    p.add_argument("--batch-size", type=int, default=SHARED["batch_size"])
    p.add_argument("--seq-len", type=int, default=SHARED["seq_len"])
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    # compile defaults to the model config (True); forced off on CPU.
    p.add_argument("--compile", dest="compile", action="store_true", default=None)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--wandb", action="store_true", help="Enable W&B (off by default).")
    p.add_argument("--wandb-project", default="3d-vla-objectnav-offline-ablation")
    p.add_argument("--wandb-name", default=None)
    p.add_argument(
        "--wandb-tags",
        default="offline-ablation,3d-24,framework:pytorch-external",
        help="Comma-separated. `variant:<encoder>` is appended automatically.",
    )
    return p


# ---------------------------------------------------------------------------
# Config / spaces
# ---------------------------------------------------------------------------
def build_model_config(model_size: str, device: str, compile_flag: bool | None):
    """Compose the external model config without hydra's env layer.

    The shipped ``configs/model/_base_.yaml`` interpolates ``${env.encoder.*}``
    and ``${device}``; we supply those nodes here and point the encoder/decoder
    at the single ``vector`` obs key (CNN disabled via the ``$^`` regex).
    """
    from omegaconf import OmegaConf

    base = OmegaConf.load(EXT / "configs" / "model" / "_base_.yaml")
    size = OmegaConf.load(EXT / "configs" / "model" / f"{model_size}.yaml")
    if "defaults" in size:
        del size["defaults"]  # hydra-only key; we merge _base_ explicitly
    model = OmegaConf.merge(base, size)

    top = OmegaConf.create(
        {
            "device": device,
            "env": {
                "encoder": {"mlp_keys": "vector", "cnn_keys": "$^"},
                "decoder": {"mlp_keys": "vector", "cnn_keys": "$^"},
            },
            "model": model,
        }
    )
    OmegaConf.resolve(top)
    model_cfg = top.model
    if compile_flag is not None:
        model_cfg.compile = bool(compile_flag)
    # Dreamer mutates a few config fields in place (actor.shape, actor.dist, ...).
    OmegaConf.set_struct(model_cfg, False)
    return model_cfg


def make_spaces(obs_dim: int, num_actions: int):
    """Build a gymnasium obs Dict + a discrete (one-hot) action Box.

    Mirrors ``envs/wrappers.OneHotAction``: a Box with a ``.discrete`` marker so
    ``Dreamer`` selects the one-hot actor head. Only the ``vector`` key is
    declared, so ``MultiEncoder`` routes it (and nothing else) to the MLP encoder.
    """
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {"vector": gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)}
    )
    act_space = gym.spaces.Box(low=0.0, high=1.0, shape=(num_actions,), dtype=np.float32)
    act_space.discrete = True
    return obs_space, act_space


# ---------------------------------------------------------------------------
# Offline replay buffer (drop-in for external Buffer's sample/update/count)
# ---------------------------------------------------------------------------
class OfflineVectorBuffer:
    """Replays ``data/offline_buffer`` windows with JAX-identical semantics.

    Exposes the ``(data, index, initial) = sample()`` / ``update(index, stoch,
    deter)`` / ``count()`` interface that ``Dreamer.update`` calls, so the agent's
    training step is reused unchanged. Sampling matches
    ``src/buffer/offline_buffer_dataset.OfflineBufferDataset``: contiguous windows
    over a single split's index range, ``is_first[:,0]=True`` plus wherever
    ``done`` flipped, ``is_terminal=is_last=done``, and a zeroed initial latent
    (episode reset is driven entirely by ``is_first``).
    """

    def __init__(
        self,
        buffer_dir: Path,
        encoder: str,
        split: str,
        *,
        batch_size: int,
        seq_len: int,
        device,
        seed: int,
        deter_size: int,
        stoch_classes: int,
        stoch_discrete: int,
    ) -> None:
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.B = int(batch_size)
        self.L = int(seq_len)
        self._init_dims = (int(stoch_classes), int(stoch_discrete), int(deter_size))

        spec = ENCODER_SPECS[encoder]
        meta = _load_metadata(buffer_dir)
        self.metadata = meta
        n = meta["n_completed_steps"]
        heldout_start = meta["heldout_start_episode"]

        skeleton = np.load(buffer_dir / "trajectory_skeleton.npz")
        try:
            actions = skeleton["action"][:n]
            rewards = skeleton["reward"][:n]
            dones = skeleton["done"][:n]
            episode_ids = skeleton["episode_id"][:n]
        finally:
            skeleton.close()

        z = _load_features(buffer_dir / spec["z_file"])[:n]

        if split == "train":
            mask = episode_ids < heldout_start
        elif split == "heldout":
            mask = episode_ids >= heldout_start
        else:
            raise ValueError(f"unknown split: {split!r}")
        if not mask.any():
            raise ValueError(
                f"split={split!r} empty (heldout_start={heldout_start}, "
                f"episodes={meta['num_episodes']})"
            )
        idx = np.where(mask)[0]
        if not np.all(np.diff(idx) == 1):
            raise ValueError(
                f"split={split!r} is not a contiguous transition range — the "
                f"collector reordered episodes? Aborting to avoid wrap-around."
            )
        start, end = int(idx[0]), int(idx[-1]) + 1

        self.obs = z[start:end]  # kept float16 in RAM; cast per-batch
        self.actions = actions[start:end]
        self.rewards = rewards[start:end]
        self.dones = dones[start:end]
        self.size = int(self.obs.shape[0])
        self.obs_dim = int(self.obs.shape[1])
        if self.size < self.L:
            raise ValueError(f"split has {self.size} steps < seq_len {self.L}")
        self._rng = np.random.default_rng(seed)

        print(
            f"OfflineVectorBuffer[{split}]: {self.size} steps, obs={self.obs.shape} "
            f"{self.obs.dtype}, {encoder} from {buffer_dir}"
        )

    def sample(self):
        torch = self._torch
        n_valid = self.size - self.L + 1
        starts = self._rng.integers(0, n_valid, size=self.B)
        win = starts[:, None] + np.arange(self.L)[None, :]  # (B, L)

        dev = self.device
        obs = torch.as_tensor(self.obs[win].astype(np.float32), device=dev)
        act_idx = torch.as_tensor(self.actions[win].astype(np.int64), device=dev)
        action = torch.nn.functional.one_hot(act_idx, NUM_ACTIONS).to(torch.float32)
        reward = torch.as_tensor(
            self.rewards[win].astype(np.float32), device=dev
        ).unsqueeze(-1)

        dones = self.dones[win]  # (B, L) bool
        is_first = np.zeros_like(dones)
        is_first[:, 0] = True
        is_first[:, 1:] = dones[:, :-1]
        is_first_t = torch.as_tensor(is_first, device=dev).unsqueeze(-1)
        done_t = torch.as_tensor(dones.astype(np.float32), device=dev).unsqueeze(-1)

        from tensordict import TensorDict

        data = TensorDict(
            {
                "vector": obs,
                "action": action,
                "reward": reward,
                "is_first": is_first_t,
                "is_last": done_t,
                "is_terminal": done_t,
            },
            batch_size=(self.B, self.L),
            device=dev,
        )
        s_classes, s_discrete, deter = self._init_dims
        initial = (
            torch.zeros(self.B, s_classes, s_discrete, device=dev),
            torch.zeros(self.B, deter, device=dev),
        )
        return data, win, initial

    def update(self, index, stoch, deter):  # noqa: D401 - offline no-op
        """No-op: offline replay carries no cross-batch latent state (matches JAX)."""

    def count(self):
        return self.size


def _load_metadata(buffer_dir: Path) -> dict:
    """Read collection_metadata.json; tolerate a missing file (all-train)."""
    skeleton = np.load(buffer_dir / "trajectory_skeleton.npz")
    try:
        episode_ids = skeleton["episode_id"]
        n_rows = int(episode_ids.shape[0])
    finally:
        skeleton.close()

    meta_path = buffer_dir / "collection_metadata.json"
    raw = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n_completed = int(raw.get("n_completed_steps", n_rows))
    if n_completed > n_rows:
        raise ValueError(
            f"metadata n_completed_steps={n_completed} > skeleton rows {n_rows}"
        )
    num_episodes = int(episode_ids[:n_completed].max()) + 1 if n_completed else 0
    heldout = raw.get("heldout_split", {})
    return {
        "n_completed_steps": n_completed,
        "num_episodes": num_episodes,
        "heldout_start_episode": int(
            heldout.get("episode_id_start_inclusive", num_episodes)
        ),
        "code_sha": raw.get("code_sha"),
        "checkpoint_sha256": raw.get("checkpoint_sha256"),
        "collect_seed": int(raw.get("collect_seed", -1)),
    }


def _load_features(path: Path) -> np.ndarray:
    data = np.load(path)
    try:
        key = "features" if "features" in data.files else data.files[0]
        return data[key]
    finally:
        data.close()


# ---------------------------------------------------------------------------
# Logger (TensorBoard/jsonl via tools.Logger, + optional W&B mirror)
# ---------------------------------------------------------------------------
def make_logger(output_dir: Path, wandb_run):
    import tools  # external/r2dreamer/tools.py (EXT is on sys.path)

    if wandb_run is None:
        return tools.Logger(output_dir)

    class _WandbLogger(tools.Logger):
        def write(self, step, fps=False):
            scalars = dict(self._scalars)  # snapshot before super() clears
            super().write(step, fps=fps)
            wandb_run.log(scalars, step=int(step))

    return _WandbLogger(output_dir)


# ---------------------------------------------------------------------------
# Offline trainer
# ---------------------------------------------------------------------------
class OfflineTrainer:
    """Pure offline loop: ``steps`` grad updates, no env, no eval episodes."""

    def __init__(self, agent, buffer, logger, *, steps, log_every, checkpoint_every, output_dir):
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.steps = int(steps)
        self.log_every = int(log_every)
        self.checkpoint_every = int(checkpoint_every)
        self.output_dir = Path(output_dir)

    def run(self):
        import torch

        import tools

        self.agent.train()
        t0 = time.time()
        last = t0
        for step in range(self.steps):
            metrics = self.agent.update(self.buffer)
            scalars = {k: _to_float(v) for k, v in metrics.items()}

            bad = [k for k, v in scalars.items() if not math.isfinite(v)]
            if bad:
                raise RuntimeError(f"non-finite metric(s) at step {step}: {bad}")

            if step % self.log_every == 0:
                now = time.time()
                for k, v in scalars.items():
                    self.logger.scalar(f"train/{k}", v)
                self.logger.scalar("perf/sps", self.log_every / max(now - last, 1e-6))
                self.logger.scalar("perf/elapsed_min", (now - t0) / 60.0)
                self.logger.write(step, fps=True)
                last = now

            if self.checkpoint_every and step > 0 and step % self.checkpoint_every == 0:
                self._save(step)

        self._save(self.steps, latest=True)
        print(f"Done: {self.steps} grad steps in {(time.time() - t0) / 60:.1f} min")

    def _save(self, step, latest=False):
        import torch

        import tools

        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": step,
            "agent_state_dict": self.agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(self.agent),
        }
        torch.save(payload, self.output_dir / f"checkpoint_{step:09d}.pt")
        if latest:
            torch.save(payload, self.output_dir / "latest.pt")
        print(f"  saved checkpoint @ step {step}")


def _to_float(v) -> float:
    try:
        return float(v.item()) if hasattr(v, "item") else float(v)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Make the external package importable (it imports siblings by bare name).
    if str(EXT) not in sys.path:
        sys.path.insert(0, str(EXT))

    import torch

    import tools
    from dreamer import Dreamer

    buffer_dir = _resolve(args.buffer_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    compile_flag = args.compile
    if device.startswith("cpu"):
        compile_flag = False  # reduce-overhead/cudagraphs require CUDA

    tools.set_seed_everywhere(args.seed)
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    spec = ENCODER_SPECS[args.encoder]
    model_cfg = build_model_config(args.model_size, device, compile_flag)
    obs_space, act_space = make_spaces(spec["obs_dim"], NUM_ACTIONS)

    print(
        f"3D-45 external offline: encoder={args.encoder} seed={args.seed} "
        f"steps={args.steps} device={device} compile={bool(model_cfg.compile)}"
    )
    agent = Dreamer(model_cfg, obs_space, act_space).to(device)

    train_buf = OfflineVectorBuffer(
        buffer_dir,
        args.encoder,
        split="train",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        seed=args.seed,
        deter_size=int(model_cfg.rssm.deter),
        stoch_classes=int(model_cfg.rssm.stoch),
        stoch_discrete=int(model_cfg.rssm.discrete),
    )
    if train_buf.obs_dim != spec["obs_dim"]:
        raise ValueError(
            f"buffer obs_dim={train_buf.obs_dim} != encoder {args.encoder} "
            f"expected {spec['obs_dim']}"
        )

    # W&B (off by default; real 3D-46 runs pass --wandb).
    wandb_run = None
    if args.wandb:
        import wandb
        from omegaconf import OmegaConf

        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        tags.append(f"variant:{args.encoder}")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"ext-{args.encoder}-seed{args.seed}",
            tags=tags,
            config=OmegaConf.to_container(model_cfg, resolve=True),
        )

    # Reproducibility record.
    from omegaconf import OmegaConf

    run_config = {
        "issue": "3D-45/3D-46",
        "framework": "pytorch-external",
        "encoder": args.encoder,
        "obs_dim": spec["obs_dim"],
        "num_actions": NUM_ACTIONS,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "device": device,
        "model_size": args.model_size,
        "rep_loss": str(model_cfg.rep_loss),
        "compile": bool(model_cfg.compile),
        "buffer_dir": str(buffer_dir),
        "train_split_size": train_buf.size,
        "buffer_metadata": train_buf.metadata,
        "code_sha_main": _git_sha(REPO_ROOT),
        "code_sha_external": _git_sha(EXT),
        "wandb_run_id": wandb_run.id if wandb_run is not None else None,
        "wandb_run_url": wandb_run.url if wandb_run is not None else None,
        "model_config": OmegaConf.to_container(model_cfg, resolve=True),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    logger = make_logger(output_dir, wandb_run)
    OfflineTrainer(
        agent,
        train_buf,
        logger,
        steps=args.steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=output_dir,
    ).run()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
