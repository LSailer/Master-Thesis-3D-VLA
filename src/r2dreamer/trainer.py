"""Training orchestration for R2-Dreamer experiments.

Provides Trainer (training loop), convert_batch (buffer→agent format),
save/load_checkpoint, ObsAdapter (env→buffer/agent bridge), and
habitat_defaults (pre-configured Habitat+CNN settings).
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.replay_buffer import BufferConfig, ReplayBuffer
from src.r2dreamer.adapters import ObsAdapter  # noqa: F401 — re-exported for callers
from src.r2dreamer.manifest import write_manifest_end, write_manifest_start


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class Env(Protocol):
    def reset(self) -> dict: ...
    def step(self, action: int) -> dict: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# convert_batch
# ---------------------------------------------------------------------------

def convert_batch(batch: dict[str, jnp.ndarray],
                  num_actions: int) -> dict[str, jnp.ndarray]:
    """Convert replay buffer output to agent training format.

    - actions: int32 (B,T) -> one_hot float32 (B,T,A)
    - dones -> is_last
    - terminals -> is_terminal
    """
    return {
        "obs": batch["obs"],
        "actions": jax.nn.one_hot(batch["actions"], num_actions),
        "rewards": batch["rewards"],
        "is_first": batch["is_first"],
        "is_last": batch["dones"],
        "is_terminal": batch["terminals"],
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(agent: Any, step: int, output_dir: str) -> str:
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
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(path: str) -> dict[str, Any]:
    """Load checkpoint dict from disk. Returns raw dict — caller restores."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Controls the training loop (separate from R2DreamerConfig model arch)."""
    output_dir: str = "output/runs/r2dreamer"
    total_steps: int = 10_000_000
    prefill_steps: int = 5000
    log_every: int = 250
    checkpoint_every: int = 50_000
    seed: int = 0

    # When True, a fully completed run hard-exits the process (os._exit(0))
    # after the final checkpoint, MANIFEST, and W&B are flushed — skipping
    # habitat_sim's GL teardown, which SIGABRTs ("no current context") on some
    # magnum builds and would otherwise poison the exit code of a successful
    # run. Set by the SLURM launcher (env R2DREAMER_HARD_EXIT_ON_FINISH=1);
    # left False for notebook/test callers so they keep the normal close() path
    # and real failures still surface a non-zero exit.
    hard_exit_on_finish: bool = False

    # WandB (None = disabled)
    wandb_project: str | None = "3d-vla-objectnav"
    wandb_name: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["r2dreamer"])
    # Resume an existing W&B run (e.g. "87u0l6dy"). Requires the run to exist.
    wandb_id: str | None = None
    # Resume from checkpoint (.pkl produced by save_checkpoint). When set,
    # restores agent.{params, opt_state, slow_critic_params, ema_state} and
    # offsets the train loop to start at the checkpoint's step.
    resume_from: str | None = None

    # --- Karpathy step-3 diagnostic: overfit a single sampled batch ---
    # When True, the run does the normal prefill, then samples one batch
    # (overfit_batch_size, overfit_seq_len) once, freezes it, and runs
    # agent.train_step on that same batch for overfit_steps iterations.
    # No env rollouts, no validation, no checkpointing.
    overfit_one_batch: bool = False
    overfit_steps: int = 1000
    overfit_batch_size: int = 1
    overfit_seq_len: int = 8
    overfit_min_loss_drop: float = 0.20


# ---------------------------------------------------------------------------
# habitat_defaults
# ---------------------------------------------------------------------------

EpisodeMetricsFn = Callable[..., dict[str, Any]]


def habitat_defaults(env: Any, *, track_collision_rate: bool = False) -> dict[str, Any]:
    """Pre-configured ObsAdapter and episode_metrics_fn for Habitat+CNN.

    Returns dict with keys "obs_adapter" and "episode_metrics_fn".

    Pass track_collision_rate=True for standalone evaluation trackers; train rollouts
    leave it False so the dashboard isn't doubly-noisy.
    """
    from src.shared.wandb_utils import EpisodeTracker
    tracker = EpisodeTracker(window=100, track_collision_rate=track_collision_rate)
    action_names = {0: "stop", 1: "forward", 2: "left", 3: "right"}

    def episode_metrics_fn(
        env: Any, last_obs: dict, episode_reward: float,
        episode_steps: int, action_counts: np.ndarray,
    ) -> dict[str, Any]:
        success = last_obs.get("success", 0.0)
        spl = last_obs.get("spl", 0.0)
        softspl = last_obs.get("softspl", 0.0)
        dtg = last_obs.get("dtg", 0.0)
        collision_rate = last_obs.get("collision_rate", 0.0)
        category = getattr(env._env.current_episode, "object_category", "unknown")
        scene_raw = getattr(env._env.current_episode, "scene_id", "")
        path_length = env._path_length
        shortest_path = env._start_geodesic
        path_ratio = path_length / shortest_path if shortest_path > 0 else 0.0

        tracked = tracker.record(
            reward=episode_reward, success=success, spl=spl,
            category=category, scene_id=scene_raw,
            softspl=softspl, dtg=dtg, collision_rate=collision_rate,
        )

        action_pcts = action_counts / max(episode_steps, 1)
        return {
            **tracked,
            "episode/steps": episode_steps,
            "episode/path_length": path_length,
            "episode/shortest_path": shortest_path,
            "episode/path_ratio": path_ratio,
            "episode_reset": 1,
            **{f"action/{action_names[i]}_pct": float(action_pcts[i])
               for i in range(len(action_counts))},
        }

    return {
        "obs_adapter": ObsAdapter(),
        "episode_metrics_fn": episode_metrics_fn,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Training loop: prefill -> train (train-ratio) -> log -> checkpoint.

    Args:
        agent: R2DreamerAgent instance.
        env: Environment (Crafter, Habitat, etc.).
        agent_config: R2DreamerConfig (for batch_size, seq_len, train_ratio, etc.).
        trainer_config: TrainerConfig (for loop control, logging, checkpointing).
        obs_adapter: ObsAdapter (bridges env obs to buffer/agent).
        episode_metrics_fn: Optional callback called at episode end.
            Signature: (env, last_obs, episode_reward, episode_steps, action_counts) -> dict
    """

    def __init__(
        self,
        agent: Any,
        env: Env,
        agent_config: Any,
        trainer_config: TrainerConfig,
        obs_adapter: ObsAdapter | None = None,
        episode_metrics_fn: EpisodeMetricsFn | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.acfg = agent_config
        self.tcfg = trainer_config
        self.obs_adapter = obs_adapter or ObsAdapter()
        self.episode_metrics_fn = episode_metrics_fn
        # Build buffer from adapter settings
        self.buffer = ReplayBuffer(BufferConfig(
            capacity=agent_config.buffer_capacity,
            obs_shape=self.obs_adapter.buffer_shape,
            obs_dtype=self.obs_adapter.buffer_dtype,
            normalize_obs=self.obs_adapter.normalize_on_sample,
        ))

        # Resume from checkpoint (overwrite freshly-initialised agent state).
        self._resume_step = 0
        if trainer_config.resume_from is not None:
            if not os.path.exists(trainer_config.resume_from):
                raise FileNotFoundError(
                    f"resume_from points at non-existent path: {trainer_config.resume_from}"
                )
            state = load_checkpoint(trainer_config.resume_from)
            self.agent.params = jax.tree.map(jnp.asarray, state["params"])
            self.agent.opt_state = jax.tree.map(jnp.asarray, state["opt_state"])
            self.agent.slow_critic_params = jax.tree.map(
                jnp.asarray, state["slow_critic_params"]
            )
            self.agent.ema_state = jax.tree.map(jnp.asarray, state["ema_state"])
            self._resume_step = int(state["step"])
            print(
                f"Resumed agent state from {trainer_config.resume_from} "
                f"at step {self._resume_step}"
            )

        # Optional WandB
        self._wandb = None
        if trainer_config.wandb_project is not None:
            import wandb
            self._wandb = wandb
            init_kwargs: dict[str, Any] = dict(
                project=trainer_config.wandb_project,
                name=trainer_config.wandb_name,
                config=vars(agent_config) if hasattr(agent_config, "__dict__") else {},
                tags=trainer_config.wandb_tags,
            )
            if trainer_config.wandb_id is not None:
                # resume="must" fails loudly if the run-id does not exist,
                # which is what we want — silent re-creation orphans runs.
                init_kwargs.update(id=trainer_config.wandb_id, resume="must")
            wandb.init(**init_kwargs)

    def run(self) -> None:
        """Execute full training run: prefill + train loop + final checkpoint."""
        tcfg, acfg = self.tcfg, self.acfg

        os.makedirs(tcfg.output_dir, exist_ok=True)
        csv_path = os.path.join(tcfg.output_dir, "metrics.csv")

        # MANIFEST.json — emit on start, finalize in finally with run status.
        cfg_snapshot = asdict(acfg) if is_dataclass(acfg) else dict(vars(acfg))
        write_manifest_start(Path(tcfg.output_dir), cfg_snapshot)
        status = "failed"

        rng_key = jax.random.PRNGKey(tcfg.seed)

        try:
            # Append to existing CSV when resuming so the prior rows survive.
            is_resume = self._resume_step > 0
            csv_mode = "a" if is_resume else "w"
            with open(csv_path, csv_mode, newline="") as f:
                writer = csv.writer(f)
                if not is_resume:
                    writer.writerow(["step", "metric", "value"])

                if is_resume:
                    # Skip random prefill — the trained policy collects on-policy
                    # transitions in _train_loop until buffer >= batch_steps.
                    # env.reset() / extractor.reset() fire at _train_loop entry.
                    print(f"Resume mode: skipping prefill, jumping to step {self._resume_step}")
                else:
                    self._prefill(rng_key, writer, f)
                if tcfg.overfit_one_batch:
                    rng_key = self._overfit_loop(rng_key, writer, f)
                else:
                    rng_key = self._train_loop(rng_key, writer, f)

            save_checkpoint(self.agent, tcfg.total_steps, tcfg.output_dir)
            status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            write_manifest_end(Path(tcfg.output_dir), status)
            if self._wandb is not None:
                self._wandb.finish()
            if tcfg.hard_exit_on_finish and status == "completed":
                # habitat_sim's GL teardown SIGABRTs ("no current context") on
                # some magnum builds, poisoning the exit code AFTER the run has
                # fully completed (checkpoint + manifest + W&B already flushed
                # above). Skip the aborting close and exit cleanly. Failures
                # fall through to close() so their non-zero exit and traceback
                # survive and the smoke gate still catches real breakage.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
            self.env.close()


    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prefill(self, rng_key: jnp.ndarray, writer: Any, f: Any) -> None:
        acfg, tcfg = self.acfg, self.tcfg
        print(f"Prefilling {tcfg.prefill_steps} steps...")

        obs = self.env.reset()
        if self.obs_adapter.on_episode_reset:
            self.obs_adapter.on_episode_reset()
        buffer_obs, _ = self.obs_adapter.transform(obs)

        for _ in range(tcfg.prefill_steps):
            action = np.random.randint(0, acfg.num_actions)
            next_obs = self.env.step(action)
            next_buffer_obs, _ = self.obs_adapter.transform(next_obs)

            success = next_obs.get("success", 0.0) > 0
            self.buffer.add(buffer_obs, action, next_obs["reward"],
                            next_obs["done"], terminal=success)

            if next_obs["done"]:
                obs = self.env.reset()
                if self.obs_adapter.on_episode_reset:
                    self.obs_adapter.on_episode_reset()
                buffer_obs, _ = self.obs_adapter.transform(obs)
            else:
                obs = next_obs
                buffer_obs = next_buffer_obs

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def _reset_train_episode(self) -> tuple[dict, np.ndarray | dict[str, np.ndarray], dict]:
        obs = self.env.reset()
        if self.obs_adapter.on_episode_reset:
            self.obs_adapter.on_episode_reset()
        buffer_obs, agent_obs = self.obs_adapter.transform(obs)
        return obs, buffer_obs, agent_obs

    def _zero_episode_counters(self) -> tuple[float, int, np.ndarray]:
        return 0.0, 0, np.zeros(self.acfg.num_actions, dtype=int)

    def _record_train_transition(
        self,
        *,
        buffer_obs: np.ndarray | dict[str, np.ndarray],
        action: int,
        next_obs: dict,
    ) -> None:
        success = next_obs.get("success", 0.0) > 0
        self.buffer.add(
            buffer_obs, action, next_obs["reward"], next_obs["done"],
            terminal=success,
        )

    def _finish_train_episode(
        self,
        *,
        last_obs: dict,
        episode_reward: float,
        episode_steps: int,
        action_counts: np.ndarray,
        step: int,
        writer: Any,
        f: Any,
    ) -> tuple[
        dict,
        np.ndarray | dict[str, np.ndarray],
        dict,
        float,
        int,
        np.ndarray,
    ]:
        self._on_episode_end(
            last_obs, episode_reward, episode_steps, action_counts, step, writer, f,
        )

        episode_reward, episode_steps, action_counts = self._zero_episode_counters()
        obs, buffer_obs, agent_obs = self._reset_train_episode()
        return (
            obs, buffer_obs, agent_obs, episode_reward, episode_steps,
            action_counts,
        )

    def _train_loop(self, rng_key: jnp.ndarray, writer: Any, f: Any) -> jnp.ndarray:
        acfg, tcfg = self.acfg, self.tcfg

        start_step = self._resume_step
        print(f"Training from step {start_step} to {tcfg.total_steps}...")
        _obs, buffer_obs, agent_obs = self._reset_train_episode()
        episode_reward, episode_steps, action_counts = self._zero_episode_counters()
        self._t0 = time.time()
        batch_steps = acfg.batch_size * acfg.seq_len
        train_credit = 0.0
        metrics: dict[str, Any] = {}

        for step in range(start_step, tcfg.total_steps):
            rng_key, act_key = jax.random.split(rng_key)
            action = self.agent.act(agent_obs, act_key)
            next_obs = self.env.step(action)
            next_buffer_obs, next_agent_obs = self.obs_adapter.transform(next_obs)

            self._record_train_transition(
                buffer_obs=buffer_obs, action=action, next_obs=next_obs,
            )
            action_counts[action] += 1
            episode_reward += next_obs["reward"]
            episode_steps += 1

            if next_obs["done"]:
                (
                    _obs, buffer_obs, agent_obs, episode_reward, episode_steps,
                    action_counts,
                ) = self._finish_train_episode(
                    last_obs=next_obs,
                    episode_reward=episode_reward,
                    episode_steps=episode_steps,
                    action_counts=action_counts,
                    step=step,
                    writer=writer,
                    f=f,
                )
            else:
                _obs = next_obs
                buffer_obs = next_buffer_obs
                agent_obs = next_agent_obs

            # --- Train ---
            if self.buffer.size >= batch_steps:
                train_credit += acfg.train_ratio / batch_steps
                while train_credit >= 1.0:
                    rng_key, train_key = jax.random.split(rng_key)
                    batch = self.buffer.sample(acfg.batch_size, acfg.seq_len)
                    batch = convert_batch(batch, acfg.num_actions)
                    metrics = self.agent.train_step(batch, train_key)
                    train_credit -= 1.0

                if step % tcfg.log_every == 0 and metrics:
                    self._log_train_metrics(metrics, step, writer, f)
                    if getattr(acfg, "decoder", False):
                        self._maybe_log_recon(batch, step)

            # --- Checkpoint ---
            if (step + 1) % tcfg.checkpoint_every == 0:
                save_checkpoint(self.agent, step + 1, tcfg.output_dir)

        return rng_key

    # ------------------------------------------------------------------
    # Overfit-one-batch diagnostic loop (Karpathy step 3)
    # ------------------------------------------------------------------

    def _overfit_loop(self, rng_key: jnp.ndarray, writer: Any, f: Any) -> jnp.ndarray:
        """Freeze one sampled batch and call train_step on it repeatedly.

        Proves the full stack (encoder -> RSSM -> heads) can memorise a real
        trajectory. If loss does not drop monotonically, the gradient path is
        broken — no amount of production wall-clock will save the run.

        Disables env rollouts, validation, and checkpointing.
        """
        tcfg = self.tcfg

        if self.buffer.size < tcfg.overfit_batch_size * tcfg.overfit_seq_len:
            raise RuntimeError(
                f"overfit_one_batch: buffer too small "
                f"({self.buffer.size} < {tcfg.overfit_batch_size}*{tcfg.overfit_seq_len}). "
                f"Increase --prefill."
            )

        # Sample once, freeze, reuse.
        batch_raw = self.buffer.sample(tcfg.overfit_batch_size, tcfg.overfit_seq_len)
        batch = convert_batch(batch_raw, self.acfg.num_actions)
        print(
            f"Overfit mode: cached batch "
            f"B={tcfg.overfit_batch_size} T={tcfg.overfit_seq_len}; "
            f"running {tcfg.overfit_steps} train_step iterations."
        )

        if tcfg.overfit_steps < 1:
            raise ValueError(f"overfit_steps must be >= 1, got {tcfg.overfit_steps}")

        self._t0 = time.time()
        first_loss = last_loss = 0.0
        for step in range(tcfg.overfit_steps):
            rng_key, train_key = jax.random.split(rng_key)
            metrics = self.agent.train_step(batch, train_key)
            last_loss = metrics["total_loss"]
            if step == 0:
                first_loss = last_loss

            if step % tcfg.log_every == 0 or step == tcfg.overfit_steps - 1:
                self._log_train_metrics(metrics, step, writer, f)

        loss_drop = (first_loss - last_loss) / max(abs(first_loss), 1e-12)
        writer.writerow([tcfg.overfit_steps - 1, "verify/overfit_loss_drop", loss_drop])
        writer.writerow([
            tcfg.overfit_steps - 1,
            "verify/overfit_pass",
            float(loss_drop >= tcfg.overfit_min_loss_drop),
        ])
        f.flush()
        print(
            f"Overfit verify: first_loss={first_loss:.6g} "
            f"last_loss={last_loss:.6g} drop={loss_drop:.1%} "
            f"required={tcfg.overfit_min_loss_drop:.1%}"
        )
        if loss_drop < tcfg.overfit_min_loss_drop:
            raise RuntimeError(
                "overfit_one_batch verification failed: total_loss did not drop "
                f"by at least {tcfg.overfit_min_loss_drop:.1%}. "
                "Do not launch a production run until this passes."
            )

        return rng_key

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _on_episode_end(
        self, last_obs: dict, episode_reward: float, episode_steps: int,
        action_counts: np.ndarray, step: int, writer: Any, f: Any,
    ) -> dict[str, Any]:
        if self.episode_metrics_fn is not None:
            ep_metrics = self.episode_metrics_fn(
                self.env, last_obs, episode_reward, episode_steps, action_counts,
            )
        else:
            ep_metrics = {"episode/reward": episode_reward}

        for k, v in ep_metrics.items():
            writer.writerow([step, k, v])
        f.flush()

        if self._wandb is not None:
            self._wandb.log(ep_metrics, step=step)

        # Console summary
        sr = ep_metrics.get("metrics/sr", "")
        sr_str = f" SR={sr:.3f}" if isinstance(sr, float) else ""
        print(
            f"[step {step:>8d}] reward={episode_reward:.2f}"
            f" steps={episode_steps}{sr_str}"
        )
        return ep_metrics

    def _log_train_metrics(
        self, metrics: dict, step: int, writer: Any, f: Any,
    ) -> None:
        for k, v in metrics.items():
            writer.writerow([step, k, v])
        f.flush()

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        elapsed = time.time() - self._t0
        steps_this_run = step + 1 - self._resume_step
        fps = steps_this_run / elapsed if elapsed > 0 else 0
        print(
            f"[step {step:>8d}/{self.tcfg.total_steps}] "
            f"total={metrics.get('total_loss', 0):.3f} "
            f"dyn={metrics.get('loss/dyn', 0):.3f} "
            f"rew={metrics.get('loss/rew', 0):.3f} "
            f"policy={metrics.get('loss/policy', 0):.3f} "
            f"fps={fps:.0f}"
        )

    def _maybe_log_recon(self, batch: dict, step: int) -> None:
        """Log decoder input/reconstruction image pairs to W&B (3D-51).

        No-op unless a decoder is configured and W&B is active. Decodes the
        sampled training batch and logs up to 4 side-by-side ``input | recon``
        panels so the learned hybrid representation can be eyeballed during a run.
        """
        if self._wandb is None or not getattr(self.acfg, "decoder", False):
            return
        pair = self.agent.reconstruct(batch)
        if pair is None:
            return
        target, recon = pair  # (B*T, 3, 64, 64) in [0, 1]
        n = min(4, target.shape[0])
        images = []
        for i in range(n):
            tgt = np.transpose(target[i], (1, 2, 0))  # CHW -> HWC
            rec = np.transpose(recon[i], (1, 2, 0))
            combo = np.concatenate([tgt, rec], axis=1)  # side by side
            combo = np.clip(combo * 255.0, 0, 255).astype(np.uint8)
            images.append(self._wandb.Image(combo, caption=f"input | recon ({i})"))
        self._wandb.log({"decoder/reconstructions": images}, step=step)
