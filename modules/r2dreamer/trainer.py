"""Training orchestration for R2-Dreamer experiments.

Provides Trainer (training loop), convert_batch (buffer→agent format),
save/load_checkpoint, ObsAdapter (env→buffer/agent bridge), and
habitat_defaults (pre-configured Habitat+CNN settings).
"""

from __future__ import annotations

import csv
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from modules.shared.replay_buffer import BufferConfig, ReplayBuffer


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
# ObsAdapter
# ---------------------------------------------------------------------------

@dataclass
class ObsAdapter:
    """Bridges env observations to agent/buffer, called once per step.

    Default: extracts obs["image"] for buffer (uint8), passes obs dict
    through to agent unchanged.
    """
    buffer_dtype: str = "uint8"
    buffer_shape: tuple[int, ...] = (3, 64, 64)
    normalize_on_sample: bool = True
    on_episode_reset: Callable[[], None] | None = None

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        """Returns (buffer_obs, agent_obs_dict)."""
        return obs_dict["image"], obs_dict


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Controls the training loop (separate from R2DreamerConfig model arch)."""
    output_dir: str = "output/r2dreamer"
    total_steps: int = 10_000_000
    prefill_steps: int = 5000
    log_every: int = 250
    checkpoint_every: int = 50_000
    seed: int = 0

    # WandB (None = disabled)
    wandb_project: str | None = "3d-vla-objectnav"
    wandb_name: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["r2dreamer"])
    # Resume an existing W&B run (e.g. "87u0l6dy"). Requires the run to exist.
    wandb_id: str | None = None

    # Validation (None = disabled)
    val_data: str | None = None
    val_loss_every: int = 10_000

    # Resume from checkpoint (.pkl produced by save_checkpoint). When set,
    # restores agent.{params, opt_state, slow_critic_params, ema_state} and
    # offsets the train loop to start at the checkpoint's step.
    resume_from: str | None = None


# ---------------------------------------------------------------------------
# habitat_defaults
# ---------------------------------------------------------------------------

EpisodeMetricsFn = Callable[..., dict[str, Any]]


def habitat_defaults(env: Any) -> dict[str, Any]:
    """Pre-configured ObsAdapter and episode_metrics_fn for Habitat+CNN.

    Returns dict with keys "obs_adapter" and "episode_metrics_fn".
    """
    from modules.shared.wandb_utils import EpisodeTracker
    tracker = EpisodeTracker(window=100)
    action_names = {0: "stop", 1: "forward", 2: "left", 3: "right"}

    def episode_metrics_fn(
        env: Any, last_obs: dict, episode_reward: float,
        episode_steps: int, action_counts: np.ndarray,
    ) -> dict[str, Any]:
        success = last_obs.get("success", 0.0)
        spl = last_obs.get("spl", 0.0)
        category = getattr(env._env.current_episode, "object_category", "unknown")
        scene_raw = getattr(env._env.current_episode, "scene_id", "")
        path_length = env._path_length
        shortest_path = env._start_geodesic
        path_ratio = path_length / shortest_path if shortest_path > 0 else 0.0

        tracked = tracker.record(
            reward=episode_reward, success=success, spl=spl,
            category=category, scene_id=scene_raw,
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

        # Optional val dataset
        self._val_dataset = None
        if trainer_config.val_data is not None:
            from modules.shared.replay_buffer import ValReplayDataset
            self._val_dataset = ValReplayDataset(
                trainer_config.val_data,
                normalize=self.obs_adapter.normalize_on_sample,
            )

    def run(self) -> None:
        """Execute full training run: prefill + train loop + final checkpoint."""
        tcfg, acfg = self.tcfg, self.acfg

        os.makedirs(tcfg.output_dir, exist_ok=True)
        csv_path = os.path.join(tcfg.output_dir, "metrics.csv")

        rng_key = jax.random.PRNGKey(tcfg.seed)

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
            rng_key = self._train_loop(rng_key, writer, f)

        save_checkpoint(self.agent, tcfg.total_steps, tcfg.output_dir)

        if self._wandb is not None:
            self._wandb.finish()
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

    def _train_loop(self, rng_key: jnp.ndarray, writer: Any, f: Any) -> jnp.ndarray:
        acfg, tcfg = self.acfg, self.tcfg

        start_step = self._resume_step
        print(f"Training from step {start_step} to {tcfg.total_steps}...")
        obs = self.env.reset()
        if self.obs_adapter.on_episode_reset:
            self.obs_adapter.on_episode_reset()
        buffer_obs, agent_obs = self.obs_adapter.transform(obs)

        episode_reward = 0.0
        episode_steps = 0
        episode_count = 0
        action_counts = np.zeros(acfg.num_actions, dtype=int)
        self._t0 = time.time()
        batch_steps = acfg.batch_size * acfg.seq_len
        train_credit = 0.0
        metrics: dict[str, Any] = {}

        for step in range(start_step, tcfg.total_steps):
            rng_key, act_key = jax.random.split(rng_key)
            action = self.agent.act(agent_obs, act_key)
            next_obs = self.env.step(action)
            next_buffer_obs, next_agent_obs = self.obs_adapter.transform(next_obs)

            success = next_obs.get("success", 0.0) > 0
            self.buffer.add(buffer_obs, action, next_obs["reward"],
                            next_obs["done"], terminal=success)
            action_counts[action] += 1
            episode_reward += next_obs["reward"]
            episode_steps += 1

            if next_obs["done"]:
                episode_count += 1
                ep_metrics = self._on_episode_end(
                    next_obs, episode_reward, episode_steps,
                    action_counts, step, writer, f,
                )
                episode_reward = 0.0
                episode_steps = 0
                action_counts = np.zeros(acfg.num_actions, dtype=int)
                obs = self.env.reset()
                if self.obs_adapter.on_episode_reset:
                    self.obs_adapter.on_episode_reset()
                buffer_obs, agent_obs = self.obs_adapter.transform(obs)
            else:
                obs = next_obs
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

            # --- Val loss ---
            if (self._val_dataset is not None
                    and (step + 1) % tcfg.val_loss_every == 0):
                rng_key, val_key = jax.random.split(rng_key)
                self._log_val_loss(val_key, step, writer, f)

            # --- Checkpoint ---
            if (step + 1) % tcfg.checkpoint_every == 0:
                save_checkpoint(self.agent, step + 1, tcfg.output_dir)

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

    def _log_val_loss(
        self, val_key: jnp.ndarray, step: int, writer: Any, f: Any,
    ) -> None:
        val_batch = self._val_dataset.sample(
            self.acfg.batch_size, self.acfg.seq_len)
        val_batch = convert_batch(val_batch, self.acfg.num_actions)
        val_metrics = self.agent.eval_loss(val_batch, val_key)
        val_logged = {f"val/{k}": v for k, v in val_metrics.items()}
        for k, v in val_logged.items():
            writer.writerow([step, k, v])
        f.flush()
        if self._wandb is not None:
            self._wandb.log(val_logged, step=step)
        print(
            f"[step {step:>8d}] VAL: "
            f"total={val_logged.get('val/total_loss', 0):.3f} "
            f"dyn={val_logged.get('val/loss/dyn', 0):.3f} "
            f"rew={val_logged.get('val/loss/rew', 0):.3f}"
        )
