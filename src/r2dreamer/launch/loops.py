"""Training-loop orchestration as plain functions (ADR 0006).

The run shape — prefill -> train (train-ratio) -> val cadence -> checkpoint —
lives here at the composition root, not inside a class in the model package.
Every function takes protocol-typed collaborators (``ExperienceSource``,
``RunLoggerLike``, an agent), so tests drive them directly with fakes.

``run_training`` is the top-level scaffold (manifest / CSV / W&B lifecycle,
final checkpoint, ``hard_exit_on_finish``); the launcher wires collectors and
calls it. Loop knobs live in ``TrainerConfig`` (``src.configs.config``).
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import wandb as _wandb_module

from src.buffer.replay_buffer import ReplayBatch
from src.configs.config import R2DreamerConfig, TrainerConfig
from src.r2dreamer.agent import materialize_metrics
from src.r2dreamer.checkpointing import (
    config_snapshot,
    load_checkpoint,
    save_checkpoint,
)
from src.r2dreamer.experience import EpisodeSummary, ExperienceSource
from src.r2dreamer.manifest import write_manifest_end, write_manifest_start
from src.shared.video_utils import log_episode_video

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class R2DreamerAgentLike(Protocol):
    """Interface the loops need from an R2Dreamer-style agent.

    Using a protocol is stricter than ``Any`` while avoiding a hard dependency
    on the concrete ``R2DreamerAgent`` class. Tests and future agent variants
    can still be passed to the loops if they expose the same public contract.
    """

    cfg: Any
    train_state: Any
    params: Any
    opt_state: Any
    slow_critic_params: Any
    ema_state: Any

    def initial_act_state(self) -> Any: ...

    def train_step(
        self,
        train_state: Any,
        batch: ReplayBatch,
        rng_key: jnp.ndarray,
        *,
        materialize: bool = True,
    ) -> tuple[Any, dict[str, Any]]: ...

    def act(
        self,
        params: Any,
        obs: Any,
        is_first: Any,
        state: Any,
        rng_key: jnp.ndarray,
        training: Any = True,
    ) -> tuple[jnp.ndarray, Any]: ...

    # ``(target, recon)`` as device arrays, or None without a decoder. The
    # second element is the raw Flax ``apply`` result: its static type is a
    # union the stubs cannot narrow, and the loops only device_get and index it,
    # so pinning it tighter here would only lie about the implementation.
    def reconstruct(self, batch: Any) -> tuple[jnp.ndarray, Any] | None: ...


class RunLoggerLike(Protocol):
    """Metric/video sinks the loops write to; no file or W&B types leak in."""

    @property
    def wandb_active(self) -> bool: ...
    def start_timing(self, start_step: int) -> None: ...
    def log_episode(self, episode: EpisodeSummary, step: int) -> None: ...
    def log_video(self, key: str, frames: list[np.ndarray], step: int) -> None: ...
    def log_train_metrics(self, metrics: dict[str, Any], step: int) -> None: ...
    def log_reconstructions(
        self, target: np.ndarray, recon: np.ndarray, step: int
    ) -> None: ...
    def log_val_metrics(self, metrics: dict[str, Any], step: int, elapsed: float) -> None: ...
    def write_row(self, step: int, key: str, value: Any) -> None: ...


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------


class RunLogger:
    """All run sinks in one place: metrics.csv, W&B, console, MANIFEST.json.

    Owns the CSV/W&B/manifest lifecycle so the loops never touch file handles
    or the ``wandb`` module. Construct right before the run starts (writes the
    manifest-start entry and opens metrics.csv); call :meth:`finish` in a
    ``finally`` block.

    Args:
        agent_config: R2DreamerConfig, snapshotted into the manifest and W&B.
        trainer_config: TrainerConfig (output_dir, W&B knobs, cadences).
        resume: Whether this run resumes a previous one — appends to the
            existing metrics.csv instead of truncating it.
    """

    def __init__(
        self,
        agent_config: R2DreamerConfig,
        trainer_config: TrainerConfig,
        *,
        resume: bool = False,
    ) -> None:
        self.acfg = agent_config
        self.tcfg = trainer_config
        os.makedirs(trainer_config.output_dir, exist_ok=True)

        # MANIFEST.json — emit on start, finalized by finish() with run status.
        write_manifest_start(
            Path(trainer_config.output_dir), config_snapshot(agent_config)
        )

        self._wandb = None
        if trainer_config.wandb_project is not None:
            self._wandb = _wandb_module
            init_kwargs: dict[str, Any] = {
                "project": trainer_config.wandb_project,
                "name": trainer_config.wandb_name,
                "config": config_snapshot(agent_config),
                "tags": trainer_config.wandb_tags,
            }
            if trainer_config.wandb_id is not None:
                # resume="must" fails loudly if the run-id does not exist,
                # which is what we want — silent re-creation orphans runs.
                init_kwargs.update(id=trainer_config.wandb_id, resume="must")
            self._wandb.init(**init_kwargs)

        csv_path = os.path.join(trainer_config.output_dir, "metrics.csv")
        # Append to the existing CSV when resuming so the prior rows survive.
        self._file = open(
            csv_path, "a" if resume else "w", newline="", encoding="utf-8"
        )
        self._writer = csv.writer(self._file)
        if not resume:
            self._writer.writerow(["step", "metric", "value"])

        self._t0 = 0.0
        self._last_log_time = 0.0
        self._last_log_step = 0
        self._start_step = 0

    @property
    def wandb_active(self) -> bool:
        """Whether a W&B run is attached to this logger."""
        return self._wandb is not None

    def start_timing(self, start_step: int) -> None:
        """Anchor the fps counters at the training loop's first step."""
        self._t0 = time.time()
        self._last_log_time = self._t0
        self._last_log_step = start_step - 1
        self._start_step = start_step

    def write_row(self, step: int, key: str, value: Any) -> None:
        """Write one metrics.csv row and flush."""
        self._writer.writerow([step, key, value])
        self._file.flush()

    def log_episode(self, episode: EpisodeSummary, step: int) -> None:
        """Log one finished episode's metrics to CSV, W&B, and console."""
        for k, v in episode.metrics.items():
            self._writer.writerow([step, k, v])
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(episode.metrics, step=step)

        sr = episode.metrics.get("metrics/sr", "")
        sr_str = f" SR={sr:.3f}" if isinstance(sr, float) else ""
        print(
            f"[step {step:>8d}] reward={episode.reward:.2f}"
            f" steps={episode.steps}{sr_str}"
        )

    def log_video(self, key: str, frames: list[np.ndarray], step: int) -> None:
        """Log an episode video to W&B (no-op when W&B is off)."""
        log_episode_video(self._wandb, key, frames, step)

    def log_train_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log train-step metrics plus derived perf/fps counters."""
        now = time.time()
        elapsed = now - self._t0
        steps_this_run = step + 1 - self._start_step
        fps = steps_this_run / elapsed if elapsed > 0 else 0

        interval_steps = max(1, step - self._last_log_step)
        interval_elapsed = now - self._last_log_time
        fps_interval = interval_steps / interval_elapsed if interval_elapsed > 0 else 0
        metrics["perf/fps_cumulative"] = fps
        metrics["perf/fps_interval"] = fps_interval
        metrics["perf/ms_per_step_interval"] = (
            1000.0 / fps_interval if fps_interval > 0 else 0
        )
        self._last_log_time = now
        self._last_log_step = step

        for k, v in metrics.items():
            self._writer.writerow([step, k, v])
        self._file.flush()

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        print(
            f"[step {step:>8d}/{self.tcfg.total_steps}] "
            f"total={metrics.get('total_loss', 0):.3f} "
            f"dyn={metrics.get('loss/dyn', 0):.3f} "
            f"rew={metrics.get('loss/rew', 0):.3f} "
            f"policy={metrics.get('loss/policy', 0):.3f} "
            f"fps={fps:.0f} "
            f"fps_interval={fps_interval:.1f} "
            f"ms_step={metrics['perf/ms_per_step_interval']:.1f}"
        )

    def log_reconstructions(
        self, target: np.ndarray, recon: np.ndarray, step: int
    ) -> None:
        """Log up to 4 side-by-side ``input | recon`` panels to W&B (3D-51)."""
        if self._wandb is None:
            return
        n = min(4, target.shape[0])
        images = []
        for i in range(n):
            combo = np.concatenate([target[i], recon[i]], axis=1)  # side by side
            combo = np.clip(combo * 255.0, 0, 255).astype(np.uint8)
            images.append(self._wandb.Image(combo, caption=f"input | recon ({i})"))
        self._wandb.log({"decoder/reconstructions": images}, step=step)

    def log_val_metrics(
        self, metrics: dict[str, Any], step: int, elapsed: float
    ) -> None:
        """Prefix, persist, and print one val loop's final metrics snapshot."""
        val_logged = {
            f"val/{k}" if not k.startswith("val/") else k: v for k, v in metrics.items()
        }
        for k, v in val_logged.items():
            self._writer.writerow([step, k, v])
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(val_logged, step=step)

        sr = val_logged.get("val/metrics/sr", 0.0)
        spl = val_logged.get("val/metrics/spl", 0.0)
        softspl = val_logged.get("val/metrics/softspl", 0.0)
        dtg = val_logged.get("val/metrics/dtg", 0.0)
        sr_str = f"{sr:.3f}" if isinstance(sr, float) else str(sr)
        spl_str = f"{spl:.3f}" if isinstance(spl, float) else str(spl)
        soft_str = f"{softspl:.3f}" if isinstance(softspl, float) else str(softspl)
        dtg_str = f"{dtg:.3f}" if isinstance(dtg, float) else str(dtg)
        print(
            f"[step {step:>8d}] VAL-LOOP "
            f"sr={sr_str} spl={spl_str} softspl={soft_str} dtg={dtg_str}m "
            f"({self.tcfg.val_episodes} eps in {elapsed:.1f}s)"
        )

    def log_adapter_summary(
        self,
        stats: dict[str, float],
        final_step: int,
    ) -> None:
        """Write the adapter's end-of-run diagnostics.

        Rows land in ``metrics.csv`` under the final trainer step; the same
        stats also go to the W&B run summary when W&B is active. Adapters
        without a diagnostics hook report nothing and this is a no-op.
        """
        if not stats:
            return
        for k, v in stats.items():
            self._writer.writerow([final_step, k, v])
        self._file.flush()
        if self._wandb is not None:
            self._wandb.summary.update(stats)
        print("=== adapter summary ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    def close_metrics_file(self) -> None:
        """Close metrics.csv (idempotent); called before the final checkpoint."""
        if not self._file.closed:
            self._file.close()

    def finish(self, status: str) -> None:
        """Finalize the run: close CSV, write manifest end, finish W&B."""
        self.close_metrics_file()
        write_manifest_end(Path(self.tcfg.output_dir), status)
        if self._wandb is not None:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def apply_resume(agent: R2DreamerAgentLike, resume_from: str) -> int:
    """Overwrite freshly-initialised agent state from a checkpoint.

    Args:
        agent: Agent whose params/opt/EMA state get replaced in place.
        resume_from: Path to a checkpoint written by ``save_checkpoint``.

    Returns:
        The checkpoint's step, used as the training loop's start step.

    Raises:
        FileNotFoundError: If ``resume_from`` does not exist.
    """
    if not os.path.exists(resume_from):
        raise FileNotFoundError(
            f"resume_from points at non-existent path: {resume_from}"
        )
    state = load_checkpoint(resume_from)
    agent.params = jax.tree.map(jnp.asarray, state["params"])
    agent.opt_state = jax.tree.map(jnp.asarray, state["opt_state"])
    agent.slow_critic_params = jax.tree.map(jnp.asarray, state["slow_critic_params"])
    agent.ema_state = jax.tree.map(jnp.asarray, state["ema_state"])
    resume_step = int(state["step"])
    print(f"Resumed agent state from {resume_from} at step {resume_step}")
    return resume_step


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


def prefill(
    experience: ExperienceSource,
    *,
    num_steps: int,
    num_actions: int,
    rng_key: jnp.ndarray,
) -> jnp.ndarray:
    """Fill the replay buffer with uniformly random actions.

    The collector's reset fires the scene-aware on_episode_reset callback
    (VGGT PERSIST_SCENE saves/restores per scene) even though prefill discards
    reset observations for replay purposes — otherwise reset_for_scene never
    runs during prefill and the first train episode fresh-resets, orphaning
    the prefill frame (see PROTOCOL.md §2 / smoke 5738008). ``summarize=False``
    keeps the episode metrics fn (and its rolling trackers) untouched during
    random collection.

    Args:
        experience: Recording collector to fill.
        num_steps: Number of random env steps.
        num_actions: Discrete action-space size to sample from.
        rng_key: JAX PRNG key; split once per step.

    Returns:
        The advanced PRNG key.
    """
    print(f"Prefilling {num_steps} steps...")
    experience.reset()
    for _ in range(num_steps):
        rng_key, action_key = jax.random.split(rng_key)
        action = int(jax.random.randint(action_key, (), 0, num_actions))
        experience.step(action, summarize=False)
    return rng_key


def _should_record_video(
    tcfg: TrainerConfig,
    logger: RunLoggerLike,
    experience: ExperienceSource,
    step: int,
    next_video_step: int,
) -> bool:
    return (
        logger.wandb_active
        and tcfg.video_log_every > 0
        and tcfg.video_log_episodes > 0
        and step >= next_video_step
        and experience.supports_video
    )


def train_loop(
    agent: R2DreamerAgentLike,
    experience: ExperienceSource,
    acfg: R2DreamerConfig,
    tcfg: TrainerConfig,
    logger: RunLoggerLike,
    rng_key: jnp.ndarray,
    *,
    start_step: int = 0,
    val_experience: ExperienceSource | None = None,
) -> jnp.ndarray:
    """Run the act -> collect -> train-ratio loop from start_step to total_steps.

    Args:
        agent: Agent providing ``act`` and ``train_step``. Both are pure: the
            loop threads the acting carry as a local and writes the train state
            each gradient step back to ``agent.train_state``, which is what
            checkpointing and ``agent.params`` read.
        experience: Recording collector for on-policy rollouts and sampling.
        acfg: Agent config (batch_size, seq_len, train_ratio, decoder flag).
        tcfg: Loop-control config (total_steps, cadences).
        logger: Metric/video sinks.
        rng_key: JAX PRNG key threaded through acting/training/validation.
        start_step: First loop step (non-zero when resuming).
        val_experience: Optional non-recording collector for the val cadence.

    Returns:
        The advanced PRNG key.
    """
    print(f"Training from step {start_step} to {tcfg.total_steps}...")
    agent_step = experience.reset()
    act_state = agent.initial_act_state()
    logger.start_timing(start_step)
    batch_steps = acfg.batch_size * acfg.seq_len
    train_credit = 0.0
    log_pending = False
    video_next_step = start_step
    if _should_record_video(tcfg, logger, experience, start_step, video_next_step):
        experience.start_video_capture()

    for step in range(start_step, tcfg.total_steps):
        rng_key, act_key = jax.random.split(rng_key)
        action_array, act_state = agent.act(
            agent.params,
            agent_step.encoder_obs,
            agent_step.is_first,
            act_state,
            act_key,
        )
        result = experience.step(int(action_array))
        agent_step = result.agent_step

        if result.episode is not None:
            logger.log_episode(result.episode, step)
            if result.episode.video_frames is not None:
                logger.log_video(
                    "train/episode_video", result.episode.video_frames, step
                )
                video_next_step = step + max(1, tcfg.video_log_every)
            if _should_record_video(
                tcfg, logger, experience, step + 1, video_next_step
            ):
                experience.start_video_capture()

        # --- Train ---
        if experience.buffer_size >= batch_steps:
            train_credit += acfg.train_ratio / batch_steps
            if step % tcfg.log_every == 0:
                log_pending = True
            # With fractional credit the update and the log cadence can have
            # opposite parity, so a due log waits for the next real update.
            will_log = log_pending and train_credit >= 1.0
            batch = None
            metrics = None
            while train_credit >= 1.0:
                rng_key, train_key = jax.random.split(rng_key)
                batch = experience.sample(acfg.batch_size, acfg.seq_len)
                agent.train_state, metrics = agent.train_step(
                    agent.train_state, batch, train_key, materialize=will_log
                )
                train_credit -= 1.0

            if will_log and metrics is not None:
                logger.log_train_metrics(materialize_metrics(metrics), step)
                log_pending = False
                if (
                    getattr(acfg, "decoder", False)
                    and batch is not None
                    and logger.wandb_active
                ):
                    pair = agent.reconstruct(batch)
                    if pair is not None:
                        target, recon = jax.device_get(pair)  # (B*T, H, W, C) HWC
                        logger.log_reconstructions(target, recon, step)

        # --- Val-Episode-Loop (3D-36): deterministic held-out rollouts ---
        if (
            val_experience is not None
            and tcfg.val_every > 0
            and (step + 1) % tcfg.val_every == 0
        ):
            rng_key, val_key = jax.random.split(rng_key)
            val_loop(agent, val_experience, tcfg, logger, val_key, step)

        # --- Checkpoint ---
        if (step + 1) % tcfg.checkpoint_every == 0:
            save_checkpoint(agent, step + 1, tcfg.output_dir)

    return rng_key


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _run_single_val_episode(
    agent: R2DreamerAgentLike,
    val_experience: ExperienceSource,
    tcfg: TrainerConfig,
    logger: RunLoggerLike,
    rng_key: jnp.ndarray,
    *,
    record_video: bool,
    step: int,
) -> tuple[dict[str, Any], jnp.ndarray]:
    agent_step = val_experience.reset()
    act_state = agent.initial_act_state()
    if record_video:
        val_experience.start_video_capture()

    for _ in range(tcfg.val_max_episode_steps):
        rng_key, act_key = jax.random.split(rng_key)
        action_array, act_state = agent.act(
            agent.params,
            agent_step.encoder_obs,
            agent_step.is_first,
            act_state,
            act_key,
            training=False,
        )
        result = val_experience.step(int(action_array))
        if result.done:
            break
        agent_step = result.agent_step

    episode = val_experience.finish_episode()
    if episode.video_frames is not None:
        logger.log_video("val/episode_video", episode.video_frames, step)
    return episode.metrics, rng_key


def val_loop(
    agent: R2DreamerAgentLike,
    val_experience: ExperienceSource,
    tcfg: TrainerConfig,
    logger: RunLoggerLike,
    rng_key: jnp.ndarray,
    step: int,
) -> jnp.ndarray:
    """Deterministic Val-Episode-Loop (3D-36) + video recording (3D-41).

    Runs ``val_episodes`` greedy rollouts in the pinned eval env and logs
    rolling val/* metrics. The first val_video_episodes are captured as W&B
    videos (deterministic playback — same scene across runs because the eval
    episode order is pinned by the curriculum JSON). Each val episode threads
    its own acting carry, so the training rollout's carry is untouched.

    Args:
        agent: Agent to evaluate (acting with ``training=False``).
        val_experience: Non-recording, non-auto-reset collector on the eval env.
        tcfg: Loop-control config (val_episodes, val_max_episode_steps, ...).
        logger: Metric/video sinks.
        rng_key: JAX PRNG key.
        step: Train step this val loop is logged under.

    Returns:
        The advanced PRNG key.
    """
    last_val_metrics: dict[str, Any] = {}
    videos_recorded = 0
    val_t0 = time.time()

    for _ep_idx in range(tcfg.val_episodes):
        record_video = videos_recorded < tcfg.val_video_episodes and logger.wandb_active
        last_val_metrics, rng_key = _run_single_val_episode(
            agent,
            val_experience,
            tcfg,
            logger,
            rng_key,
            record_video=record_video,
            step=step,
        )
        if record_video:
            videos_recorded += 1

    # The final episode's tracker snapshot is logged; the rolling-mean fields
    # already reflect the whole val loop (the tracker is shared across
    # episodes within this run).
    logger.log_val_metrics(last_val_metrics, step, time.time() - val_t0)
    return rng_key


# ---------------------------------------------------------------------------
# Overfit-one-batch diagnostic loop (Karpathy step 3)
# ---------------------------------------------------------------------------


def overfit_loop(
    agent: R2DreamerAgentLike,
    experience: ExperienceSource,
    tcfg: TrainerConfig,
    logger: RunLoggerLike,
    rng_key: jnp.ndarray,
) -> jnp.ndarray:
    """Freeze one sampled batch and call train_step on it repeatedly.

    Proves the full stack (encoder -> RSSM -> heads) can memorise a real
    trajectory. If loss does not drop monotonically, the gradient path is
    broken — no amount of production wall-clock will save the run.
    Disables env rollouts, validation, and checkpointing.

    Args:
        agent: Agent under diagnosis.
        experience: Recording collector holding at least one prefilled batch.
        tcfg: Overfit knobs (overfit_steps, overfit_batch_size, ...).
        logger: Metric sinks.
        rng_key: JAX PRNG key.

    Returns:
        The advanced PRNG key.

    Raises:
        RuntimeError: If the buffer is too small or the loss drop verification
            fails.
        ValueError: If ``overfit_steps`` is below one.
    """
    buffer_size = experience.buffer_size
    if buffer_size < tcfg.overfit_batch_size * tcfg.overfit_seq_len:
        raise RuntimeError(
            f"overfit_one_batch: buffer too small "
            f"({buffer_size} < {tcfg.overfit_batch_size}*{tcfg.overfit_seq_len}). "
            f"Increase --prefill."
        )

    # Sample once, freeze, reuse.
    batch = experience.sample(tcfg.overfit_batch_size, tcfg.overfit_seq_len)
    print(
        f"Overfit mode: cached batch "
        f"B={tcfg.overfit_batch_size} T={tcfg.overfit_seq_len}; "
        f"running {tcfg.overfit_steps} train_step iterations."
    )

    if tcfg.overfit_steps < 1:
        raise ValueError(f"overfit_steps must be >= 1, got {tcfg.overfit_steps}")

    logger.start_timing(0)
    first_loss = last_loss = 0.0
    for step in range(tcfg.overfit_steps):
        rng_key, train_key = jax.random.split(rng_key)
        agent.train_state, device_metrics = agent.train_step(
            agent.train_state, batch, train_key
        )
        metrics = materialize_metrics(device_metrics)
        last_loss = metrics["total_loss"]
        if step == 0:
            first_loss = last_loss

        if step % tcfg.log_every == 0 or step == tcfg.overfit_steps - 1:
            logger.log_train_metrics(metrics, step)

    loss_drop = (first_loss - last_loss) / max(abs(first_loss), 1e-12)
    logger.write_row(tcfg.overfit_steps - 1, "verify/overfit_loss_drop", loss_drop)
    logger.write_row(
        tcfg.overfit_steps - 1,
        "verify/overfit_pass",
        float(loss_drop >= tcfg.overfit_min_loss_drop),
    )
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


# ---------------------------------------------------------------------------
# run_training
# ---------------------------------------------------------------------------


def run_training(
    agent: R2DreamerAgentLike,
    experience: ExperienceSource,
    acfg: R2DreamerConfig,
    tcfg: TrainerConfig,
    *,
    val_experience: ExperienceSource | None = None,
    resume_step: int = 0,
) -> None:
    """Execute a full training run: prefill + train loop + final checkpoint.

    Owns the run scaffold: RunLogger lifecycle (manifest/CSV/W&B), status
    bookkeeping, ``hard_exit_on_finish``, and collector shutdown. Call
    :func:`apply_resume` on the agent first when resuming, and pass the
    returned step as ``resume_step``.

    Args:
        agent: Agent to train.
        experience: Recording collector (train env + adapter + buffer).
        acfg: Agent config.
        tcfg: Loop-control config.
        val_experience: Optional non-recording collector for the val cadence.
        resume_step: Step to resume from; skips prefill when positive.
    """
    logger = RunLogger(acfg, tcfg, resume=resume_step > 0)
    status = "failed"
    rng_key = jax.random.PRNGKey(tcfg.seed)

    try:
        if resume_step > 0:
            # Skip random prefill — the trained policy collects on-policy
            # transitions in train_loop until buffer >= batch_steps.
            # env.reset() / extractor.reset() fire at train_loop entry.
            print(f"Resume mode: skipping prefill, jumping to step {resume_step}")
        else:
            rng_key = prefill(
                experience,
                num_steps=tcfg.prefill_steps,
                num_actions=acfg.num_actions,
                rng_key=rng_key,
            )
        if tcfg.overfit_one_batch:
            rng_key = overfit_loop(agent, experience, tcfg, logger, rng_key)
        else:
            rng_key = train_loop(
                agent,
                experience,
                acfg,
                tcfg,
                logger,
                rng_key,
                start_step=resume_step,
                val_experience=val_experience,
            )
        logger.log_adapter_summary(experience.diagnostics(), tcfg.total_steps)
        logger.close_metrics_file()

        save_checkpoint(agent, tcfg.total_steps, tcfg.output_dir)
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        logger.finish(status)
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
        experience.close()
        if val_experience is not None:
            val_experience.close()
