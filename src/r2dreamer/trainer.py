"""Training orchestration for R2-Dreamer experiments.

Provides Trainer (training loop), replay_batch_to_arrays (transition-window
packing), save/load_checkpoint, and ObsAdapter (env→buffer/agent bridge).
Loop/orchestration knobs live in ``TrainerConfig`` (``src.configs.config``).
Habitat-specific episode metrics live in ``src.environments.habitat``.

Reporting concerns (CSV/W&B metrics logging, video/topdown recording) are
delegated to the ``MetricsLogger``/``EpisodeRecorder`` collaborators in
``src.r2dreamer.reporting``; replay-batch packing lives in
``src.r2dreamer.replay_packing``. Both are re-exported here for backward
compatibility.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.replay_buffer import ReplayBatch, ReplayBuffer, ReplayTransition
from src.configs.config import R2DreamerConfig, TrainerConfig
from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters import ObsAdapter
from src.r2dreamer.checkpointing import (
    config_snapshot,
    load_checkpoint,
    save_checkpoint,
)
from src.r2dreamer.manifest import write_manifest_end, write_manifest_start
from src.r2dreamer.replay_packing import (
    ArrayObservation,
    ReplayArrayBatch,
    ReplayObservation,
    replay_batch_to_arrays,
)
from src.r2dreamer.reporting import EpisodeRecorder, MetricsLogger
from src.shared.dtypes import compute_jnp_dtype

__all__ = [
    "Env",
    "R2DreamerAgentLike",
    "Trainer",
    "EpisodeMetricsFn",
    "replay_batch_to_arrays",
    "ArrayObservation",
    "ReplayObservation",
    "ReplayArrayBatch",
    "config_snapshot",
    "load_checkpoint",
    "save_checkpoint",
]

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class Env(Protocol):
    _env: Any
    current_episode: Any

    def reset(self) -> ObservationFrame: ...
    def step(self, action: int) -> ObservationFrame: ...
    def close(self) -> None: ...


class R2DreamerAgentLike(Protocol):
    """Interface the trainer needs from an R2Dreamer-style agent.

    Using a protocol is stricter than ``Any`` while avoiding a hard dependency
    on the concrete ``R2DreamerAgent`` class. Tests and future agent variants can
    still be passed to ``Trainer`` if they expose the same public contract.
    """

    cfg: Any
    params: Any
    opt_state: Any
    slow_critic_params: Any
    ema_state: Any

    def train_step(
        self, batch: ReplayBatch, rng_key: jnp.ndarray
    ) -> dict[str, float]: ...

    def act(
        self,
        encoder_obs: Any,
        is_first: bool,
        rng_key: jnp.ndarray,
        training: bool = True,
    ) -> int: ...

    def reconstruct(self, batch: Any) -> tuple[np.ndarray, np.ndarray] | None: ...


# ---------------------------------------------------------------------------
# Episode metrics callback
# ---------------------------------------------------------------------------

# Called at episode end with the final observation and episode aggregates. The
# env (if needed) is bound by the callable itself (e.g. HabitatEpisodeMetrics in
# src.environments.habitat), so it is not passed here.
EpisodeMetricsFn = Callable[
    [ObservationFrame, float, int, np.ndarray], dict[str, Any]
]


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
            Signature: (last_obs, episode_reward, episode_steps, action_counts) -> dict
            Any env access is bound by the callable itself (e.g.
            HabitatEpisodeMetrics).
        val_env: Optional pinned environment for the Val-Episode-Loop.
        val_obs_adapter: Optional ObsAdapter for the validation env.
        val_episode_metrics_fn: Optional episode-metrics callback for validation.
        metrics_logger: Optional ``MetricsLogger`` collaborator. Defaults to a
            fresh instance wired to the same W&B run as the trainer, so
            behavior is unchanged unless a test/caller injects its own.
        episode_recorder: Optional ``EpisodeRecorder`` collaborator. Defaults
            analogously to ``metrics_logger``.
    """

    def __init__(
        self,
        agent: R2DreamerAgentLike,
        env: Env,
        agent_config: R2DreamerConfig,
        trainer_config: TrainerConfig,
        obs_adapter: ObsAdapter | None = None,
        episode_metrics_fn: EpisodeMetricsFn | None = None,
        val_env: Env | None = None,
        val_obs_adapter: ObsAdapter | None = None,
        val_episode_metrics_fn: EpisodeMetricsFn | None = None,
        metrics_logger: MetricsLogger | None = None,
        episode_recorder: EpisodeRecorder | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.acfg = agent_config
        self.tcfg = trainer_config
        self.obs_adapter = obs_adapter or ObsAdapter()
        self.episode_metrics_fn = episode_metrics_fn
        # Val-Episode-Loop wiring (3D-36). All three must be non-None for the
        # loop to run; the launcher constructs them together when val is on.
        self.val_env = val_env
        self.val_obs_adapter = val_obs_adapter
        self.val_episode_metrics_fn = val_episode_metrics_fn

        # Replay shape and dtype are inferred lazily from the first prepared
        # observation; normalization belongs to Observation Preparation.
        self.buffer = ReplayBuffer(
            capacity=agent_config.buffer_capacity,
            num_actions=agent_config.num_actions,
            float_dtype=compute_jnp_dtype(agent_config.compute_dtype),
        )

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
                config=config_snapshot(agent_config),
                tags=trainer_config.wandb_tags,
            )
            if trainer_config.wandb_id is not None:
                # resume="must" fails loudly if the run-id does not exist,
                # which is what we want — silent re-creation orphans runs.
                init_kwargs.update(id=trainer_config.wandb_id, resume="must")
            wandb.init(**init_kwargs)

        # Reporting collaborators (CSV/W&B metrics logging, video capture).
        # Both default to fresh instances wired to this trainer's W&B handle
        # (None if W&B is disabled), reproducing prior inline behavior.
        self.metrics_logger = metrics_logger or MetricsLogger(self._wandb)
        self.episode_recorder = episode_recorder or EpisodeRecorder(self._wandb)

    def run(self) -> None:
        """Execute full training run: prefill + train loop + final checkpoint."""
        tcfg, acfg = self.tcfg, self.acfg

        os.makedirs(tcfg.output_dir, exist_ok=True)
        csv_path = os.path.join(tcfg.output_dir, "metrics.csv")

        # MANIFEST.json — emit on start, finalize in finally with run status.
        write_manifest_start(Path(tcfg.output_dir), config_snapshot(acfg))
        status = "failed"

        rng_key = jax.random.PRNGKey(tcfg.seed)

        try:
            # Append to existing CSV when resuming so the prior rows survive;
            # the header is written only on a fresh run. MetricsLogger owns the
            # open writer/handle for the block, so the loop methods below never
            # thread raw CSV handles.
            is_resume = self._resume_step > 0
            with self.metrics_logger.open_csv(csv_path, is_resume):
                if is_resume:
                    # Skip random prefill — the trained policy collects on-policy
                    # transitions in _train_loop until buffer >= batch_steps.
                    # env.reset() / extractor.reset() fire at _train_loop entry.
                    print(
                        f"Resume mode: skipping prefill, jumping to step {self._resume_step}"
                    )
                else:
                    rng_key = self._prefill(rng_key)
                if tcfg.overfit_one_batch:
                    rng_key = self._overfit_loop(rng_key)
                else:
                    rng_key = self._train_loop(rng_key)
                self._log_adapter_summary()

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
            if self.val_env is not None:
                self.val_env.close()

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prepare_observation(
        self, adapter: ObsAdapter, obs: ObservationFrame
    ) -> tuple[Any, Any, bool]:
        prepared = adapter.prepare_env_step(obs)
        return prepared.replay_obs, prepared.encoder_obs, prepared.is_first

    def _prefill(self, rng_key: jnp.ndarray) -> jnp.ndarray:
        acfg, tcfg = self.acfg, self.tcfg
        print(f"Prefilling {tcfg.prefill_steps} steps...")

        # Capture the reset frame so its scene_id reaches the scene-aware
        # on_episode_reset callback (VGGT PERSIST_SCENE saves/restores per
        # scene). Prefill discards the reset obs for replay purposes, but the
        # extractor reset MUST still fire here — otherwise reset_for_scene
        # never runs during prefill and the first train episode fresh-resets,
        # orphaning the prefill frame (see PROTOCOL.md §2 / smoke 5738008).
        _rst_obs = self.env.reset()
        if self.obs_adapter.on_episode_reset:
            self.obs_adapter.on_episode_reset(
                getattr(_rst_obs, "scene_id", None) or "scene"
            )

        for _ in range(tcfg.prefill_steps):
            rng_key, action_key = jax.random.split(rng_key)
            action = int(jax.random.randint(action_key, (), 0, acfg.num_actions))
            next_obs = self.env.step(action)
            next_buffer_obs, _, _ = self._prepare_observation(
                self.obs_adapter, next_obs
            )

            self._record_train_transition(
                buffer_obs=next_buffer_obs,
                action=action,
                next_obs=next_obs,
            )

            if next_obs.done:
                _rst_obs = self.env.reset()
                if self.obs_adapter.on_episode_reset:
                    self.obs_adapter.on_episode_reset(
                        getattr(_rst_obs, "scene_id", None) or "scene"
                    )
        return rng_key

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def _reset_train_episode(
        self,
    ) -> tuple[ObservationFrame, np.ndarray | dict[str, np.ndarray], Any, bool]:
        obs = self.env.reset()
        if self.obs_adapter.on_episode_reset:
            self.obs_adapter.on_episode_reset(
                getattr(obs, "scene_id", None) or "scene"
            )
        buffer_obs, encoder_obs, is_first = self._prepare_observation(
            self.obs_adapter, obs
        )
        return obs, buffer_obs, encoder_obs, is_first

    def _record_train_transition(
        self,
        *,
        buffer_obs: np.ndarray | dict[str, np.ndarray],
        action: int,
        next_obs: ObservationFrame,
    ) -> None:
        if next_obs.previous_action != action:
            raise ValueError(
                "ObservationFrame.previous_action does not match recorded action: "
                f"expected {action}, got {next_obs.previous_action}"
            )
        self.buffer.add(ReplayTransition.from_frame(buffer_obs, next_obs))

    def _finish_train_episode(
        self,
        *,
        last_obs: ObservationFrame,
        episode_reward: float,
        episode_steps: int,
        action_counts: np.ndarray,
        step: int,
        video_recording: dict[str, Any] | None,
        video_next_step: int,
    ) -> tuple[
        ObservationFrame,
        np.ndarray | dict[str, np.ndarray],
        Any,
        bool,
        float,
        int,
        np.ndarray,
        dict[str, Any] | None,
        int,
    ]:
        self._on_episode_end(
            last_obs,
            episode_reward,
            episode_steps,
            action_counts,
            step,
        )
        if video_recording is not None:
            self.episode_recorder.log_video(
                "train/episode_video", video_recording, step
            )
            video_recording = None
            video_next_step = step + max(1, self.tcfg.video_log_every)

        episode_reward, episode_steps, action_counts = (
            0.0,
            0,
            np.zeros(self.acfg.num_actions, dtype=int),
        )
        obs, buffer_obs, encoder_obs, is_first = self._reset_train_episode()
        if self._should_record_video(step + 1, video_next_step):
            video_recording = self.episode_recorder.start_recording(self.env, obs)
        return (
            obs,
            buffer_obs,
            encoder_obs,
            is_first,
            episode_reward,
            episode_steps,
            action_counts,
            video_recording,
            video_next_step,
        )

    def _train_loop(self, rng_key: jnp.ndarray) -> jnp.ndarray:
        acfg, tcfg = self.acfg, self.tcfg

        start_step = self._resume_step
        print(f"Training from step {start_step} to {tcfg.total_steps}...")
        obs, _buffer_obs, encoder_obs, is_first = self._reset_train_episode()
        episode_reward, episode_steps, action_counts = (
            0.0,
            0,
            np.zeros(acfg.num_actions, dtype=int),
        )
        self.metrics_logger.start_timing(start_step)
        batch_steps = acfg.batch_size * acfg.seq_len
        train_credit = 0.0
        metrics: dict[str, Any] = {}
        video_next_step = start_step
        video_recording = None
        if self._should_record_video(start_step, video_next_step):
            video_recording = self.episode_recorder.start_recording(self.env, obs)

        for step in range(start_step, tcfg.total_steps):
            rng_key, act_key = jax.random.split(rng_key)
            action = self.agent.act(encoder_obs, is_first, act_key)
            next_obs = self.env.step(action)
            next_buffer_obs, next_encoder_obs, next_is_first = (
                self._prepare_observation(
                    self.obs_adapter,
                    next_obs,
                )
            )

            self._record_train_transition(
                buffer_obs=next_buffer_obs,
                action=action,
                next_obs=next_obs,
            )
            action_counts[action] += 1
            episode_reward += next_obs.reward
            episode_steps += 1
            if video_recording is not None:
                self.episode_recorder.append_frame(self.env, video_recording, next_obs)

            if next_obs.done:
                (
                    obs,
                    _buffer_obs,
                    encoder_obs,
                    is_first,
                    episode_reward,
                    episode_steps,
                    action_counts,
                    video_recording,
                    video_next_step,
                ) = self._finish_train_episode(
                    last_obs=next_obs,
                    episode_reward=episode_reward,
                    episode_steps=episode_steps,
                    action_counts=action_counts,
                    step=step,
                    video_recording=video_recording,
                    video_next_step=video_next_step,
                )
            else:
                obs = next_obs
                encoder_obs = next_encoder_obs
                is_first = next_is_first

            # --- Train ---
            if self.buffer.size >= batch_steps:
                train_credit += acfg.train_ratio / batch_steps
                while train_credit >= 1.0:
                    rng_key, train_key = jax.random.split(rng_key)
                    batch = self.buffer.sample(acfg.batch_size, acfg.seq_len)
                    batch = self.obs_adapter.augment_replay_batch(batch)
                    metrics = self.agent.train_step(batch, train_key)
                    train_credit -= 1.0

                if step % tcfg.log_every == 0 and metrics:
                    self._log_train_metrics(metrics, step)
                    if getattr(acfg, "decoder", False):
                        self._maybe_log_recon(batch, step)

            # --- Val-Episode-Loop (3D-36): deterministic held-out rollouts ---
            if (
                self.val_env is not None
                and tcfg.val_every > 0
                and (step + 1) % tcfg.val_every == 0
            ):
                rng_key, val_key = jax.random.split(rng_key)
                self._run_val_loop(val_key, step)

            # --- Checkpoint ---
            if (step + 1) % tcfg.checkpoint_every == 0:
                save_checkpoint(self.agent, step + 1, tcfg.output_dir)

        return rng_key

    def _should_record_video(self, step: int, next_video_step: int) -> bool:
        return self.episode_recorder.should_record_video(
            self.env,
            step,
            next_video_step,
            self.tcfg.video_log_every,
            self.tcfg.video_log_episodes,
        )

    # ------------------------------------------------------------------
    # Overfit-one-batch diagnostic loop (Karpathy step 3)
    # ------------------------------------------------------------------

    def _overfit_loop(self, rng_key: jnp.ndarray) -> jnp.ndarray:
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
        batch = self.buffer.sample(tcfg.overfit_batch_size, tcfg.overfit_seq_len)
        batch = self.obs_adapter.augment_replay_batch(batch)
        print(
            f"Overfit mode: cached batch "
            f"B={tcfg.overfit_batch_size} T={tcfg.overfit_seq_len}; "
            f"running {tcfg.overfit_steps} train_step iterations."
        )

        if tcfg.overfit_steps < 1:
            raise ValueError(f"overfit_steps must be >= 1, got {tcfg.overfit_steps}")

        self.metrics_logger.start_timing(0)
        first_loss = last_loss = 0.0
        for step in range(tcfg.overfit_steps):
            rng_key, train_key = jax.random.split(rng_key)
            metrics = self.agent.train_step(batch, train_key)
            last_loss = metrics["total_loss"]
            if step == 0:
                first_loss = last_loss

            if step % tcfg.log_every == 0 or step == tcfg.overfit_steps - 1:
                self._log_train_metrics(metrics, step)

        loss_drop = (first_loss - last_loss) / max(abs(first_loss), 1e-12)
        self.metrics_logger.write_metric_rows(
            [
                (tcfg.overfit_steps - 1, "verify/overfit_loss_drop", loss_drop),
                (
                    tcfg.overfit_steps - 1,
                    "verify/overfit_pass",
                    float(loss_drop >= tcfg.overfit_min_loss_drop),
                ),
            ]
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

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _on_episode_end(
        self,
        last_obs: ObservationFrame,
        episode_reward: float,
        episode_steps: int,
        action_counts: np.ndarray,
        step: int,
    ) -> dict[str, Any]:
        if self.episode_metrics_fn is not None:
            ep_metrics = self.episode_metrics_fn(
                last_obs,
                episode_reward,
                episode_steps,
                action_counts,
            )
        else:
            ep_metrics = {"episode/reward": episode_reward}

        self.metrics_logger.log_episode_end(
            ep_metrics, episode_reward, episode_steps, step
        )
        return ep_metrics

    def _log_adapter_summary(self) -> None:
        self.metrics_logger.log_adapter_summary(
            self.obs_adapter.diagnostics(),
            self.obs_adapter.growth_history,
            self.tcfg.total_steps,
        )

    def _log_train_metrics(
        self,
        metrics: dict,
        step: int,
    ) -> None:
        self.metrics_logger.log_train_metrics(
            metrics, step, self.tcfg.total_steps, self._resume_step
        )

    def _maybe_log_recon(self, batch: ReplayBatch, step: int) -> None:
        self.metrics_logger.maybe_log_recon(
            self.agent, batch, step, getattr(self.acfg, "decoder", False)
        )

    def _run_single_val_episode(
        self,
        *,
        rng_key: jnp.ndarray,
        record_video: bool,
        step: int,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        val_env = self.val_env
        val_adapter = self.val_obs_adapter
        val_episode_metrics_fn = self.val_episode_metrics_fn
        if val_env is None or val_adapter is None or val_episode_metrics_fn is None:
            raise RuntimeError("validation loop is not configured")

        obs = val_env.reset()
        if val_adapter.on_episode_reset:
            val_adapter.on_episode_reset(
                getattr(obs, "scene_id", None) or "scene"
            )
        _, encoder_obs, is_first = self._prepare_observation(val_adapter, obs)

        episode_reward = 0.0
        episode_steps = 0
        action_counts = np.zeros(self.acfg.num_actions, dtype=int)

        recording = None
        if record_video:
            recording = self.episode_recorder.start_recording(val_env, obs)

        for _ in range(self.tcfg.val_max_episode_steps):
            rng_key, act_key = jax.random.split(rng_key)
            action = self.agent.act(encoder_obs, is_first, act_key, training=False)
            next_obs = val_env.step(action)
            _, next_encoder_obs, next_is_first = self._prepare_observation(
                val_adapter, next_obs
            )

            action_counts[action] += 1
            episode_reward += next_obs.reward
            episode_steps += 1
            if recording is not None:
                self.episode_recorder.append_frame(val_env, recording, next_obs)

            if next_obs.done:
                obs = next_obs
                break
            obs = next_obs
            encoder_obs = next_encoder_obs
            is_first = next_is_first

        val_metrics = val_episode_metrics_fn(
            obs,
            episode_reward,
            episode_steps,
            action_counts,
        )
        if recording is not None:
            self.episode_recorder.log_video("val/episode_video", recording, step)
        return val_metrics, rng_key

    def _prefix_val_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return self.metrics_logger.prefix_val_metrics(metrics)

    def _log_val_metrics(
        self,
        val_logged: dict[str, Any],
        step: int,
    ) -> None:
        self.metrics_logger.log_val_metrics(val_logged, step)

    def _print_val_summary(
        self,
        val_logged: dict[str, Any],
        step: int,
        elapsed: float,
    ) -> None:
        self.metrics_logger.print_val_summary(
            val_logged, step, elapsed, self.tcfg.val_episodes
        )

    def _run_val_loop(
        self,
        rng_key: jnp.ndarray,
        step: int,
    ) -> None:
        """Deterministic Val-Episode-Loop (3D-36) + video recording (3D-41).

        Runs `val_episodes` greedy rollouts in the pinned eval env and logs
        rolling val/* metrics. The first val_video_episodes are captured
        as W&B videos (deterministic playback — same scene across runs
        because the eval episode order is pinned by the curriculum JSON).
        """
        tcfg = self.tcfg
        if (
            self.val_env is None
            or self.val_obs_adapter is None
            or self.val_episode_metrics_fn is None
        ):
            return

        act_state = self._snapshot_agent_act_state()
        last_val_metrics: dict[str, Any] = {}
        videos_recorded = 0
        val_t0 = time.time()

        try:
            for _ep_idx in range(tcfg.val_episodes):
                record_video = (
                    videos_recorded < tcfg.val_video_episodes
                    and self._wandb is not None
                )
                last_val_metrics, rng_key = self._run_single_val_episode(
                    rng_key=rng_key,
                    record_video=record_video,
                    step=step,
                )
                if record_video:
                    videos_recorded += 1
        finally:
            self._restore_agent_act_state(act_state)

        # Prefix the final episode's tracker snapshot with `val/`. The
        # rolling-mean fields already reflect the whole val loop (since the
        # tracker is shared across episodes within this run).
        val_logged = self._prefix_val_metrics(last_val_metrics)
        self._log_val_metrics(val_logged, step)
        elapsed = time.time() - val_t0
        self._print_val_summary(val_logged, step, elapsed)

    def _snapshot_agent_act_state(self) -> Any | None:
        """Return a copy of the stateful acting latent, when the agent has one."""
        snapshot = getattr(self.agent, "snapshot_act_state", None)
        if snapshot is None:
            return None
        return snapshot()

    def _restore_agent_act_state(self, state: Any | None) -> None:
        """Restore stateful acting latent after validation rollouts."""
        if state is None:
            return
        restore = getattr(self.agent, "restore_act_state", None)
        if restore is not None:
            restore(state)
