"""Training orchestration for R2-Dreamer experiments.

Provides Trainer (training loop), convert_batch (buffer→agent format),
save/load_checkpoint, ObsAdapter (env→buffer/agent bridge), and
habitat_defaults (pre-configured Habitat+CNN settings). Loop/orchestration
knobs live in ``TrainerConfig`` (``src.configs.config``).
"""

from __future__ import annotations

import csv
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Protocol, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.replay_buffer import (
    HybridObservation,
    ReplayBatch,
    ReplayBuffer,
    ReplayTransition,
)
from src.environments.observation import ObservationFrame
from src.shared.video_utils import (
    compose_frame,
    log_episode_video,
    render_topdown_frame,
)
from src.r2dreamer.adapters import ObsAdapter  # noqa: F401 — re-exported for callers
from src.r2dreamer.checkpointing import (
    config_snapshot,
    load_checkpoint,
    save_checkpoint,
)
from src.configs.config import R2DreamerConfig, TrainerConfig
from src.r2dreamer.manifest import write_manifest_end, write_manifest_start
from src.r2dreamer.obs_batch import ObservationPacker


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
        self, batch: dict[str, jnp.ndarray], rng_key: jnp.ndarray
    ) -> dict[str, float]: ...

    def act(
        self,
        encoder_obs: Any,
        is_first: bool,
        rng_key: jnp.ndarray,
        training: bool = True,
    ) -> int: ...

    def reconstruct(
        self, batch: dict[str, jnp.ndarray]
    ) -> tuple[np.ndarray, np.ndarray] | None: ...


# ---------------------------------------------------------------------------
# convert_batch
# ---------------------------------------------------------------------------

ArrayObservation: TypeAlias = np.ndarray | jax.Array
ReplayObservation: TypeAlias = (
    ArrayObservation | Mapping[str, ArrayObservation] | HybridObservation
)
ReplayArrayBatch: TypeAlias = dict[str, Any]


def _stack_array_grid(values: list[list[ArrayObservation]]) -> jax.Array:
    """Stack ``(B, T)`` replay observation arrays into one JAX array."""
    return jnp.stack(
        [jnp.stack([jnp.asarray(value) for value in sequence]) for sequence in values]
    )


def _transition_observation(transition: ReplayTransition) -> ReplayObservation:
    """Return a transition observation, including structured adapter mappings."""
    return cast(ReplayObservation, transition.obs)


def _stack_hybrid_observations(
    obs_grid: list[list[ReplayObservation]],
) -> dict[str, jax.Array]:
    """Stack replay-domain hybrid observations as explicit replay fields."""
    images: list[list[ArrayObservation]] = []
    wp_cp_values: list[list[ArrayObservation]] = []
    for sequence in obs_grid:
        image_sequence: list[ArrayObservation] = []
        wp_cp_sequence: list[ArrayObservation] = []
        for obs in sequence:
            if not isinstance(obs, HybridObservation):
                raise TypeError("cannot mix hybrid and non-hybrid replay observations")
            image_sequence.append(obs.image)
            wp_cp_sequence.append(obs.wp_cp)
        images.append(image_sequence)
        wp_cp_values.append(wp_cp_sequence)
    return {
        "image": _stack_array_grid(images),
        "wp_cp": _stack_array_grid(wp_cp_values),
    }


def _stack_mapping_observations(
    obs_grid: list[list[ReplayObservation]],
    first_obs: Mapping[str, ArrayObservation],
) -> dict[str, jax.Array]:
    """Stack structured replay mappings after checking each window has same keys."""
    keys = tuple(first_obs.keys())
    expected_keys = set(keys)
    for sequence in obs_grid:
        for obs in sequence:
            if not isinstance(obs, Mapping):
                raise TypeError("cannot mix mapping and non-mapping replay observations")
            if set(obs.keys()) != expected_keys:
                raise KeyError(
                    "replay observation keys changed inside sampled batch: "
                    f"expected={sorted(expected_keys)}, got={sorted(obs.keys())}"
                )
    return {
        key: _stack_array_grid(
            [
                [cast(Mapping[str, ArrayObservation], obs)[key] for obs in sequence]
                for sequence in obs_grid
            ]
        )
        for key in keys
    }


def _stack_replay_observations(batch: ReplayBatch) -> jax.Array | dict[str, jax.Array]:
    """Stack transition observations into the raw replay-batch observation form."""
    obs_grid = [
        [_transition_observation(transition) for transition in sequence]
        for sequence in batch
    ]
    first_obs = obs_grid[0][0]
    if isinstance(first_obs, HybridObservation):
        return _stack_hybrid_observations(obs_grid)
    if isinstance(first_obs, Mapping):
        return _stack_mapping_observations(obs_grid, first_obs)
    return _stack_array_grid(cast(list[list[ArrayObservation]], obs_grid))


def replay_batch_to_arrays(batch: ReplayBatch) -> ReplayArrayBatch:
    """Pack transition-object replay windows into arrays with ``(B, T)`` prefix.

    Args:
        batch: Non-empty sampled replay windows returned by ``ReplayBuffer.sample``.

    Returns:
        Raw replay arrays keyed by ``obs``, ``actions``, ``rewards``,
        ``is_first``, and ``is_episode_end``. Observation leaves preserve their
        stored dtype and have shape ``(batch_size, seq_len, *obs_shape)``.
    """
    if not batch:
        raise ValueError("cannot convert an empty replay batch")
    seq_len = len(batch[0])
    if seq_len == 0:
        raise ValueError("cannot convert replay sequences with length zero")
    if any(len(sequence) != seq_len for sequence in batch):
        raise ValueError("all replay sequences must have the same length")

    episode_ends = [
        [bool(transition.is_episode_end) for transition in sequence]
        for sequence in batch
    ]
    is_first = [
        [
            offset == 0
            or bool(transition.is_first)
            or (offset > 0 and episode_ends[batch_index][offset - 1])
            for offset, transition in enumerate(sequence)
        ]
        for batch_index, sequence in enumerate(batch)
    ]

    return {
        "obs": _stack_replay_observations(batch),
        "actions": jnp.asarray(
            [[int(transition.action) for transition in sequence] for sequence in batch],
            dtype=jnp.int32,
        ),
        "rewards": jnp.asarray(
            [
                [float(transition.reward) for transition in sequence]
                for sequence in batch
            ],
            dtype=jnp.float32,
        ),
        "is_first": jnp.asarray(is_first, dtype=jnp.bool_),
        "is_episode_end": jnp.asarray(episode_ends, dtype=jnp.bool_),
    }


def convert_batch(batch: ReplayBatch | ReplayArrayBatch, num_actions: int) -> dict[str, Any]:
    """Convert replay output to agent training format.

    Replay stores transitions as ``obs_t, action_t, reward_{t+1}``. The RSSM
    posterior for ``obs_t`` must receive the previous action/labels, so shift
    transition fields right by one step inside each sampled window.
    """
    replay_arrays = replay_batch_to_arrays(batch) if isinstance(batch, list) else batch
    actions = jax.nn.one_hot(cast(jax.Array, replay_arrays["actions"]), num_actions)
    rewards = jnp.asarray(replay_arrays["rewards"], dtype=jnp.float32)
    episode_end = jnp.asarray(replay_arrays["is_episode_end"], dtype=jnp.float32)
    zero_action = jnp.zeros_like(actions[:, :1])
    zero_scalar = jnp.zeros_like(rewards[:, :1])
    return {
        "obs": replay_arrays["obs"],
        "actions": jnp.concatenate([zero_action, actions[:, :-1]], axis=1),
        "rewards": jnp.concatenate([zero_scalar, rewards[:, :-1]], axis=1),
        "is_first": jnp.asarray(replay_arrays["is_first"], dtype=jnp.float32),
        "is_episode_end": jnp.concatenate(
            [zero_scalar, episode_end[:, :-1]], axis=1
        ),
    }


# ---------------------------------------------------------------------------
# habitat_defaults
# ---------------------------------------------------------------------------

EpisodeMetricsFn = Callable[..., dict[str, Any]]


def habitat_defaults(env: Any, *, track_collision_rate: bool = False) -> dict[str, Any]:
    """Pre-configured ObsAdapter and episode_metrics_fn for Habitat+CNN.

    Returns dict with keys "obs_adapter" and "episode_metrics_fn".

    Pass track_collision_rate=True for the val-loop tracker; train rollouts
    leave it False so the dashboard isn't doubly-noisy.
    """
    from src.shared.wandb_utils import EpisodeTracker

    tracker = EpisodeTracker(window=100, track_collision_rate=track_collision_rate)
    action_names = {0: "stop", 1: "forward", 2: "left", 3: "right"}

    def episode_metrics_fn(
        env: Any,
        last_obs: ObservationFrame,
        episode_reward: float,
        episode_steps: int,
        action_counts: np.ndarray,
    ) -> dict[str, Any]:
        success = last_obs.success
        spl = last_obs.spl
        softspl = last_obs.softspl
        dtg = last_obs.dtg
        collision_rate = last_obs.collision_rate
        episode = env.current_episode
        category = getattr(episode, "object_category", "unknown")
        scene_raw = getattr(episode, "scene_id", "")
        path_length = env._path_length
        shortest_path = env._start_geodesic
        path_ratio = path_length / shortest_path if shortest_path > 0 else 0.0

        tracked = tracker.record(
            reward=episode_reward,
            success=success,
            spl=spl,
            category=category,
            scene_id=scene_raw,
            softspl=softspl,
            dtg=dtg,
            collision_rate=collision_rate,
        )

        action_pcts = action_counts / max(episode_steps, 1)
        return {
            **tracked,
            "episode/steps": episode_steps,
            "episode/path_length": path_length,
            "episode/shortest_path": shortest_path,
            "episode/path_ratio": path_ratio,
            "episode_reset": 1,
            **{
                f"action/{action_names[i]}_pct": float(action_pcts[i])
                for i in range(len(action_counts))
            },
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
        agent: R2DreamerAgentLike,
        env: Env,
        agent_config: R2DreamerConfig,
        trainer_config: TrainerConfig,
        obs_adapter: ObsAdapter | None = None,
        episode_metrics_fn: EpisodeMetricsFn | None = None,
        val_env: Env | None = None,
        val_obs_adapter: ObsAdapter | None = None,
        val_episode_metrics_fn: EpisodeMetricsFn | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.acfg = agent_config
        self.tcfg = trainer_config
        self.obs_adapter = obs_adapter or ObsAdapter()
        self.obs_packer = ObservationPacker(agent_config)
        self.episode_metrics_fn = episode_metrics_fn
        # Val-Episode-Loop wiring (3D-36). All three must be non-None for the
        # loop to run; the launcher constructs them together when val is on.
        self.val_env = val_env
        self.val_obs_adapter = val_obs_adapter
        self.val_episode_metrics_fn = val_episode_metrics_fn

        # Replay shape and dtype are inferred lazily from the first prepared
        # observation; normalization belongs to Observation Preparation.
        self.buffer = ReplayBuffer(capacity=agent_config.buffer_capacity)

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
                    print(
                        f"Resume mode: skipping prefill, jumping to step {self._resume_step}"
                    )
                else:
                    rng_key = self._prefill(rng_key, writer, f)
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
            if self.val_env is not None:
                self.val_env.close()

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prepare_observation(
        self, adapter: ObsAdapter, obs: ObservationFrame
    ) -> tuple[Any, Any, bool]:
        prepared = adapter.prepare_env_step(obs, self.obs_packer)
        return prepared.replay_obs, prepared.encoder_obs, prepared.is_first

    def _prefill(self, rng_key: jnp.ndarray, writer: Any, f: Any) -> jnp.ndarray:
        acfg, tcfg = self.acfg, self.tcfg
        print(f"Prefilling {tcfg.prefill_steps} steps...")

        obs = self.env.reset()
        if self.obs_adapter.on_episode_reset:
            self.obs_adapter.on_episode_reset()
        buffer_obs, _, _ = self._prepare_observation(self.obs_adapter, obs)

        for _ in range(tcfg.prefill_steps):
            rng_key, action_key = jax.random.split(rng_key)
            action = int(jax.random.randint(action_key, (), 0, acfg.num_actions))
            next_obs = self.env.step(action)
            next_buffer_obs, _, _ = self._prepare_observation(
                self.obs_adapter, next_obs
            )

            self._record_train_transition(
                buffer_obs=buffer_obs,
                action=action,
                next_obs=next_obs,
            )

            if next_obs.done:
                obs = self.env.reset()
                if self.obs_adapter.on_episode_reset:
                    self.obs_adapter.on_episode_reset()
                buffer_obs, _, _ = self._prepare_observation(self.obs_adapter, obs)
            else:
                obs = next_obs
                buffer_obs = next_buffer_obs
        return rng_key

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def _reset_train_episode(
        self,
    ) -> tuple[ObservationFrame, np.ndarray | dict[str, np.ndarray], Any, bool]:
        obs = self.env.reset()
        if self.obs_adapter.on_episode_reset:
            self.obs_adapter.on_episode_reset()
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
        writer: Any,
        f: Any,
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
            writer,
            f,
        )
        if video_recording is not None:
            log_episode_video(
                self._wandb,
                "train/episode_video",
                video_recording["frames"],
                step,
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
            video_recording = self._start_video_recording(self.env, obs)
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

    def _train_loop(self, rng_key: jnp.ndarray, writer: Any, f: Any) -> jnp.ndarray:
        acfg, tcfg = self.acfg, self.tcfg

        start_step = self._resume_step
        print(f"Training from step {start_step} to {tcfg.total_steps}...")
        obs, buffer_obs, encoder_obs, is_first = self._reset_train_episode()
        episode_reward, episode_steps, action_counts = (
            0.0,
            0,
            np.zeros(acfg.num_actions, dtype=int),
        )
        self._t0 = time.time()
        self._last_log_time = self._t0
        self._last_log_step = start_step - 1
        batch_steps = acfg.batch_size * acfg.seq_len
        train_credit = 0.0
        metrics: dict[str, Any] = {}
        video_next_step = start_step
        video_recording = None
        if self._should_record_video(start_step, video_next_step):
            video_recording = self._start_video_recording(self.env, obs)

        for step in range(start_step, tcfg.total_steps):
            rng_key, act_key = jax.random.split(rng_key)
            action = self.agent.act(encoder_obs, is_first, act_key)
            next_obs = self.env.step(action)
            next_buffer_obs, next_encoder_obs, next_is_first = self._prepare_observation(
                self.obs_adapter,
                next_obs,
            )

            self._record_train_transition(
                buffer_obs=buffer_obs,
                action=action,
                next_obs=next_obs,
            )
            action_counts[action] += 1
            episode_reward += next_obs.reward
            episode_steps += 1
            if video_recording is not None:
                self._append_video_frame(self.env, video_recording, next_obs)

            if next_obs.done:
                (
                    obs,
                    buffer_obs,
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
                    writer=writer,
                    f=f,
                    video_recording=video_recording,
                    video_next_step=video_next_step,
                )
            else:
                obs = next_obs
                buffer_obs = next_buffer_obs
                encoder_obs = next_encoder_obs
                is_first = next_is_first

            # --- Train ---
            if self.buffer.size >= batch_steps:
                train_credit += acfg.train_ratio / batch_steps
                while train_credit >= 1.0:
                    rng_key, train_key = jax.random.split(rng_key)
                    batch_raw = replay_batch_to_arrays(
                        self.buffer.sample(acfg.batch_size, acfg.seq_len)
                    )
                    batch_raw = self.obs_adapter.augment_replay_batch(batch_raw)
                    batch = convert_batch(batch_raw, acfg.num_actions)
                    metrics = self.agent.train_step(batch, train_key)
                    train_credit -= 1.0

                if step % tcfg.log_every == 0 and metrics:
                    self._log_train_metrics(metrics, step, writer, f)
                    if getattr(acfg, "decoder", False):
                        self._maybe_log_recon(batch, step)

            # --- Val-Episode-Loop (3D-36): deterministic held-out rollouts ---
            if (
                self.val_env is not None
                and tcfg.val_every > 0
                and (step + 1) % tcfg.val_every == 0
            ):
                rng_key, val_key = jax.random.split(rng_key)
                self._run_val_loop(val_key, step, writer, f)

            # --- Checkpoint ---
            if (step + 1) % tcfg.checkpoint_every == 0:
                save_checkpoint(self.agent, step + 1, tcfg.output_dir)

        return rng_key

    def _should_record_video(self, step: int, next_video_step: int) -> bool:
        return (
            self._wandb is not None
            and self.tcfg.video_log_every > 0
            and self.tcfg.video_log_episodes > 0
            and step >= next_video_step
            and hasattr(self.env, "_env")
        )

    def _goal_positions(self, env: Env) -> list[list[float]]:
        positions = []
        for goal in env.current_episode.goals:
            if goal.view_points:
                pos = goal.view_points[0].agent_state.position
            else:
                pos = goal.position
            positions.append(pos.tolist() if hasattr(pos, "tolist") else list(pos))
        return positions

    def _agent_position(self, env: Env) -> list[float]:
        pos = env._env.sim.get_agent_state().position
        return pos.tolist() if hasattr(pos, "tolist") else list(pos)

    def _start_video_recording(self, env: Env, obs: ObservationFrame) -> dict[str, Any]:
        recording = {
            "trajectory": [self._agent_position(env)],
            "goals": self._goal_positions(env),
            "frames": [],
        }
        self._append_video_frame(env, recording, obs)
        return recording

    def _append_video_frame(
        self, env: Env, recording: dict[str, Any], obs: ObservationFrame
    ) -> None:
        if recording["frames"]:
            recording["trajectory"].append(self._agent_position(env))
        topdown = render_topdown_frame(env, recording["trajectory"], recording["goals"])
        recording["frames"].append(compose_frame(obs.image, topdown))

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
        batch_raw = replay_batch_to_arrays(
            self.buffer.sample(tcfg.overfit_batch_size, tcfg.overfit_seq_len)
        )
        batch_raw = self.obs_adapter.augment_replay_batch(batch_raw)
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
        writer.writerow(
            [
                tcfg.overfit_steps - 1,
                "verify/overfit_pass",
                float(loss_drop >= tcfg.overfit_min_loss_drop),
            ]
        )
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
        self,
        last_obs: ObservationFrame,
        episode_reward: float,
        episode_steps: int,
        action_counts: np.ndarray,
        step: int,
        writer: Any,
        f: Any,
    ) -> dict[str, Any]:
        if self.episode_metrics_fn is not None:
            ep_metrics = self.episode_metrics_fn(
                self.env,
                last_obs,
                episode_reward,
                episode_steps,
                action_counts,
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
        self,
        metrics: dict,
        step: int,
        writer: Any,
        f: Any,
    ) -> None:
        now = time.time()
        elapsed = now - self._t0
        steps_this_run = step + 1 - self._resume_step
        fps = steps_this_run / elapsed if elapsed > 0 else 0

        interval_steps = max(1, step - getattr(self, "_last_log_step", step - 1))
        interval_elapsed = now - getattr(self, "_last_log_time", now)
        fps_interval = interval_steps / interval_elapsed if interval_elapsed > 0 else 0
        metrics["perf/fps_cumulative"] = fps
        metrics["perf/fps_interval"] = fps_interval
        metrics["perf/ms_per_step_interval"] = (
            1000.0 / fps_interval if fps_interval > 0 else 0
        )
        self._last_log_time = now
        self._last_log_step = step

        for k, v in metrics.items():
            writer.writerow([step, k, v])
        f.flush()

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
        target, recon = jax.device_get(pair)  # (B*T, 3, 64, 64) in [0, 1]
        n = min(4, target.shape[0])
        images = []
        for i in range(n):
            tgt = np.transpose(target[i], (1, 2, 0))  # CHW -> HWC
            rec = np.transpose(recon[i], (1, 2, 0))
            combo = np.concatenate([tgt, rec], axis=1)  # side by side
            combo = np.clip(combo * 255.0, 0, 255).astype(np.uint8)
            images.append(self._wandb.Image(combo, caption=f"input | recon ({i})"))
        self._wandb.log({"decoder/reconstructions": images}, step=step)

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
            val_adapter.on_episode_reset()
        _, encoder_obs, is_first = self._prepare_observation(val_adapter, obs)

        episode_reward = 0.0
        episode_steps = 0
        action_counts = np.zeros(self.acfg.num_actions, dtype=int)

        recording = None
        if record_video:
            recording = self._start_video_recording(val_env, obs)

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
                self._append_video_frame(val_env, recording, next_obs)

            if next_obs.done:
                obs = next_obs
                break
            obs = next_obs
            encoder_obs = next_encoder_obs
            is_first = next_is_first

        val_metrics = val_episode_metrics_fn(
            val_env,
            obs,
            episode_reward,
            episode_steps,
            action_counts,
        )
        if recording is not None:
            log_episode_video(
                self._wandb,
                "val/episode_video",
                recording["frames"],
                step,
            )
        return val_metrics, rng_key

    def _prefix_val_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            f"val/{k}" if not k.startswith("val/") else k: v for k, v in metrics.items()
        }

    def _log_val_metrics(
        self,
        val_logged: dict[str, Any],
        step: int,
        writer: Any,
        f: Any,
    ) -> None:
        for k, v in val_logged.items():
            writer.writerow([step, k, v])
        f.flush()
        if self._wandb is not None:
            self._wandb.log(val_logged, step=step)

    def _print_val_summary(
        self,
        val_logged: dict[str, Any],
        step: int,
        elapsed: float,
    ) -> None:
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

    def _run_val_loop(
        self,
        rng_key: jnp.ndarray,
        step: int,
        writer: Any,
        f: Any,
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
        self._log_val_metrics(val_logged, step, writer, f)
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
