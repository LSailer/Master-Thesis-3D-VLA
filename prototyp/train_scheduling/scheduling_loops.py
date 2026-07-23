"""Timed act/collect/train scheduling loops for the train_scheduling study.

Three schedules over the same duck-typed collaborators (an agent exposing
``act``/``train_step`` and an ``ExperienceSource``-shaped collector):

- ``run_interleaved`` — the production credit scheme from
  ``src/r2dreamer/launch/loops.py::train_loop`` (train steps inline).
- ``run_episode`` — identical credit arithmetic, but train steps only run at
  episode boundaries (``result.done``) plus one final drain.
- ``run_threaded`` — actor thread (act + env step + buffer add) and learner
  thread (sample + train_step) over a ``LockedReplayBuffer``; the learner
  paces itself against the actor's credit-eligible step counter.

All loops are stripped of logging/val/checkpoint cadences so they time exactly
the act -> step -> sample -> train work, and end with
``jax.block_until_ready`` on the agent params so JAX's async dispatch cannot
hide pending device work from the wall clock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import jax


@dataclass
class LoopStats:
    """Wall-clock outcome of one scheduling-mode run.

    Attributes:
        mode: Schedule name (``interleaved`` / ``episode`` / ``threaded``).
        env_steps: Env steps executed in the timed section.
        train_steps: ``agent.train_step`` calls executed in the timed section.
        wall_time_s: Total wall time of the timed section in seconds.
        env_steps_per_s: ``env_steps / wall_time_s``.
        train_steps_per_s: ``train_steps / wall_time_s``.
        achieved_ratio: Achieved ``train_steps * batch_steps / env_steps``
            (comparable to the configured ``train_ratio``).
        extras: Mode-specific diagnostics (e.g. episode boundary count,
            learner pacing target).
    """

    mode: str
    env_steps: int
    train_steps: int
    wall_time_s: float
    env_steps_per_s: float
    train_steps_per_s: float
    achieved_ratio: float
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns the stats as a plain JSON-serializable dict."""
        return asdict(self)


def _finish_stats(
    mode: str,
    *,
    env_steps: int,
    train_steps: int,
    wall_time_s: float,
    batch_steps: int,
    extras: dict[str, Any],
) -> LoopStats:
    """Builds a ``LoopStats`` from raw counters.

    Args:
        mode: Schedule name.
        env_steps: Env steps executed.
        train_steps: Train steps executed.
        wall_time_s: Elapsed wall time in seconds.
        batch_steps: ``batch_size * seq_len`` (credit denominator).
        extras: Mode-specific diagnostics to attach.

    Returns:
        The populated stats record.
    """
    return LoopStats(
        mode=mode,
        env_steps=env_steps,
        train_steps=train_steps,
        wall_time_s=wall_time_s,
        env_steps_per_s=env_steps / wall_time_s if wall_time_s > 0 else 0.0,
        train_steps_per_s=train_steps / wall_time_s if wall_time_s > 0 else 0.0,
        achieved_ratio=(
            train_steps * batch_steps / env_steps if env_steps > 0 else 0.0
        ),
        extras=extras,
    )


def _progress(
    mode: str, step: int, total: int, train_steps: int, t0: float, log_every: int
) -> None:
    """Prints a light progress line every ``log_every`` env steps (0 = off)."""
    if log_every > 0 and step % log_every == 0:
        elapsed = time.perf_counter() - t0
        print(
            f"[{mode} {step:>6d}/{total}] train_steps={train_steps}"
            f" elapsed={elapsed:.1f}s",
            flush=True,
        )


def run_interleaved(
    agent: Any,
    experience: Any,
    *,
    total_steps: int,
    batch_size: int,
    seq_len: int,
    train_ratio: float,
    rng_key: jax.Array,
    log_every: int = 0,
) -> LoopStats:
    """Runs the production inline credit schedule and times it.

    Faithful to ``train_loop`` (loops.py:437-465): every env step whose
    post-step buffer size reaches one batch accrues
    ``train_ratio / batch_steps`` credit, and whole credits are drained into
    ``train_step`` calls immediately.

    Args:
        agent: Agent exposing ``act`` and ``train_step``.
        experience: ``ExperienceSource``-shaped collector.
        total_steps: Env steps to run in the timed section.
        batch_size: Replay batch size per train step.
        seq_len: Replay sequence length per train step.
        train_ratio: Replayed-frames-per-env-step target.
        rng_key: JAX PRNG key threaded through acting and training.
        log_every: Print progress every N env steps (0 disables).

    Returns:
        Timing stats for the run.
    """
    batch_steps = batch_size * seq_len
    agent_step = experience.reset()
    credit = 0.0
    train_steps = 0
    t0 = time.perf_counter()
    for step in range(total_steps):
        rng_key, act_key = jax.random.split(rng_key)
        action = agent.act(agent_step.encoder_obs, agent_step.is_first, act_key)
        result = experience.step(action)
        agent_step = result.agent_step

        if experience.buffer_size >= batch_steps:
            credit += train_ratio / batch_steps
            while credit >= 1.0:
                rng_key, train_key = jax.random.split(rng_key)
                batch = experience.sample(batch_size, seq_len)
                agent.train_step(batch, train_key, materialize=False)
                credit -= 1.0
                train_steps += 1
        _progress("interleaved", step, total_steps, train_steps, t0, log_every)

    jax.block_until_ready(agent.params)
    wall = time.perf_counter() - t0
    return _finish_stats(
        "interleaved",
        env_steps=total_steps,
        train_steps=train_steps,
        wall_time_s=wall,
        batch_steps=batch_steps,
        extras={"leftover_credit": credit},
    )


def run_episode(
    agent: Any,
    experience: Any,
    *,
    total_steps: int,
    batch_size: int,
    seq_len: int,
    train_ratio: float,
    rng_key: jax.Array,
    log_every: int = 0,
) -> LoopStats:
    """Runs the episode-boundary schedule: train only when an episode ends.

    Credit accrues exactly as in :func:`run_interleaved`, but whole credits
    are drained only when ``result.done`` is observed (the collector has
    auto-reset into the next episode by then) and once more after the last
    env step, so the total train-step count matches the interleaved schedule
    for the same accrual trace.

    Args:
        agent: Agent exposing ``act`` and ``train_step``.
        experience: ``ExperienceSource``-shaped collector.
        total_steps: Env steps to run in the timed section.
        batch_size: Replay batch size per train step.
        seq_len: Replay sequence length per train step.
        train_ratio: Replayed-frames-per-env-step target.
        rng_key: JAX PRNG key threaded through acting and training.
        log_every: Print progress every N env steps (0 disables).

    Returns:
        Timing stats for the run (``extras`` counts episode boundaries).
    """
    batch_steps = batch_size * seq_len
    agent_step = experience.reset()
    credit = 0.0
    train_steps = 0
    episode_boundaries = 0
    t0 = time.perf_counter()

    def drain(key: jax.Array, count: int) -> tuple[jax.Array, int]:
        """Drains whole credits into train steps; returns (key, new count)."""
        nonlocal credit
        while credit >= 1.0 and experience.buffer_size >= batch_steps:
            key, train_key = jax.random.split(key)
            batch = experience.sample(batch_size, seq_len)
            agent.train_step(batch, train_key, materialize=False)
            credit -= 1.0
            count += 1
        return key, count

    for step in range(total_steps):
        rng_key, act_key = jax.random.split(rng_key)
        action = agent.act(agent_step.encoder_obs, agent_step.is_first, act_key)
        result = experience.step(action)
        agent_step = result.agent_step

        if experience.buffer_size >= batch_steps:
            credit += train_ratio / batch_steps
        if result.done:
            episode_boundaries += 1
            rng_key, train_steps = drain(rng_key, train_steps)
        _progress("episode", step, total_steps, train_steps, t0, log_every)

    # Final drain so deferred credit is not silently dropped on short runs.
    rng_key, train_steps = drain(rng_key, train_steps)

    jax.block_until_ready(agent.params)
    wall = time.perf_counter() - t0
    return _finish_stats(
        "episode",
        env_steps=total_steps,
        train_steps=train_steps,
        wall_time_s=wall,
        batch_steps=batch_steps,
        extras={
            "episode_boundaries": episode_boundaries,
            "leftover_credit": credit,
        },
    )


def run_threaded(
    agent: Any,
    experience: Any,
    *,
    total_steps: int,
    batch_size: int,
    seq_len: int,
    train_ratio: float,
    rng_key: jax.Array,
    log_every: int = 0,
    learner_poll_s: float = 0.001,
) -> LoopStats:
    """Runs actor and learner in two threads over a locked buffer.

    The actor thread runs act -> ``experience.step`` (which adds into the
    buffer — swap in a ``LockedReplayBuffer`` first) and counts
    credit-eligible env steps. The learner thread converts that counter into
    a train-step target (``eligible * train_ratio / batch_steps``) and keeps
    sampling/training toward it, sleeping briefly when it has caught up.
    When the actor finishes it sets the stop event; the learner drains the
    remaining target before exiting, so the achieved ratio matches the
    interleaved schedule up to rounding.

    Args:
        agent: Agent exposing ``act`` and ``train_step``. ``act`` runs on the
            actor thread while ``train_step`` runs on the learner thread.
        experience: ``ExperienceSource``-shaped collector whose ``buffer``
            must already be thread-safe (``LockedReplayBuffer``).
        total_steps: Env steps the actor runs in the timed section.
        batch_size: Replay batch size per train step.
        seq_len: Replay sequence length per train step.
        train_ratio: Replayed-frames-per-env-step target.
        rng_key: JAX PRNG key; split once into actor and learner streams.
        log_every: Actor prints progress every N env steps (0 disables).
        learner_poll_s: Learner sleep while waiting for new credit.

    Returns:
        Timing stats for the run (``extras`` carries the pacing target).

    Raises:
        BaseException: Re-raises the first exception hit by either thread.
    """
    batch_steps = batch_size * seq_len
    actor_key, learner_key = jax.random.split(rng_key)
    counters = {"env_steps": 0, "eligible_steps": 0, "train_steps": 0}
    stop = threading.Event()
    errors: list[BaseException] = []

    def target() -> int:
        """Current train-step target from the actor's eligible-step count."""
        return int(counters["eligible_steps"] * train_ratio / batch_steps)

    def actor() -> None:
        key = actor_key
        t0 = time.perf_counter()
        try:
            agent_step = experience.reset()
            for step in range(total_steps):
                key, act_key = jax.random.split(key)
                action = agent.act(
                    agent_step.encoder_obs, agent_step.is_first, act_key
                )
                result = experience.step(action)
                agent_step = result.agent_step
                counters["env_steps"] += 1
                if experience.buffer_size >= batch_steps:
                    counters["eligible_steps"] += 1
                _progress(
                    "threaded/actor",
                    step,
                    total_steps,
                    counters["train_steps"],
                    t0,
                    log_every,
                )
        except BaseException as exc:  # noqa: BLE001 — surfaced to the caller
            errors.append(exc)
        finally:
            stop.set()

    def learner() -> None:
        key = learner_key
        try:
            while True:
                behind = counters["train_steps"] < target()
                if behind and experience.buffer_size >= batch_steps:
                    key, train_key = jax.random.split(key)
                    batch = experience.sample(batch_size, seq_len)
                    agent.train_step(batch, train_key, materialize=False)
                    counters["train_steps"] += 1
                elif stop.is_set():
                    break
                else:
                    time.sleep(learner_poll_s)
        except BaseException as exc:  # noqa: BLE001 — surfaced to the caller
            errors.append(exc)
            stop.set()

    actor_thread = threading.Thread(target=actor, name="sched-actor")
    learner_thread = threading.Thread(target=learner, name="sched-learner")
    t0 = time.perf_counter()
    actor_thread.start()
    learner_thread.start()
    actor_thread.join()
    learner_thread.join()
    jax.block_until_ready(agent.params)
    wall = time.perf_counter() - t0

    if errors:
        raise errors[0]

    return _finish_stats(
        "threaded",
        env_steps=counters["env_steps"],
        train_steps=counters["train_steps"],
        wall_time_s=wall,
        batch_steps=batch_steps,
        extras={
            "final_target": target(),
            "eligible_steps": counters["eligible_steps"],
        },
    )


LOOPS = {
    "interleaved": run_interleaved,
    "episode": run_episode,
    "threaded": run_threaded,
}
