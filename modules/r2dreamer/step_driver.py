"""StepDriver — owns one acting step end-to-end.

Single source of truth for episode boundaries during an end-to-end run.
On every call to ``step()`` it decides — based on the previous step's
``done`` flag — whether an episode boundary occurred, and if so fires
all three resets atomically:

  1. ``env.reset()``
  2. ``obs_adapter.reset()`` (e.g. flush VGGT extractor KV cache)
  3. zero the RSSM acting state (stoch / deter / prev_action)

The pure policy is delegated to ``Agent.policy_step``; the StepDriver
owns the mutable RSSM acting state that used to live on ``Agent``.
A ``policy="random"`` mode is supported for prefill — the same module
serves both prefill (random actions) and training/eval (agent policy).

See modules/r2dreamer/CONTEXT.md ("Acting" section) for vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from modules.r2dreamer.adapters import ObsAdapter
from modules.r2dreamer.agent import R2DreamerAgent, agent_obs_to_obs_jax


PolicyMode = Literal["agent", "random"]


@dataclass
class Transition:
    """One acting step's contribution to the replay buffer + episode tracking.

    Mirrors the (obs, action, reward, done, terminal) tuple ReplayBuffer.add
    consumes, plus the raw env ``info`` dict for episode-end metric callbacks.
    """
    buffer_obs: np.ndarray
    action: int
    reward: float
    done: bool
    terminal: bool
    info: dict


class StepDriver:
    """One acting step per call. See module docstring for the protocol."""

    def __init__(
        self,
        env: Any,
        agent: R2DreamerAgent,
        obs_adapter: ObsAdapter,
    ):
        self._env = env
        self._agent = agent
        self._adapter = obs_adapter
        self._encoder_type = agent.cfg.encoder_type
        self._num_actions = agent.cfg.num_actions

        cfg = agent.cfg
        self._stoch = np.zeros(
            (1, cfg.stoch_classes, cfg.stoch_discrete), dtype=np.float32,
        )
        self._deter = np.zeros((1, cfg.deter_size), dtype=np.float32)
        self._prev_action = np.zeros((1, cfg.num_actions), dtype=np.float32)

        # Single source of truth for episode boundaries. Initialized True so
        # the first call to step() triggers a fresh start.
        self._previous_done = True
        self._last_buffer_obs: np.ndarray | None = None
        self._last_agent_obs: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        rng_key: jnp.ndarray,
        policy: PolicyMode = "agent",
        training: bool = True,
    ) -> tuple[Transition, bool]:
        """Run one acting step. Returns (transition, episode_just_ended).

        ``episode_just_ended`` is True iff the env reported done=True on
        this step. The caller (trainer) typically uses this to fire
        per-episode logging — the actual reset happens on the *next* call.
        """
        if self._previous_done:
            self._reset_episode()

        action_int = self._choose_action(rng_key, policy, training)
        next_obs = self._env.step(action_int)
        next_buffer_obs, next_agent_obs = self._adapter.transform(next_obs)
        terminal = next_obs.get("success", 0.0) > 0
        done = bool(next_obs["done"])

        transition = Transition(
            buffer_obs=self._last_buffer_obs,
            action=action_int,
            reward=float(next_obs["reward"]),
            done=done,
            terminal=bool(terminal),
            info=next_obs,
        )

        # Roll forward.
        self._previous_done = done
        self._last_buffer_obs = next_buffer_obs
        self._last_agent_obs = next_agent_obs

        return transition, done

    def force_reset(self) -> None:
        """Mark the next ``step()`` as the start of a fresh episode.

        Used by the trainer between prefill and training to guarantee
        training begins on an episode boundary, matching the
        unconditional ``env.reset()`` the original loop did at
        train-loop entry.
        """
        self._previous_done = True

    def begin_episode(self) -> None:
        """Eagerly fire env.reset + adapter.reset + RSSM zero, *now*.

        Use when the caller needs to inspect env state (agent position,
        episode metadata, top-down map) BEFORE the first ``step()`` of a
        new episode. The next ``step()`` runs ``env.step`` (not reset),
        because the episode just started.
        """
        self._reset_episode()
        self._previous_done = False

    def peek_state(self) -> dict:
        """Return a snapshot of the RSSM acting state. Read-only.

        Used by debug-viz (`scripts/debug_viz/evaluate_debug.py`) to dump
        per-step latents. Returns numpy copies — modifying the returned
        arrays does not affect the driver's internal state.
        """
        return {
            "stoch": self._stoch.copy(),
            "deter": self._deter.copy(),
            "prev_action": self._prev_action.copy(),
        }

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_episode(self) -> None:
        """Fire env reset + adapter reset + RSSM zero atomically."""
        obs = self._env.reset()
        self._adapter.reset()
        self._stoch[:] = 0.0
        self._deter[:] = 0.0
        self._prev_action[:] = 0.0
        buffer_obs, agent_obs = self._adapter.transform(obs)
        self._last_buffer_obs = buffer_obs
        self._last_agent_obs = agent_obs

    def _choose_action(
        self, rng_key: jnp.ndarray, policy: PolicyMode, training: bool,
    ) -> int:
        if policy == "random":
            # Match existing prefill RNG semantics (uses np.random, not JAX).
            return int(np.random.randint(0, self._num_actions))

        if policy != "agent":
            raise ValueError(f"Unknown policy mode: {policy!r}")

        obs_jax = agent_obs_to_obs_jax(self._last_agent_obs, self._encoder_type)
        action, new_stoch, new_deter = self._agent.policy_step(
            self._agent.params,
            obs_jax,
            jnp.asarray(self._stoch),
            jnp.asarray(self._deter),
            jnp.asarray(self._prev_action),
            rng_key,
            training,
        )
        action_int = int(action)
        self._stoch = np.array(new_stoch)
        self._deter = np.array(new_deter)
        self._prev_action[:] = 0.0
        self._prev_action[0, action_int] = 1.0
        return action_int
