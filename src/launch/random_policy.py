"""Uniform-random policy behind the agent's functional act contract.

``--random`` scores the random baseline through the exact run loop the trained
agent uses - same collector, same episode metrics, same artifacts. That works
because this policy implements the same functional ``act`` signature as
``R2DreamerAgent``: it simply ignores parameters, observation and carry.

Distinct from ``src.baselines.random_agent.RandomAgent``, which is a
standalone baseline runner with its own CLI and deliberately no JAX
dependency.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from src.r2dreamer.agent import ActState, ParamTree


class RandomPolicy:
    """Uniform-random discrete-action policy with the agent's act signature.

    Args:
        num_actions: Discrete action-space size to sample from.
        seed: Unused; actions derive from the per-step ``rng_key`` the loop
            splits, which is already seeded. Kept so composition can pass the
            run seed uniformly.
    """

    def __init__(self, num_actions: int, seed: int = 42) -> None:
        self.num_actions = int(num_actions)
        self._seed = int(seed)

    @property
    def params(self) -> ParamTree:
        """Empty parameter tree: this policy has nothing to learn."""
        return {}

    def initial_act_state(self) -> ActState:
        """A minimal carry; :meth:`act` returns it untouched."""
        zero = jnp.zeros((1, 1))
        return ActState(stoch=jnp.zeros((1, 1, 1)), deter=zero, prev_action=zero)

    def act(
        self,
        params: ParamTree,
        obs: object,
        is_first: ArrayLike,
        state: ActState,
        rng_key: jax.Array,
        training: ArrayLike = True,
    ) -> tuple[jax.Array, ActState]:
        """Sample one action uniformly; every other input is ignored.

        Args:
            params: Ignored (present for signature compatibility).
            obs: Ignored.
            is_first: Ignored.
            state: Returned unchanged.
            rng_key: Source of the action sample.
            training: Ignored - random is random in both modes.

        Returns:
            ``(action, state)`` with a 0-d int32 action array.
        """
        del params, obs, is_first, training
        action = jax.random.randint(rng_key, (), 0, self.num_actions)
        return action, state
