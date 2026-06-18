"""Return EMA — tracks 5th/95th percentile of imagined returns.

Used by `behavior.loss` to scale the policy advantage.
"""

import jax.numpy as jnp


class ReturnEMA:
    """Tracks 5th/95th percentile of returns with exponential moving average.

    state: jnp.array([p05_ema, p95_ema]) initialised at zeros.
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def init_state(self):
        return jnp.zeros(2)

    def update(self, state, returns):
        quantiles = jnp.array(
            [
                jnp.percentile(returns, 5),
                jnp.percentile(returns, 95),
            ]
        )
        return self.alpha * quantiles + (1 - self.alpha) * state

    def get_stats(self, state):
        offset = state[0]
        scale = jnp.maximum(state[1] - state[0], 1.0)
        return offset, scale
