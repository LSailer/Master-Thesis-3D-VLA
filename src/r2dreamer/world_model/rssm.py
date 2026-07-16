"""Recurrent State-Space Model (R2RSSM) and its block-diagonal primitives.

The RSSM owns the agent's latent dynamics: prior `p(s_t | s_{t-1}, a_{t-1})`,
posterior `q(s_t | s_{t-1}, a_{t-1}, o_t)`, and the deterministic transition
`h_t = f(h_{t-1}, s_{t-1}, a_{t-1})` implemented as a block-GRU.

`RMSNorm` and `BlockLinear` live here too because they are only used by the
RSSM's deterministic head; encoders and MLP heads have their own conventions.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.typing import DTypeLike


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Follows the input dtype: statistics are accumulated in float32 for
    stability, then the normalized output is returned in ``x.dtype`` so
    reduced-precision activations stay reduced through the norm.
    """

    eps: float = 1e-4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(
            jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
            + self.eps
        )
        return (x / rms.astype(x.dtype)) * scale.astype(x.dtype)


class BlockLinear(nn.Module):
    """Block-diagonal linear layer.
    Weight layout: (out_per_block, in_per_block, blocks).
    Matches r2dreamer's einsum: "...gi,oig->...go".
    """

    out_features: int
    blocks: int = 8
    compute_dtype: DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        in_per_block = in_features // self.blocks
        out_per_block = self.out_features // self.blocks

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (out_per_block, in_per_block, self.blocks),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.out_features,))

        # Params stay float32 masters; the einsum runs in compute_dtype
        # (mixed precision when the full_bf16 gate sets it to bfloat16).
        x = x.astype(self.compute_dtype)
        kernel = kernel.astype(self.compute_dtype)
        bias = bias.astype(self.compute_dtype)
        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, self.blocks, in_per_block)
        x = jnp.einsum("...gi,oig->...go", x, kernel)
        x = x.reshape(*batch_shape, self.out_features)
        return x + bias


class Deter(nn.Module):
    """Block-GRU deterministic state transition."""

    deter_size: int = 2048
    stoch_size: int = 512
    act_dim: int = 17
    hidden: int = 256
    blocks: int = 8
    dyn_layers: int = 1
    compute_dtype: DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, stoch, deter, action):
        # stoch: (B, stoch_size), deter: (B, deter_size), action: (B, act_dim)
        # Keep the deterministic recurrent state as a float32 master (like
        # DreamerV3 mixed precision): compute the GRU internally in
        # compute_dtype for speed, but carry `deter` across timesteps in its
        # original dtype so the recurrence stays stable and the jitted acting
        # cond (initial vs. carried state) sees matching dtypes.
        deter_dtype = deter.dtype
        stoch = stoch.astype(self.compute_dtype)
        deter = deter.astype(self.compute_dtype)
        action = action.astype(self.compute_dtype)

        # Normalize action magnitude
        action = action / jnp.clip(jnp.abs(action), min=1.0)

        # Three input projections
        dt = self.compute_dtype
        x0 = nn.silu(
            RMSNorm(name="in_norm0")(nn.Dense(self.hidden, name="in0", dtype=dt)(deter))
        )
        x1 = nn.silu(
            RMSNorm(name="in_norm1")(nn.Dense(self.hidden, name="in1", dtype=dt)(stoch))
        )
        x2 = nn.silu(
            RMSNorm(name="in_norm2")(
                nn.Dense(self.hidden, name="in2", dtype=dt)(action)
            )
        )

        # Concatenate: (B, 3*hidden)
        x = jnp.concatenate([x0, x1, x2], axis=-1)

        # Broadcast across blocks: (B, blocks, 3*hidden)
        x = jnp.broadcast_to(x[:, None, :], (x.shape[0], self.blocks, x.shape[-1]))

        # Per-block deter slice: (B, blocks, deter_size//blocks)
        deter_blocked = deter.reshape(
            deter.shape[0], self.blocks, self.deter_size // self.blocks
        )

        # Combine: (B, blocks, deter/blocks + 3*hidden) -> flatten
        x = jnp.concatenate([deter_blocked, x], axis=-1)
        x = x.reshape(x.shape[0], -1)  # (B, blocks*(deter/blocks + 3*hidden))

        # Hidden layers
        for i in range(self.dyn_layers):
            x = nn.silu(
                RMSNorm(name=f"hid_norm{i}")(
                    BlockLinear(
                        self.deter_size,
                        self.blocks,
                        compute_dtype=dt,
                        name=f"hid{i}",
                    )(x)
                )
            )

        # GRU gates: (B, 3*deter_size)
        gates = BlockLinear(
            3 * self.deter_size, self.blocks, compute_dtype=dt, name="gru"
        )(x)

        # Split block-wise: reshape to (B, blocks, 3*dpb), then chunk
        dpb = self.deter_size // self.blocks
        gates = gates.reshape(gates.shape[0], self.blocks, 3 * dpb)
        gate_chunks = jnp.split(gates, 3, axis=-1)  # 3x (B, blocks, dpb)
        reset = jax.nn.sigmoid(gate_chunks[0].reshape(gates.shape[0], -1))
        cand = gate_chunks[1].reshape(gates.shape[0], -1)
        update = jax.nn.sigmoid(gate_chunks[2].reshape(gates.shape[0], -1) - 1.0)

        new_deter = update * jnp.tanh(reset * cand) + (1.0 - update) * deter
        return new_deter.astype(deter_dtype)


class R2RSSM(nn.Module):
    """Recurrent State-Space Model with Block-GRU (R2-Dreamer).

    stoch is always (B, stoch_classes, stoch_discrete) — never flattened
    in the external interface. Internally, stoch is flattened to (B, S*K)
    before feeding into Deter.
    """

    deter_size: int = 2048
    stoch_classes: int = 32
    stoch_discrete: int = 16
    num_actions: int = 17
    hidden: int = 256
    blocks: int = 8
    dyn_layers: int = 1
    obs_layers: int = 1
    img_layers: int = 2
    unimix_ratio: float = 0.01
    compute_dtype: DTypeLike = jnp.float32

    @property
    def stoch_size(self):
        """Return total stochastic latent dimensionality."""
        return self.stoch_classes * self.stoch_discrete

    @property
    def feat_size(self):
        """Return RSSM feature size (stochastic + deterministic)."""
        return self.stoch_size + self.deter_size

    def setup(self):
        """Create deterministic core and prior/posterior heads."""
        self.deter_net = Deter(
            deter_size=self.deter_size,
            stoch_size=self.stoch_classes * self.stoch_discrete,
            act_dim=self.num_actions,
            hidden=self.hidden,
            blocks=self.blocks,
            dyn_layers=self.dyn_layers,
            compute_dtype=self.compute_dtype,
        )

        # Posterior head (obs_net): obs_layers Dense+RMSNorm+SiLU then Dense→logits
        dt = self.compute_dtype
        self.obs_fcs = [
            nn.Dense(self.hidden, name=f"obs_fc{i}", dtype=dt)
            for i in range(self.obs_layers)
        ]
        self.obs_norms = [RMSNorm(name=f"obs_norm{i}") for i in range(self.obs_layers)]
        self.obs_out = nn.Dense(
            self.stoch_classes * self.stoch_discrete, name="obs_out", dtype=dt
        )

        # Prior head (img_net): img_layers Dense+RMSNorm+SiLU then Dense→logits
        self.img_fcs = [
            nn.Dense(self.hidden, name=f"img_fc{i}", dtype=dt)
            for i in range(self.img_layers)
        ]
        self.img_norms = [RMSNorm(name=f"img_norm{i}") for i in range(self.img_layers)]
        self.img_out = nn.Dense(
            self.stoch_classes * self.stoch_discrete, name="img_out", dtype=dt
        )

    def __call__(self, stoch, deter, action, embed):
        """Single posterior step: Deter transition then posterior head.

        Args:
            stoch: (B, stoch_classes, stoch_discrete)
            deter: (B, deter_size)
            action: (B, num_actions)
            embed: (B, embed_dim) — encoder output

        Returns:
            new_stoch: (B, stoch_classes, stoch_discrete)
            new_deter: (B, deter_size)
            post_logit: (B, stoch_classes, stoch_discrete)
        """
        B = stoch.shape[0]
        stoch_flat = stoch.reshape(B, -1)
        deter = self.deter_net(stoch_flat, deter, action)

        # Touch prior head so its params are created during init
        self._prior(deter)

        # Posterior: condition on deter + embed
        x = jnp.concatenate([deter, embed.astype(deter.dtype)], axis=-1)
        for fc, norm in zip(self.obs_fcs, self.obs_norms):
            x = nn.silu(norm(fc(x)))
        # Logits are pinned float32 so sampling, KL, and entropy stay
        # full-precision even when the layers above compute in bfloat16.
        logit = (
            self.obs_out(x)
            .astype(jnp.float32)
            .reshape(B, self.stoch_classes, self.stoch_discrete)
        )
        stoch = self._sample(logit)
        return stoch, deter, logit

    def img_step(self, stoch, deter, action):
        """Single prior step: Deter transition then prior head.

        Args:
            stoch: (B, stoch_classes, stoch_discrete)
            deter: (B, deter_size)
            action: (B, num_actions)

        Returns:
            new_stoch: (B, stoch_classes, stoch_discrete)
            new_deter: (B, deter_size)
        """
        B = stoch.shape[0]
        stoch_flat = stoch.reshape(B, -1)
        deter = self.deter_net(stoch_flat, deter, action)
        stoch, _ = self._prior(deter)
        return stoch, deter

    def _prior(self, deter):
        """Compute prior logits and sample from deter only."""
        B = deter.shape[0]
        x = deter
        for fc, norm in zip(self.img_fcs, self.img_norms):
            x = nn.silu(norm(fc(x)))
        # Same float32 logit pin as the posterior head (see __call__).
        logit = (
            self.img_out(x)
            .astype(jnp.float32)
            .reshape(B, self.stoch_classes, self.stoch_discrete)
        )
        stoch = self._sample(logit)
        return stoch, logit

    def prior(self, deter):
        """Public prior: returns (stoch, logit)."""
        return self._prior(deter)

    def observe(self, embed, actions, initial, is_first):
        """Roll out posterior over T timesteps.

        Args:
            embed: (B, T, embed_dim)
            actions: (B, T, num_actions)
            initial: (stoch0, deter0) — initial states
            is_first: (B, T) — 1.0 on episode boundaries

        Returns:
            stochs: (B, T, stoch_classes, stoch_discrete)
            deters: (B, T, deter_size)
            logits: (B, T, stoch_classes, stoch_discrete)
        """
        stoch, deter = initial
        stochs, deters, logits = [], [], []

        for t in range(embed.shape[1]):
            # Reset mechanism: zero out state on episode boundaries
            mask = 1.0 - is_first[:, t]
            stoch = stoch * mask[:, None, None]
            deter = deter * mask[:, None]
            action = actions[:, t] * mask[:, None]

            stoch, deter, logit = self(stoch, deter, action, embed[:, t])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)

        return (
            jnp.stack(stochs, axis=1),
            jnp.stack(deters, axis=1),
            jnp.stack(logits, axis=1),
        )

    def get_feat(self, stoch, deter):
        """Flatten stoch and concat with deter to form the feature vector.

        Args:
            stoch: (..., stoch_classes, stoch_discrete)
            deter: (..., deter_size)

        Returns:
            feat: (..., stoch_size + deter_size)
        """
        flat = stoch.reshape(
            *stoch.shape[:-2], self.stoch_classes * self.stoch_discrete
        )
        return jnp.concatenate([flat, deter], axis=-1)

    def initial_state(self, batch_size):
        """Return zero initial state."""
        return (
            jnp.zeros((batch_size, self.stoch_classes, self.stoch_discrete)),
            jnp.zeros((batch_size, self.deter_size)),
        )

    def _sample(self, logits):
        """Unimix + straight-through Gumbel-Softmax (hard=True).

        Uses ``self.make_rng('sample')`` so that callers pass the RNG
        collection via ``apply(..., rngs={'sample': key})``.
        """
        if self.unimix_ratio > 0:
            probs = jax.nn.softmax(logits, axis=-1)
            uniform = jnp.ones_like(probs) / self.stoch_discrete
            probs = (1 - self.unimix_ratio) * probs + self.unimix_ratio * uniform
            logits = jnp.log(probs + 1e-8)
        # Gumbel-Softmax: stochastic forward, soft gradient backward
        rng = self.make_rng("sample")
        gumbel_noise = -jnp.log(
            -jnp.log(jax.random.uniform(rng, logits.shape, minval=1e-20)) + 1e-20
        )
        soft = jax.nn.softmax(logits + gumbel_noise, axis=-1)
        hard = jax.nn.one_hot(
            jnp.argmax(logits + gumbel_noise, axis=-1), self.stoch_discrete
        )
        return hard + soft - jax.lax.stop_gradient(soft)
