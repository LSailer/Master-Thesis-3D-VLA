"""R2-Dreamer network modules — JAX/Flax port of NM512/r2dreamer."""

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------- TwoHot Distribution (R2-Dreamer real-space parameterization) ----------


def _symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def _make_bins(num_bins: int = 255):
    """Build R2-Dreamer's symexp-spaced bins in real space.

    Matches PyTorch: symexp(linspace(-20, 0, half)) mirrored.
    """
    if num_bins % 2 == 1:
        half = jnp.linspace(-20.0, 0.0, (num_bins - 1) // 2 + 1)
        half = _symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]])
    else:
        half = jnp.linspace(-20.0, 0.0, num_bins // 2)
        half = _symexp(half)
        bins = jnp.concatenate([half, -half[::-1]])
    return bins


class R2TwoHotDist:
    """TwoHot distribution matching R2-Dreamer's real-space parameterization.

    Unlike DreamerV3's TwoHotDist (which operates in symlog space), this uses
    bins = symexp(linspace(-20, 0, 128)) mirrored to 255 bins in real space.
    No symlog/symexp transform is applied to targets or predictions.
    """

    def __init__(self, num_bins: int = 255):
        self.num_bins = num_bins
        self.bins = _make_bins(num_bins)

    def encode(self, target: jnp.ndarray) -> jnp.ndarray:
        """Two-hot encode a scalar target in real space. target: (...)."""
        # Find below/above indices (matching PyTorch's logic)
        below = jnp.sum((self.bins <= target[..., None]).astype(jnp.int32), axis=-1) - 1
        above = self.num_bins - jnp.sum((self.bins > target[..., None]).astype(jnp.int32), axis=-1)
        below = jnp.clip(below, 0, self.num_bins - 1)
        above = jnp.clip(above, 0, self.num_bins - 1)
        equal = below == above
        dist_to_below = jnp.where(equal, 1.0, jnp.abs(self.bins[below] - target))
        dist_to_above = jnp.where(equal, 1.0, jnp.abs(self.bins[above] - target))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        return (weight_below[..., None] * jax.nn.one_hot(below, self.num_bins) +
                weight_above[..., None] * jax.nn.one_hot(above, self.num_bins))

    def loss(self, logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Cross-entropy loss. logits: (..., bins), target: (...) in real space."""
        twohot = jax.lax.stop_gradient(self.encode(target))
        log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        return -(twohot * log_probs).sum(axis=-1)

    def pred(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Expected value in real space (symmetric sum for numerical stability).

        logits: (..., bins). Returns: (..., 1) to match PyTorch's mode() shape.
        """
        probs = jax.nn.softmax(logits, axis=-1)
        n = self.num_bins
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = probs[..., :m]
            p2 = probs[..., m:m + 1]
            p3 = probs[..., m + 1:]
            b1 = self.bins[:m]
            b2 = self.bins[m:m + 1]
            b3 = self.bins[m + 1:]
            wavg = (jnp.sum(p2 * b2, axis=-1, keepdims=True) +
                    jnp.sum(p1[..., ::-1] * b1[::-1] + p3 * b3, axis=-1, keepdims=True))
        else:
            p1 = probs[..., :n // 2]
            p2 = probs[..., n // 2:]
            b1 = self.bins[:n // 2]
            b2 = self.bins[n // 2:]
            wavg = jnp.sum(p1[..., ::-1] * b1[::-1] + p2 * b2, axis=-1, keepdims=True)
        return wavg  # (..., 1) — real space, no unsquash needed


def onehot_mode_st(logits: jnp.ndarray, unimix_ratio: float = 0.0) -> jnp.ndarray:
    """Straight-through one-hot at argmax — JAX port of OneHotDist.mode.

    Forward: ``one_hot(argmax(log_probs))`` where ``log_probs`` are unimix-mixed
    log-softmax of ``logits``. Backward: identity through ``log_probs``, which
    propagates the ``log_softmax`` Jacobian back to ``logits``.

    Matches PyTorch R2-Dreamer (``distributions.py``): ``OneHotDist`` reparameterises
    ``self.logits = log(softmax(input) * (1-r) + r/K)`` in ``__init__``, then
    ``mode`` returns ``one_hot(argmax).detach() + self.logits - self.logits.detach()``.
    """
    K = logits.shape[-1]
    if unimix_ratio > 0:
        probs = jax.nn.softmax(logits, axis=-1)
        probs = (1.0 - unimix_ratio) * probs + unimix_ratio / K
        log_probs = jnp.log(probs + 1e-8)
    else:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
    hard = jax.nn.one_hot(jnp.argmax(log_probs, axis=-1), K)
    return hard + log_probs - jax.lax.stop_gradient(log_probs)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    eps: float = 1e-4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * scale


class BlockLinear(nn.Module):
    """Block-diagonal linear layer.
    Weight layout: (out_per_block, in_per_block, blocks).
    Matches r2dreamer's einsum: "...gi,oig->...go".
    """
    out_features: int
    blocks: int = 8

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

    @nn.compact
    def __call__(self, stoch, deter, action):
        # stoch: (B, stoch_size), deter: (B, deter_size), action: (B, act_dim)

        # Normalize action magnitude
        action = action / jnp.clip(jnp.abs(action), min=1.0)

        # Three input projections
        x0 = nn.silu(RMSNorm(name="in_norm0")(nn.Dense(self.hidden, name="in0")(deter)))
        x1 = nn.silu(RMSNorm(name="in_norm1")(nn.Dense(self.hidden, name="in1")(stoch)))
        x2 = nn.silu(RMSNorm(name="in_norm2")(nn.Dense(self.hidden, name="in2")(action)))

        # Concatenate: (B, 3*hidden)
        x = jnp.concatenate([x0, x1, x2], axis=-1)

        # Broadcast across blocks: (B, blocks, 3*hidden)
        x = jnp.broadcast_to(x[:, None, :], (x.shape[0], self.blocks, x.shape[-1]))

        # Per-block deter slice: (B, blocks, deter_size//blocks)
        deter_blocked = deter.reshape(deter.shape[0], self.blocks, self.deter_size // self.blocks)

        # Combine: (B, blocks, deter/blocks + 3*hidden) -> flatten
        x = jnp.concatenate([deter_blocked, x], axis=-1)
        x = x.reshape(x.shape[0], -1)  # (B, blocks*(deter/blocks + 3*hidden))

        # Hidden layers
        for i in range(self.dyn_layers):
            x = nn.silu(RMSNorm(name=f"hid_norm{i}")(
                BlockLinear(self.deter_size, self.blocks, name=f"hid{i}")(x)))

        # GRU gates: (B, 3*deter_size)
        gates = BlockLinear(3 * self.deter_size, self.blocks, name="gru")(x)

        # Split block-wise: reshape to (B, blocks, 3*dpb), then chunk
        dpb = self.deter_size // self.blocks
        gates = gates.reshape(gates.shape[0], self.blocks, 3 * dpb)
        gate_chunks = jnp.split(gates, 3, axis=-1)  # 3x (B, blocks, dpb)
        reset = jax.nn.sigmoid(gate_chunks[0].reshape(gates.shape[0], -1))
        cand = gate_chunks[1].reshape(gates.shape[0], -1)
        update = jax.nn.sigmoid(gate_chunks[2].reshape(gates.shape[0], -1) - 1.0)

        return update * jnp.tanh(reset * cand) + (1.0 - update) * deter


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

    @property
    def stoch_size(self):
        return self.stoch_classes * self.stoch_discrete

    @property
    def feat_size(self):
        return self.stoch_size + self.deter_size

    def setup(self):
        self.deter_net = Deter(
            deter_size=self.deter_size,
            stoch_size=self.stoch_classes * self.stoch_discrete,
            act_dim=self.num_actions,
            hidden=self.hidden,
            blocks=self.blocks,
            dyn_layers=self.dyn_layers,
        )

        # Posterior head (obs_net): obs_layers Dense+RMSNorm+SiLU then Dense→logits
        self.obs_fcs = [nn.Dense(self.hidden, name=f"obs_fc{i}") for i in range(self.obs_layers)]
        self.obs_norms = [RMSNorm(name=f"obs_norm{i}") for i in range(self.obs_layers)]
        self.obs_out = nn.Dense(self.stoch_classes * self.stoch_discrete, name="obs_out")

        # Prior head (img_net): img_layers Dense+RMSNorm+SiLU then Dense→logits
        self.img_fcs = [nn.Dense(self.hidden, name=f"img_fc{i}") for i in range(self.img_layers)]
        self.img_norms = [RMSNorm(name=f"img_norm{i}") for i in range(self.img_layers)]
        self.img_out = nn.Dense(self.stoch_classes * self.stoch_discrete, name="img_out")

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
        x = jnp.concatenate([deter, embed], axis=-1)
        for fc, norm in zip(self.obs_fcs, self.obs_norms):
            x = nn.silu(norm(fc(x)))
        logit = self.obs_out(x).reshape(B, self.stoch_classes, self.stoch_discrete)
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
        logit = self.img_out(x).reshape(B, self.stoch_classes, self.stoch_discrete)
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

        return jnp.stack(stochs, axis=1), jnp.stack(deters, axis=1), jnp.stack(logits, axis=1)

    def get_feat(self, stoch, deter):
        """Flatten stoch and concat with deter to form the feature vector.

        Args:
            stoch: (..., stoch_classes, stoch_discrete)
            deter: (..., deter_size)

        Returns:
            feat: (..., stoch_size + deter_size)
        """
        flat = stoch.reshape(*stoch.shape[:-2], self.stoch_classes * self.stoch_discrete)
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
        gumbel_noise = -jnp.log(-jnp.log(
            jax.random.uniform(rng, logits.shape, minval=1e-20)
        ) + 1e-20)
        soft = jax.nn.softmax(logits + gumbel_noise, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(logits + gumbel_noise, axis=-1), self.stoch_discrete)
        return hard + soft - jax.lax.stop_gradient(soft)


class R2Encoder(nn.Module):
    """Convolutional encoder ported from R2-Dreamer.

    Expects obs in CHW format (JAX codebase convention: B, C, H, W).
    Applies Conv+MaxPool+RMSNorm+SiLU for each channel multiplier, then flattens.
    """
    depth: int = 16
    kernel_size: int = 5
    mults: tuple = (2, 3, 4, 4)

    @nn.compact
    def __call__(self, obs):
        # obs: (B, C, H, W) float [0,1]
        x = obs - 0.5
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        for i, mult in enumerate(self.mults):
            ch = self.depth * mult
            x = nn.Conv(ch, (self.kernel_size, self.kernel_size),
                        padding="SAME", name=f"conv{i}")(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        # Transpose back to NCHW before flatten to match PyTorch's flatten order
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        return x.reshape(x.shape[0], -1)


class R2MLP(nn.Module):
    """MLP with RMSNorm, matching PyTorch R2-Dreamer's MLP + MLPHead."""
    hidden: int = 256
    layers: int = 2
    out_dim: int = 1
    outscale: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.layers):
            x = nn.Dense(self.hidden, name=f"fc{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        if self.outscale == 0.0:
            out_init = nn.initializers.zeros
        elif self.outscale != 1.0:
            base_init = nn.initializers.lecun_normal()
            def out_init(key, shape, dtype=jnp.float32):
                return base_init(key, shape, dtype) * self.outscale
        else:
            out_init = nn.initializers.lecun_normal()
        x = nn.Dense(self.out_dim, kernel_init=out_init, name="out")(x)
        return x


class VGGTEncoder(nn.Module):
    """Linear projection encoder for VGGT features.

    Takes flattened world_points (37*37*3=4107) + camera_pose (9) = 4116 dim
    and projects to embed_dim via a single Dense layer.
    """
    embed_dim: int = 1024

    @nn.compact
    def __call__(self, obs):
        # obs: (B, 4116) float32 — already flat
        out = nn.Dense(self.embed_dim, name="proj")(obs)
        # === BP #4 (inside JIT — debug walkthrough — uncomment to re-enable) ===
        # import jax as _jax_dbg
        # _jax_dbg.debug.print(
        #     "[BP#4] encoder out: shape={s} mean={m} std={st} min={mn} max={mx}",
        #     s=out.shape, m=out.mean(), st=out.std(), mn=out.min(), mx=out.max(),
        # )
        return out


class VGGTAggregatorMLPEncoder(nn.Module):
    """Variant 1 encoder for all pre-head VGGT aggregator tokens.

    Architecture follows the paper-style VGGT-DP/VGGT-World token path:
    (B, N, D) all tokens, including camera/register special tokens and patch
    tokens -> tokenwise linear projection -> mean pool over tokens.
    """
    embed_dim: int = 1024
    channels: int = 64
    hidden: int = 1024

    @nn.compact
    def __call__(self, obs):
        # obs: (B, N, D) float32 all-token pre-head aggregator features.
        if obs.ndim != 3:
            raise ValueError(f"expected (B, N, D) all-token VGGT features, got {obs.shape}")
        tokens = nn.Dense(self.embed_dim, name="proj")(obs)
        return tokens.mean(axis=1)


class Projector(nn.Module):
    """Single linear projection without bias (maps feat_size -> embed_dim)."""
    out_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.out_dim, use_bias=False, name="proj")(x)


class ReturnEMA:
    """Tracks 5th/95th percentile of returns with exponential moving average.

    state: jnp.array([p05_ema, p95_ema]) initialised at zeros.
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def init_state(self):
        return jnp.zeros(2)

    def update(self, state, returns):
        quantiles = jnp.array([
            jnp.percentile(returns, 5),
            jnp.percentile(returns, 95),
        ])
        return self.alpha * quantiles + (1 - self.alpha) * state

    def get_stats(self, state):
        offset = state[0]
        scale = jnp.maximum(state[1] - state[0], 1.0)
        return offset, scale
