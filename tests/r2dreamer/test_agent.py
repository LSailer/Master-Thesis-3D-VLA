"""Behavior tests for R2DreamerAgent on the routed-adapter surface.

The agent takes ``fields=`` - one adapter call's routed output - as its
architecture description, so these tests build ``AdapterField``s directly and
assert on what the agent does with them: encoder composition, acting, the train
step's determinism and loss composition, the decoder probe, and how a live
(``buffer=False``) field travels on ``batch.global_feature``.

Per-variant coverage of the registered adapters lives in
``tests/adapters/test_routed_pipeline.py``; nothing here duplicates it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.adapters.contract import AdapterField, AdapterOutput, Encoder
from src.buffer.replay_buffer import ReplayBatch
from src.configs.config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent

IMAGE_SHAPE = (64, 64, 3)
POSE_DIM = 9
CLOUD_ROWS = 64
POINT_DIM = 6
# encoder_mults=(1, 1, 1, 1) at depth 4: 64x64 -> 4x4, flattened at 4 channels.
CONV_EMBED_DIM = 4 * 4 * 4
FUSION_DIM = 24

# Small branches keep every forward pass on CPU cheap; the routing is unchanged.
SMALL_BRANCHES: dict[str, object] = {
    "branch_embed_dim": 16,
    "mlp_hidden": 8,
    "pointnet_num_points": 32,
    "fusion_dim": FUSION_DIM,
}


def _cfg(**overrides) -> R2DreamerConfig:
    params: dict[str, object] = dict(
        # Pure provenance: the architecture comes from ``fields``, not from here.
        adapter="rgb",
        num_actions=4,
        deter_size=32,
        hidden_size=16,
        stoch_classes=4,
        stoch_discrete=4,
        blocks=4,
        encoder_depth=4,
        encoder_kernel=3,
        encoder_mults=(1, 1, 1, 1),
        mlp_units=16,
        mlp_layers_reward=1,
        mlp_layers_cont=1,
        mlp_layers_actor=1,
        mlp_layers_critic=1,
        twohot_bins=21,
        imagination_horizon=2,
        horizon=10,
        lr=1e-3,
        warmup_steps=0,
    )
    params.update(overrides)
    return R2DreamerConfig(**params)


def _image_field(key: str = "image", *, decoder_target: bool = True) -> AdapterField:
    return AdapterField(
        key=key,
        encoder=Encoder.CONV,
        buffer=True,
        value=jnp.zeros(IMAGE_SHAPE, jnp.uint8),
        decoder_target=decoder_target,
    )


def _pose_field(key: str = "camera_pose") -> AdapterField:
    return AdapterField(
        key=key,
        encoder=Encoder.MLP,
        buffer=True,
        value=jnp.zeros((POSE_DIM,), jnp.float32),
    )


def _cloud_field(key: str = "house_context") -> AdapterField:
    return AdapterField(
        key=key,
        encoder=Encoder.POINTNET,
        buffer=False,
        value=jax.random.uniform(
            jax.random.PRNGKey(4), (CLOUD_ROWS, POINT_DIM), dtype=jnp.float32
        ).astype(jnp.float16),
    )


def _agent(cfg: R2DreamerConfig, fields: AdapterOutput, *, seed: int = 42):
    return R2DreamerAgent(
        cfg, jax.random.PRNGKey(seed), fields=fields, encoder_overrides=SMALL_BRANCHES
    )


def _step_value(field: AdapterField, batch_size: int, seq_len: int) -> jnp.ndarray:
    shape = (batch_size, seq_len, *field.value.shape)
    if field.value.dtype == jnp.uint8:
        # Structured, not constant: a constant frame makes any conv regression
        # invisible, since every window would encode identically.
        pixels = jnp.arange(int(np.prod(shape)), dtype=jnp.int32) % 256
        return pixels.reshape(shape).astype(jnp.uint8)
    values = jnp.linspace(-1.0, 1.0, int(np.prod(shape)), dtype=jnp.float32)
    return values.reshape(shape).astype(field.value.dtype)


def _batch(
    cfg: R2DreamerConfig,
    fields: AdapterOutput,
    *,
    batch_size: int = 2,
    seq_len: int = 3,
    global_feature: jnp.ndarray | None = ...,
) -> ReplayBatch:
    """Build the batch the buffer would hand back for ``fields``.

    Replayed fields get a ``(B, T)`` prefix; the live field rides along as one
    ``global_feature`` with no prefix, exactly as ``ReplayBuffer`` stores it.
    """
    live = [f for f in fields if not f.buffer]
    if global_feature is ...:
        global_feature = live[0].value if live else None
    action_ids = jnp.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    return ReplayBatch(
        obs={
            f.key: _step_value(f, batch_size, seq_len) for f in fields if f.buffer
        },
        actions=jax.nn.one_hot(
            action_ids % cfg.num_actions, cfg.num_actions, dtype=jnp.float32
        ),
        rewards=jnp.linspace(
            -1.0, 1.0, batch_size * seq_len, dtype=jnp.float32
        ).reshape(batch_size, seq_len),
        is_first=jnp.zeros((batch_size, seq_len), jnp.float32).at[:, 0].set(1.0),
        is_episode_end=jnp.zeros((batch_size, seq_len), jnp.float32).at[:, -1].set(1.0),
        global_feature=global_feature,
        encoders={f.key: f.encoder.value for f in fields},
    )


def _live_obs(fields: AdapterOutput) -> dict[str, jnp.ndarray]:
    """One env step's encoder obs: every field's value, no leading dims."""
    return {f.key: f.value for f in fields}


def _tree_allclose(left, right, *, atol=1e-6) -> bool:
    pairs = zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    return all(np.allclose(np.asarray(a), np.asarray(b), atol=atol) for a, b in pairs)


def _tree_any_changed(before, after, *, atol=1e-7) -> bool:
    pairs = zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after))
    return any(
        not np.allclose(np.asarray(a), np.asarray(b), atol=atol) for a, b in pairs
    )


@pytest.fixture(name="rgb_agent")
def rgb_agent_fixture():
    """The single-branch baseline: one replayed RGB field on a conv branch."""
    fields = [_image_field()]
    return _agent(_cfg(), fields), fields


class TestConstruction:
    def test_fields_are_required(self):
        # The architecture has exactly one source now; there is no config
        # fallback to fall back to.
        with pytest.raises(TypeError):
            R2DreamerAgent(_cfg(), jax.random.PRNGKey(0))

    def test_embed_size_is_the_single_branch_width(self, rgb_agent):
        agent, _fields = rgb_agent
        assert agent.embed_size == CONV_EMBED_DIM

    def test_embed_size_is_the_fusion_width_once_branches_are_composed(self):
        # Whatever a variant observes, the RSSM sees a fixed input width.
        fields = [_image_field(), _pose_field(), _cloud_field()]
        agent = _agent(_cfg(), fields)

        assert agent.embed_size == FUSION_DIM

    def test_encoder_params_carry_one_named_subtree_per_route(self):
        fields = [_image_field(), _pose_field()]
        agent = _agent(_cfg(), fields)

        assert {"conv_image", "mlp_camera_pose", "fusion"} <= set(
            agent.params["encoder"]["params"]
        )

    def test_decoder_is_left_unbuilt_by_default(self, rgb_agent):
        agent, _fields = rgb_agent
        assert "decoder" not in agent.params
        assert agent.reconstruct(_batch(agent.cfg, _fields)) is None

    def test_decoder_needs_a_flagged_field(self):
        fields = [_image_field(decoder_target=False)]
        with pytest.raises(ValueError, match="decoder_target"):
            _agent(_cfg(decoder=True), fields)


class TestActing:
    def test_act_returns_an_action_in_range(self, rgb_agent):
        agent, fields = rgb_agent

        action = agent.act(_live_obs(fields), True, jax.random.PRNGKey(0))

        assert 0 <= action < agent.cfg.num_actions

    def test_act_with_state_matches_mutable_act_and_jits(self, rgb_agent):
        agent, fields = rgb_agent
        state = agent.initial_act_state()
        encoder_obs = _live_obs(fields)

        for step, is_first in enumerate((True, False, True, False)):
            key = jax.random.PRNGKey(step)
            mutable_action = agent.act(encoder_obs, is_first, key, training=False)
            state_action, state = agent.act_with_state(
                encoder_obs, is_first, state, key, training=False
            )
            assert state_action == mutable_action
            assert _tree_allclose(state, agent.snapshot_act_state())

        compiled = jax.jit(agent.act_with_state_pure)
        action, _state = compiled(
            agent.params,
            {"image": encoder_obs["image"][None]},
            agent.initial_act_state(),
            jnp.asarray(True),
            jax.random.PRNGKey(99),
            False,
        )
        assert 0 <= int(action) < agent.cfg.num_actions

    def test_act_does_not_add_a_batch_dim_to_the_live_field(self):
        # The live field is one global event the encoder broadcasts itself; an
        # extra axis would trip the cloud branch's shape check.
        fields = [_image_field(), _cloud_field()]
        agent = _agent(_cfg(), fields)

        batched = agent._batch_live_obs(_live_obs(fields))

        assert batched["image"].shape == (1, *IMAGE_SHAPE)
        assert batched["house_context"].shape == (CLOUD_ROWS, POINT_DIM)
        assert 0 <= agent.act(_live_obs(fields), True, jax.random.PRNGKey(1)) < 4


class TestTrainStep:
    def test_produces_finite_metrics_for_every_sub_loss(self, rgb_agent):
        agent, fields = rgb_agent

        metrics = agent.train_step(_batch(agent.cfg, fields), jax.random.PRNGKey(1))

        for key in ("barlow", "dyn", "rep", "rew", "con", "policy", "value", "repval"):
            assert f"loss/{key}" in metrics
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} = {value}"

    def test_repeated_steps_stay_finite(self, rgb_agent):
        agent, fields = rgb_agent
        batch = _batch(agent.cfg, fields)
        rng = jax.random.PRNGKey(2)

        for _ in range(3):
            rng, key = jax.random.split(rng)
            metrics = agent.train_step(batch, key)
            assert np.isfinite(metrics["total_loss"])

    def test_is_deterministic_updates_params_and_composes_total_loss(self):
        cfg = _cfg()
        fields = [_image_field()]
        batch = _batch(cfg, fields)
        train_rng = jax.random.PRNGKey(11)
        agent_a = _agent(cfg, fields, seed=7)
        agent_b = _agent(cfg, fields, seed=7)
        before = jax.tree.map(jnp.copy, agent_a.params)

        metrics_a = agent_a.train_step(batch, train_rng)
        metrics_b = agent_b.train_step(batch, train_rng)

        assert metrics_a.keys() == metrics_b.keys()
        for key in metrics_a:
            assert metrics_a[key] == pytest.approx(metrics_b[key], abs=1e-6)
        assert _tree_allclose(agent_a.params, agent_b.params)
        assert metrics_a["nan_skipped"] == 0.0
        assert [
            name
            for name, params in agent_a.params.items()
            if _tree_any_changed(before[name], params)
        ]

        expected_total = (
            cfg.scale_dyn * metrics_a["loss/dyn"]
            + cfg.scale_rep * metrics_a["loss/rep"]
            + cfg.scale_barlow * metrics_a["loss/barlow"]
            + cfg.scale_rew * metrics_a["loss/rew"]
            + cfg.scale_con * metrics_a["loss/con"]
            + cfg.scale_policy * metrics_a["loss/policy"]
            + cfg.scale_value * metrics_a["loss/value"]
            + cfg.scale_repval * metrics_a["loss/repval"]
        )
        assert metrics_a["total_loss"] == pytest.approx(expected_total, rel=1e-6)

    def test_a_live_field_is_read_off_the_batch_global_feature(self):
        # The cloud is never stored per step; the agent merges the batch's single
        # global feature back under its routed key before encoding.
        fields = [_image_field(), _pose_field(), _cloud_field()]
        agent = _agent(_cfg(), fields)
        before = jax.tree.map(jnp.copy, agent.params["encoder"])

        metrics = agent.train_step(_batch(agent.cfg, fields), jax.random.PRNGKey(3))

        assert np.isfinite(metrics["total_loss"])
        assert _tree_any_changed(before, agent.params["encoder"])

    def test_a_batch_without_the_live_field_is_rejected(self):
        # A silently dropped global feature would train the cloud branch on
        # zeros; fail loudly with the routed key in the message instead.
        fields = [_image_field(), _cloud_field()]
        agent = _agent(_cfg(), fields)
        batch = _batch(agent.cfg, fields, global_feature=None)

        with pytest.raises(ValueError, match="global_feature"):
            agent.train_step(batch, jax.random.PRNGKey(5))

    def test_params_stay_float32_under_the_full_bf16_gate(self):
        # ``full_bf16`` changes compute, not storage: the params pytree (and so
        # every checkpoint) must stay float32.
        fields = [_image_field()]
        agent = _agent(_cfg(full_bf16=True, compute_dtype="bfloat16"), fields)

        metrics = agent.train_step(_batch(agent.cfg, fields), jax.random.PRNGKey(6))

        assert all(
            leaf.dtype == jnp.float32
            for leaf in jax.tree_util.tree_leaves(agent.params)
        )
        assert np.isfinite(metrics["total_loss"])


class TestDecoderProbe:
    def test_probe_reconstructs_the_flagged_field(self):
        # Two conv fields: the probe must follow the flag, not the key order.
        fields = [
            _image_field("first", decoder_target=False),
            _image_field("second", decoder_target=True),
        ]
        agent = _agent(_cfg(decoder=True), fields)
        batch = _batch(agent.cfg, fields, batch_size=1, seq_len=2)

        target, recon = agent.reconstruct(batch)

        assert target.shape == (2, *IMAGE_SHAPE)
        assert recon.shape == (2, *IMAGE_SHAPE)
        expected = batch.obs["second"].astype(jnp.float32).reshape(2, *IMAGE_SHAPE)
        assert jnp.allclose(target, expected / 255.0, atol=1e-6)

    def test_probe_updates_only_the_decoder_subtree(self):
        fields = [_image_field()]
        batch = _batch(_cfg(), fields, batch_size=1, seq_len=2)
        init_rng, train_rng = 17, jax.random.PRNGKey(19)

        baseline = _agent(_cfg(decoder=False), fields, seed=init_rng)
        with_decoder = _agent(_cfg(decoder=True), fields, seed=init_rng)
        before_decoder = jax.tree.map(jnp.copy, with_decoder.params["decoder"])

        baseline_metrics = baseline.train_step(batch, train_rng)
        decoder_metrics = with_decoder.train_step(batch, train_rng)

        assert np.isfinite(decoder_metrics["loss/decoder"])
        assert decoder_metrics["opt_loss"] == pytest.approx(
            decoder_metrics["total_loss"]
            + with_decoder.cfg.scale_decoder * decoder_metrics["loss/decoder"],
            rel=1e-6,
        )
        assert _tree_any_changed(before_decoder, with_decoder.params["decoder"])
        for name in baseline.params:
            assert _tree_allclose(
                baseline.params[name], with_decoder.params[name], atol=1e-6
            ), f"decoder probe changed non-decoder subtree {name}"
        assert baseline_metrics["nan_skipped"] == decoder_metrics["nan_skipped"] == 0.0
