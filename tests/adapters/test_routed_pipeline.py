"""End-to-end wiring test for every registered adapter variant.

Parametrized over ``ADAPTERS``, so a newly registered variant is covered the
moment it is added: routing -> replay -> composite encoder -> train_step -> act.
Runs on CPU against the fakes in ``conftest.py``; the real per-variant gate is a
SLURM run with Habitat and VGGT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.adapters import ADAPTERS
from src.adapters.contract import (
    Encoder,
    decoder_target_key,
    routing_from_batch,
    transition_from_fields,
)
from src.adapters.replay_image import REPLAY_IMAGE_SIZE
from src.buffer.replay_buffer import ReplayBuffer
from src.configs.config import R2DreamerConfig
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.experience import ExperienceCollector

from tests.adapters.conftest import AGG_HALF_DIM, FakeEnv, FakeExtractor

# The fakes are resolution-agnostic, so VGGT variants run at a small render size:
# the point clouds they accumulate stay cheap on CPU while the wiring is identical.
TEST_RESOLUTION = 32
STEPS = 8
BATCH, SEQ = 2, 3

# Small branches keep the CPU run in seconds; the routing under test is unchanged.
SMALL_BRANCHES = {
    "branch_embed_dim": 32,
    "pointnet_num_points": 64,
    "gnn_num_nodes": 64,
    "mlp_hidden": 32,
    "transformer_layers": 1,
    "transformer_heads": 2,
}


def _tiny_config(adapter: str, **overrides) -> R2DreamerConfig:
    return R2DreamerConfig(
        adapter=adapter,
        num_actions=4,
        deter_size=32,
        hidden_size=16,
        stoch_classes=4,
        stoch_discrete=4,
        blocks=2,
        mlp_units=16,
        twohot_bins=5,
        batch_size=BATCH,
        seq_len=SEQ,
        **overrides,
    )


def _build(name: str, extractor: FakeExtractor, **config_overrides):
    """Compose env, adapter, collector and agent exactly as ``src.main`` does."""
    adapter_cls = ADAPTERS[name]
    env = FakeEnv(resolution=TEST_RESOLUTION, episode_len=5)
    adapter = adapter_cls(extractor) if adapter_cls.NEEDS_FEATURES else adapter_cls()
    collector = ExperienceCollector(
        env=env,
        observe=adapter,
        num_actions=env.num_actions,
        buffer=ReplayBuffer(capacity=64, num_actions=env.num_actions),
    )
    fields = collector.reset_fields()
    agent = R2DreamerAgent(
        _tiny_config(name, **config_overrides),
        jax.random.PRNGKey(0),
        fields=fields,
        encoder_overrides={**dict(adapter_cls.ENCODER_OVERRIDES), **SMALL_BRANCHES},
    )
    return adapter_cls, env, collector, fields, agent


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_adapter_declares_its_pipeline(name):
    """Registry rows carry no config: the adapter declares what the run needs."""
    adapter_cls = ADAPTERS[name]
    assert isinstance(adapter_cls.RENDER_RESOLUTION, int)
    assert isinstance(adapter_cls.NEEDS_FEATURES, bool)
    assert isinstance(adapter_cls.EXTRACTOR_KWARGS, dict)
    assert isinstance(adapter_cls.ENCODER_OVERRIDES, dict)
    # Only a feature adapter may configure an extractor.
    assert adapter_cls.NEEDS_FEATURES or not adapter_cls.EXTRACTOR_KWARGS


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_routed_pipeline_trains_and_acts(name, fake_extractor):
    """Every variant survives prefill, a train step, and one acting step."""
    _cls, env, collector, fields, agent = _build(name, fake_extractor)

    step = collector.reset()
    for i in range(STEPS):
        step = collector.step(i % env.num_actions).agent_step
    assert collector.buffer_size == STEPS

    batch = collector.sample(BATCH, SEQ)
    # Routing survives the round trip through the buffer.
    assert routing_from_batch(batch) == {f.key: f.encoder for f in fields}

    agent.train_state, metrics = agent.train_step(
        agent.train_state, batch, jax.random.PRNGKey(1)
    )
    assert metrics["total_loss"] == metrics["total_loss"]  # not NaN

    action, act_state = agent.act(
        agent.params,
        step.encoder_obs,
        step.is_first,
        agent.initial_act_state(),
        jax.random.PRNGKey(2),
        True,
    )
    assert 0 <= int(action) < env.num_actions
    assert act_state.deter.shape == (1, agent.cfg.deter_size)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_adapter_declares_at_most_one_decoder_target(name, fake_extractor):
    """A variant flags at most one replayed RGB field for the decoder probe.

    The geometry-only arms observe no appearance channel, so they have nothing
    to reconstruct; the probe must refuse those runs instead of picking a field.
    """
    _cls, _env, _collector, fields, _agent = _build(name, fake_extractor)
    targets = [f for f in fields if f.decoder_target]
    assert len(targets) <= 1
    if not targets:
        with pytest.raises(ValueError):
            decoder_target_key(fields)
        return
    assert decoder_target_key(fields) == targets[0].key
    assert targets[0].buffer
    assert targets[0].encoder is Encoder.CONV
    assert targets[0].value.shape == (64, 64, 3)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_at_most_one_live_field(name, fake_extractor):
    """The buffer has a single global slot, so at most one field may be live."""
    _cls, _env, _collector, fields, _agent = _build(name, fake_extractor)
    assert sum(not f.buffer for f in fields) <= 1


def test_decoder_probe_reconstructs_the_flagged_field(fake_extractor):
    """``decoder=True`` resolves its target from the routing, not a config string."""
    _cls, env, collector, _fields, agent = _build("rgb", fake_extractor, decoder=True)
    collector.reset()
    for i in range(STEPS):
        collector.step(i % env.num_actions)
    reconstruction = agent.reconstruct(collector.sample(BATCH, SEQ))
    assert reconstruction is not None
    target, recon = reconstruction
    assert target.shape == (BATCH * SEQ, 64, 64, 3)
    assert recon.shape[0] == BATCH * SEQ


def test_feature_adapters_reset_the_extractor_per_episode(fake_extractor):
    """Boundary contract: the extractor sees the scene before the first extract."""
    _cls, env, collector, _fields, _agent = _build(
        "rgb_house_cloud_episodes", fake_extractor
    )
    assert fake_extractor.scene_resets == [env.scene_id]
    assert fake_extractor.extract_count == 1

    collector.reset()
    for i in range(env.episode_len):
        collector.step(i % env.num_actions)
    # One reset for the second episode's start, one for the auto-reset after done.
    assert fake_extractor.scene_resets == [env.scene_id] * 3


def test_cloud_branch_survives_a_degenerate_cloud():
    """A zero-extent cloud must not poison the embedding with NaN.

    Reachable in production on the first frames of a run, before any point
    clears the confidence threshold: the branch's 1e-6 scale floor is subnormal
    in float16 and flushes to zero, so the normalization has to run in float32.
    """
    from src.r2dreamer.encoders.routed_composite import (
        GnnCloudEncoder,
        PointNetCloudEncoder,
    )

    cloud = jnp.zeros((256, 6), jnp.float16)
    for branch in (
        PointNetCloudEncoder(num_points=64, embed_dim=16),
        GnnCloudEncoder(num_graph_nodes=64, embed_dim=16),
    ):
        params = branch.init(jax.random.PRNGKey(0), cloud)
        embed = branch.apply(params, cloud)
        assert not bool(jnp.isnan(jnp.asarray(embed, jnp.float32)).any())


def test_live_field_rides_along_as_global_feature(fake_extractor):
    """The live cloud is not replayed per step; the batch carries the latest one."""
    _cls, _env, _collector, fields, _agent = _build(
        "rgb_house_cloud_episodes", fake_extractor
    )
    live = [f for f in fields if not f.buffer]
    assert [f.key for f in live] == ["house_context"]

    stepped = FakeEnv()
    stepped.reset()
    transition = transition_from_fields(stepped.step(0), fields)
    assert set(transition.obs) == {f.key for f in fields if f.buffer}
    assert transition.global_feature is not None


def test_rgb_token_hybrid_replays_the_frame_and_the_pooled_tokens(fake_extractor):
    """The token twin of the pointmap hybrid: one conv field, one MLP field.

    Both are replayed per step, so the row is the ~12 KB image plus the 16 KB
    readout, and the token payload must stay the one its token-only parent
    observes - that identity is what makes the two arms comparable.
    """
    env = FakeEnv(resolution=REPLAY_IMAGE_SIZE)
    env.reset()
    # A stepped frame, not the reset one: a replay transition needs an action.
    frame = env.step(0)
    hybrid = ADAPTERS["rgb_aggregator_pooled_meanf"](fake_extractor)
    tokens_only = ADAPTERS["aggregator_pooled_meanf"](fake_extractor)

    fields = hybrid(frame)
    by_key = {f.key: f for f in fields}
    assert set(by_key) == {"agg_pooled_meanf", "image"}

    tokens = by_key["agg_pooled_meanf"]
    assert tokens.encoder is Encoder.MLP
    assert tokens.buffer and not tokens.decoder_target
    # [camera_g, patch mean_g, patch max_g, patch mean_f] over the 1024-wide half.
    assert tokens.value.shape == (4 * AGG_HALF_DIM,)
    assert tokens.value.dtype == jnp.float32

    image = by_key["image"]
    assert image.encoder is Encoder.CONV
    assert image.buffer and image.decoder_target
    assert image.value.dtype == jnp.uint8

    transition = transition_from_fields(frame, fields)
    assert transition.global_feature is None
    # The env already renders at replay size here, so the image round-trips
    # untouched: the hybrid stores the frame, not a re-encoding of it.
    np.testing.assert_array_equal(
        transition.obs["image"], np.asarray(frame.image, np.uint8)
    )
    np.testing.assert_array_equal(
        transition.obs["agg_pooled_meanf"],
        np.asarray(tokens_only(frame)[0].value),
    )
