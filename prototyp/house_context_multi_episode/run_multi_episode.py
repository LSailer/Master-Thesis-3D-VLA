
import dataclasses
import time
from typing import cast

import jax
import jax.numpy as jnp

from src.adapters.contract import AdapterFn, transition_from_fields
from src.adapters.house_cloud_episodes import HouseCloudEpisodesAdapter
from src.buffer.replay_buffer import ReplayBuffer
from src.configs.agent_config import R2DreamerConfig
from src.environments.habitat import HabitatEnvConfig, HabitatObjectNavEnv
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.experience import Env
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor

SEED = jax.random.PRNGKey(42)

    
def prefill(
    env: Env,
    adapter_fn: AdapterFn,
    replay_buffer: ReplayBuffer,
    rng_key: jnp.ndarray,
    prefill_steps: int = 100,
) -> jnp.ndarray:
    """Prefill the replay buffer with uniformly random actions.

    Reset frames pass through the adapter (so the house cloud compacts at
    episode boundaries) but are not stored: they carry no previous action.
    The VGGT cache resets per episode (the adapter's extractor does that off
    ``frame.is_first``); the house cloud persists.
    """

    def ingest_reset() -> None:
        adapter_fn(env.reset())

    ingest_reset()
    t0 = time.perf_counter()
    for i in range(prefill_steps):
        rng_key, action_key = jax.random.split(rng_key)
        action = int(jax.random.randint(action_key, (), 0, env.num_actions))
        frame = env.step(action)
        adapted = adapter_fn(frame)
        transition = transition_from_fields(frame, adapted)
        replay_buffer.add(transition)
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  prefill {i + 1}/{prefill_steps} | "
                f"{elapsed / 50 * 1000:.0f}ms/step"
            )
            t0 = time.perf_counter()
        if frame.done:
            ingest_reset()
    return rng_key

def train(
    replay_buffer: ReplayBuffer,
    dreamer_agent: R2DreamerAgent,
    rng_key: jnp.ndarray,
    train_ratio: int = 16,
) -> jnp.ndarray:
    """Train the Dreamer agent on batches sampled from the replay buffer."""
    cfg = dreamer_agent.cfg
    for _ in range(train_ratio):
        batch = replay_buffer.sample(cfg.batch_size, cfg.seq_len)
        rng_key, train_key = jax.random.split(rng_key)
        dreamer_agent.train_step(batch, train_key, materialize=False)
    return rng_key


def main():
    config = HabitatEnvConfig()
    env = HabitatObjectNavEnv(config=config)
    feature_extractor = JAXVGGTFeatureExtractor()
    adapter_fn = HouseCloudEpisodesAdapter(feature_extractor)

    # One adapter call on the first frame supplies the encoder routing and
    # field shapes the agent needs at init — no encoder_type in the config.
    first_fields = adapter_fn(env.reset())
    encoder_mapping = {f.key: f.encoder.value for f in first_fields}
    print(f"encoder routing: {encoder_mapping}")
    replay_buffer = ReplayBuffer(100_000, env.num_actions)
    dreamer_config = R2DreamerConfig(num_actions=env.num_actions)
    rng_key, init_key = jax.random.split(SEED)
    t0 = time.perf_counter()
    dreamer_agent = R2DreamerAgent(
        config=dreamer_config,
        rng_key=init_key,
        fields=first_fields,
        encoder_overrides={"fusion_dim": 1024},
    )
    print(
        f"agent initialized in {time.perf_counter() - t0:.1f}s, "
        f"embed_size={dreamer_agent.embed_size}"
    )

    # Warmup
    print("prefilling 1000 steps...")
    t0 = time.perf_counter()
    rng_key = prefill(
        env, adapter_fn, replay_buffer, rng_key,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"prefill done in {elapsed:.1f}s ({elapsed:.3f}s/step x 1000), "
        f"buffer size {replay_buffer.size}"
    )

    # Train Dreamer module
    for step in range(1000):
        t0 = time.perf_counter()
        rng_key = train(replay_buffer, dreamer_agent, rng_key)
        elapsed = time.perf_counter() - t0
        print(
            f"train step {step + 1}/1000 done in {elapsed:.2f}s "
            f"({elapsed / 16 * 1000:.0f}ms per train_step)"
        )
        breakpoint()
        # TODO: inference() — act with the trained policy, collect new steps


if __name__ == "__main__":
    main()
