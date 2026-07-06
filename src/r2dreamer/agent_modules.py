"""Flax module construction + parameter-bundle init for R2DreamerAgent.

Everything in ``R2DreamerAgent.__init__`` that instantiates Flax modules and
runs their ``.init`` to produce the flat ``params`` pytree lives here as
:class:`AgentModules`. The agent stays a thin orchestrator: it builds one
``AgentModules`` from config + an rng key, then wires it into training state.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, cast

import jax
import jax.numpy as jnp

from src.configs.config import R2DreamerConfig
from src.r2dreamer.encoders.decoder import ConvDecoder
from src.r2dreamer.encoders.registry import (
    RGB_BEARING_ENCODER_TYPES,
    EncoderModule,
    dummy_encoder_obs,
    make_encoder_module,
)
from src.r2dreamer.representation.barlow import Projector
from src.r2dreamer.world_model.heads import R2MLP
from src.r2dreamer.world_model.rssm import R2RSSM


def make_rssm_module(config: R2DreamerConfig) -> R2RSSM:
    """Build the RSSM module from config.

    Args:
      config: Effective agent config.

    Returns:
      A constructed, uninitialized ``R2RSSM`` module.
    """
    return R2RSSM(
        deter_size=config.deter_size,
        stoch_classes=config.stoch_classes,
        stoch_discrete=config.stoch_discrete,
        num_actions=config.num_actions,
        hidden=config.hidden_size,
        blocks=config.blocks,
        dyn_layers=config.dyn_layers,
        obs_layers=config.obs_layers,
        img_layers=config.img_layers,
        unimix_ratio=config.unimix_ratio,
    )


class AgentModules(NamedTuple):
    """Constructed Flax modules plus their initialized parameter pytree.

    Attributes:
      encoder_mod: The Dreamer Encoder Module (resolved via the encoder
        registry from ``config.encoder_type``/``encoder_module_cls``).
      rssm_mod: The recurrent state-space model.
      proj_mod: The Barlow-twins projector (``feat_size -> embed_size``).
      reward_mod: Reward head MLP.
      cont_mod: Continue-flag head MLP.
      actor_mod: Actor head MLP.
      critic_mod: Critic head MLP.
      decoder_mod: Optional debug RGB-reconstruction decoder (``None``
        unless ``config.decoder``).
      params: Flat pytree of all module parameters, keyed by module name
        (``"encoder"``, ``"rssm"``, ``"projector"``, ``"reward"``,
        ``"cont"``, ``"actor"``, ``"critic"``, and ``"decoder"`` when
        ``config.decoder``).
      embed_size: Encoder output width, discovered via a dummy forward pass.
      modules: Same modules as a name-keyed dict, for sub-loss functions that
        expect a ``Mapping[str, nn.Module]`` bundle.
    """

    encoder_mod: EncoderModule
    rssm_mod: R2RSSM
    proj_mod: Projector
    reward_mod: R2MLP
    cont_mod: R2MLP
    actor_mod: R2MLP
    critic_mod: R2MLP
    decoder_mod: ConvDecoder | None
    params: Dict[str, Any]
    embed_size: int
    modules: Dict[str, Any]


def build_agent_modules(
    config: R2DreamerConfig, rng_key: jnp.ndarray
) -> AgentModules:
    """Construct all Flax modules and initialize their parameters.

    Mirrors the previous inline ``R2DreamerAgent.__init__`` body: builds the
    encoder (registry-resolved), discovers ``embed_size`` via a dummy forward
    pass, then builds the RSSM, Barlow projector, and the four MLP heads
    (reward/continue/actor/critic), optionally adding the debug decoder probe
    when ``config.decoder`` is set.

    Args:
      config: Effective agent config.
      rng_key: PRNG key consumed for all module inits.

    Returns:
      The constructed :class:`AgentModules` bundle.

    Raises:
      ValueError: If ``config.decoder`` is set but ``config.encoder_type``
        carries no RGB modality to reconstruct.
    """
    encoder_mod = make_encoder_module(config, direct=True)

    # Dummy forward to discover embed_size.
    rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
    dummy_obs = dummy_encoder_obs(config, type(encoder_mod))
    enc_params = encoder_mod.init(k1, dummy_obs)
    embed = cast(jnp.ndarray, encoder_mod.apply(enc_params, dummy_obs))
    embed_size = embed.shape[-1]

    # RSSM
    rssm_mod = make_rssm_module(config)
    stoch0 = jnp.zeros((1, config.stoch_classes, config.stoch_discrete))
    deter0 = jnp.zeros((1, config.deter_size))
    action0 = jnp.zeros((1, config.num_actions))
    embed0 = jnp.zeros((1, embed_size))
    rng_key, k_sample = jax.random.split(rng_key)
    rssm_params = rssm_mod.init(
        {"params": k2, "sample": k_sample}, stoch0, deter0, action0, embed0
    )

    # Projector: feat_size -> embed_size
    proj_mod = Projector(out_dim=embed_size)
    feat0 = jnp.zeros((1, config.feat_size))
    proj_params = proj_mod.init(k3, feat0)

    # MLP heads (outscale matches PyTorch: 0.0 for reward/critic, 0.01 for actor)
    rng_key, k_rew, k_con, k_act, k_cri = jax.random.split(rng_key, 5)
    reward_mod = R2MLP(
        hidden=config.mlp_units,
        layers=config.mlp_layers_reward,
        out_dim=config.twohot_bins,
        outscale=0.0,
    )
    rew_params = reward_mod.init(k_rew, feat0)

    cont_mod = R2MLP(
        hidden=config.mlp_units,
        layers=config.mlp_layers_cont,
        out_dim=1,
    )
    con_params = cont_mod.init(k_con, feat0)

    actor_mod = R2MLP(
        hidden=config.mlp_units,
        layers=config.mlp_layers_actor,
        out_dim=config.num_actions,
        outscale=0.01,
    )
    act_params = actor_mod.init(k_act, feat0)

    critic_mod = R2MLP(
        hidden=config.mlp_units,
        layers=config.mlp_layers_critic,
        out_dim=config.twohot_bins,
        outscale=0.0,
    )
    cri_params = critic_mod.init(k_cri, feat0)

    # ---- Debug decoder probe (3D-51): built ONLY when config.decoder ----
    # Reconstructs RGB from stop-gradient `feat` for visual verification.
    # Left unbuilt by default so the params pytree (and thus checkpoints) of
    # CNN/VGGT runs is unchanged.
    decoder_mod = None
    dec_params = None
    if config.decoder:
        if config.encoder_type not in RGB_BEARING_ENCODER_TYPES:
            raise ValueError(
                "decoder=True requires an RGB-bearing encoder_type — the "
                "ConvDecoder reconstructs an RGB image, but "
                f"{config.encoder_type!r} carries no RGB modality to reconstruct."
            )
        rng_key, k_dec = jax.random.split(rng_key)
        decoder_mod = ConvDecoder(
            depth=config.encoder_depth,
            kernel_size=config.encoder_kernel,
            mults=config.encoder_mults,
        )
        dec_params = decoder_mod.init(k_dec, feat0)

    # ---- Bundle all params ----
    params: Dict[str, Any] = {
        "encoder": enc_params,
        "rssm": rssm_params,
        "projector": proj_params,
        "reward": rew_params,
        "cont": con_params,
        "actor": act_params,
        "critic": cri_params,
    }

    modules: Dict[str, Any] = {
        "encoder": encoder_mod,
        "rssm": rssm_mod,
        "projector": proj_mod,
        "reward": reward_mod,
        "cont": cont_mod,
        "actor": actor_mod,
        "critic": critic_mod,
    }

    if config.decoder:
        params["decoder"] = dec_params
        modules["decoder"] = decoder_mod

    return AgentModules(
        encoder_mod=encoder_mod,
        rssm_mod=rssm_mod,
        proj_mod=proj_mod,
        reward_mod=reward_mod,
        cont_mod=cont_mod,
        actor_mod=actor_mod,
        critic_mod=critic_mod,
        decoder_mod=decoder_mod,
        params=params,
        embed_size=embed_size,
        modules=modules,
    )
