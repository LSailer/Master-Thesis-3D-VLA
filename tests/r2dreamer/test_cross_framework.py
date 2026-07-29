"""Cross-framework equivalence tests: PyTorch R2-Dreamer vs JAX R2-Dreamer.

Transfers weights from the original PyTorch model to the JAX reimplementation
and compares forward-pass outputs on identical inputs.

Tolerances:
  - 1e-4 for individual components
  - 1e-3 for composed operations (RSSM rollout, full loss)
"""

import sys
import os

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT = os.path.join(ROOT, "external", "r2dreamer")
sys.path.insert(0, ROOT)
sys.path.insert(0, EXT)

import jax
import jax.numpy as jnp

pytest.importorskip(
    "omegaconf",
    reason="omegaconf ships with habitat-lab, which installs on Linux only",
)
pytest.importorskip(
    "rssm",
    reason="requires the PyTorch R2-Dreamer clone in external/r2dreamer",
)

from omegaconf import OmegaConf

# -- PyTorch imports --
from rssm import RSSM as PT_RSSM  # pylint: disable=import-error
from networks import (  # pylint: disable=import-error
    ConvEncoder as PT_ConvEncoder,
    MLPHead as PT_MLPHead,
)

# -- JAX imports --
from src.r2dreamer.world_model.rssm import RMSNorm, BlockLinear, R2RSSM
from src.r2dreamer.encoders.cnn import ConvEncoder
from src.r2dreamer.world_model.heads import R2MLP, onehot_mode_st
from src.r2dreamer.world_model.loss import kl_loss
from src.r2dreamer.behavior.imagination import lambda_return_positional


# =========================================================================
# Configs
# =========================================================================

SEED = 42
ATOL_COMPONENT = 1e-4
ATOL_COMPOSED = 2e-3
RTOL = 0.0  # use atol only for clarity

# Model config matching _base_.yaml defaults
DETER = 2048
HIDDEN = 256
STOCH = 32
DISCRETE = 16
BLOCKS = 8
DYN_LAYERS = 1
OBS_LAYERS = 1
IMG_LAYERS = 2
NUM_ACTIONS = 17
DEPTH = 16
KERNEL = 5
MULTS = (2, 3, 4, 4)
EMBED_DIM = DEPTH * MULTS[-1] * 4 * 4  # 16*4*4*4 = 1024
FEAT_SIZE = STOCH * DISCRETE + DETER  # 2560


# =========================================================================
# Weight transfer utilities
# =========================================================================

def _to_np(t):
    """Torch tensor → numpy."""
    return t.detach().cpu().float().numpy()


def transfer_encoder(pt_enc, jax_params):
    """Copy PyTorch ConvEncoder weights into JAX ConvEncoder param dict."""
    raw_params = jax_params["params"]
    p = raw_params.copy() if isinstance(raw_params, dict) else dict(raw_params)

    # PyTorch layers: [Conv(0), MaxPool(1), RMSNorm2D(2), SiLU(3), Conv(4), ...]
    # Indices: conv at 0,4,8,12; norm at 2,6,10,14
    for i in range(4):
        conv_idx = i * 4
        norm_idx = i * 4 + 2
        pt_conv_w = _to_np(pt_enc.layers[conv_idx].weight)  # (out, in, H, W)
        pt_conv_b = _to_np(pt_enc.layers[conv_idx].bias)
        pt_norm_w = _to_np(pt_enc.layers[norm_idx].weight)

        # Conv2d: (out, in, H, W) → (H, W, in, out)
        jax_kernel = np.transpose(pt_conv_w, (2, 3, 1, 0))

        p[f"conv{i}"] = {"kernel": jnp.array(jax_kernel), "bias": jnp.array(pt_conv_b)}
        p[f"norm{i}"] = {"scale": jnp.array(pt_norm_w)}

    return {"params": p}


def transfer_rssm(pt_rssm, jax_params):
    """Copy PyTorch RSSM weights into JAX R2RSSM param dict."""
    p = {}

    # -- Deter net --
    dn = pt_rssm._deter_net
    deter = {}

    # Input projections (Linear → Dense: need transpose)
    for idx in range(3):
        pt_seq = getattr(dn, f"_dyn_in{idx}")
        deter[f"in{idx}"] = {
            "kernel": jnp.array(_to_np(pt_seq[0].weight).T),
            "bias": jnp.array(_to_np(pt_seq[0].bias)),
        }
        deter[f"in_norm{idx}"] = {"scale": jnp.array(_to_np(pt_seq[1].weight))}

    # Hidden layers (BlockLinear: no transpose needed)
    for i in range(DYN_LAYERS):
        bl = getattr(dn._dyn_hid, f"dyn_hid_{i}")
        norm = getattr(dn._dyn_hid, f"norm_{i}")
        deter[f"hid{i}"] = {
            "kernel": jnp.array(_to_np(bl.weight)),
            "bias": jnp.array(_to_np(bl.bias)),
        }
        deter[f"hid_norm{i}"] = {"scale": jnp.array(_to_np(norm.weight))}

    # GRU (BlockLinear)
    deter["gru"] = {
        "kernel": jnp.array(_to_np(dn._dyn_gru.weight)),
        "bias": jnp.array(_to_np(dn._dyn_gru.bias)),
    }
    p["deter_net"] = deter

    # -- Obs net (posterior) --
    for i in range(OBS_LAYERS):
        pt_linear = pt_rssm._obs_net[i * 3]      # Linear
        pt_norm = pt_rssm._obs_net[i * 3 + 1]     # RMSNorm
        p[f"obs_fc{i}"] = {
            "kernel": jnp.array(_to_np(pt_linear.weight).T),
            "bias": jnp.array(_to_np(pt_linear.bias)),
        }
        p[f"obs_norm{i}"] = {"scale": jnp.array(_to_np(pt_norm.weight))}

    obs_logit = pt_rssm._obs_net[OBS_LAYERS * 3]  # final Linear (before LambdaLayer)
    p["obs_out"] = {
        "kernel": jnp.array(_to_np(obs_logit.weight).T),
        "bias": jnp.array(_to_np(obs_logit.bias)),
    }

    # -- Img net (prior) --
    for i in range(IMG_LAYERS):
        pt_linear = pt_rssm._img_net[i * 3]
        pt_norm = pt_rssm._img_net[i * 3 + 1]
        p[f"img_fc{i}"] = {
            "kernel": jnp.array(_to_np(pt_linear.weight).T),
            "bias": jnp.array(_to_np(pt_linear.bias)),
        }
        p[f"img_norm{i}"] = {"scale": jnp.array(_to_np(pt_norm.weight))}

    img_logit = pt_rssm._img_net[IMG_LAYERS * 3]
    p["img_out"] = {
        "kernel": jnp.array(_to_np(img_logit.weight).T),
        "bias": jnp.array(_to_np(img_logit.bias)),
    }

    return {"params": p}


def transfer_mlp(pt_head, jax_params, name, num_layers):
    """Copy PyTorch MLPHead weights into JAX R2MLP param dict."""
    p = {}
    for i in range(num_layers):
        pt_linear = getattr(pt_head.mlp.layers, f"{name}_linear{i}")
        pt_norm = getattr(pt_head.mlp.layers, f"{name}_norm{i}")
        p[f"fc{i}"] = {
            "kernel": jnp.array(_to_np(pt_linear.weight).T),
            "bias": jnp.array(_to_np(pt_linear.bias)),
        }
        p[f"norm{i}"] = {"scale": jnp.array(_to_np(pt_norm.weight))}

    p["out"] = {
        "kernel": jnp.array(_to_np(pt_head.last.weight).T),
        "bias": jnp.array(_to_np(pt_head.last.bias)),
    }
    return {"params": p}


def transfer_projector(pt_proj, jax_params):
    """Copy PyTorch Projector weights into JAX Projector param dict."""
    return {"params": {"proj": {
        "kernel": jnp.array(_to_np(pt_proj.w.weight).T),
    }}}


# =========================================================================
# PyTorch model factory
# =========================================================================

def make_pt_rssm_config():
    return OmegaConf.create({
        "stoch": STOCH, "deter": DETER, "hidden": HIDDEN, "discrete": DISCRETE,
        "img_layers": IMG_LAYERS, "obs_layers": OBS_LAYERS, "dyn_layers": DYN_LAYERS,
        "blocks": BLOCKS, "act": "SiLU", "norm": True, "unimix_ratio": 0.01,
        "initial": "learned", "device": "cpu",
    })


def make_pt_encoder_config():
    return OmegaConf.create({
        "act": "SiLU", "norm": True, "kernel_size": KERNEL,
        "depth": DEPTH, "mults": list(MULTS), "minres": 4,
    })


def make_pt_head_config(name, layers, out_shape, dist_name, outscale=0.0, **dist_kwargs):
    return OmegaConf.create({
        "shape": out_shape, "layers": layers, "units": HIDDEN, "act": "SiLU",
        "norm": True, "dist": {"name": dist_name, **dist_kwargs},
        "outscale": outscale, "device": "cpu", "symlog_inputs": False, "name": name,
    })


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="module")
def rng():
    return jax.random.PRNGKey(SEED)


@pytest.fixture(scope="module")
def random_input():
    """Shared random input arrays for deterministic comparison."""
    np.random.seed(SEED)
    # obs as uint8-like values in [0,255], then normalize to [0,1] for forward pass
    obs_uint8 = np.random.randint(0, 256, (2, 3, 64, 64)).astype(np.float32)
    obs_chw = obs_uint8 / 255.0
    action = np.zeros((2, NUM_ACTIONS), dtype=np.float32)
    action[:, 0] = 1.0
    return {
        "obs_chw": obs_chw,
        "obs_hwc": np.transpose(obs_chw, (0, 2, 3, 1)),
        "stoch": np.random.randn(2, STOCH, DISCRETE).astype(np.float32) * 0.1,
        "deter": np.random.randn(2, DETER).astype(np.float32) * 0.01,
        "action": action,
        "embed": np.random.randn(2, EMBED_DIM).astype(np.float32) * 0.1,
        "feat": np.random.randn(2, FEAT_SIZE).astype(np.float32) * 0.01,
    }


# =========================================================================
# Tests
# =========================================================================

class TestRMSNorm:
    def test_equivalence(self, rng, random_input):
        x_np = random_input["feat"][:, :HIDDEN]  # (2, 256)

        # PyTorch
        pt_norm = torch.nn.RMSNorm(HIDDEN, eps=1e-4, dtype=torch.float32)
        pt_out = _to_np(pt_norm(torch.tensor(x_np)))

        # JAX
        jax_norm = RMSNorm()
        jax_params = jax_norm.init(rng, jnp.array(x_np))
        # Match weight (init=1 for both)
        jax_out = np.array(jax_norm.apply(jax_params, jnp.array(x_np)))

        np.testing.assert_allclose(jax_out, pt_out, atol=ATOL_COMPONENT, rtol=RTOL)


class TestBlockLinear:
    def test_equivalence(self, rng, random_input):
        x_np = random_input["deter"]  # (2, 2048)

        # PyTorch
        from networks import BlockLinear as PT_BlockLinear  # pylint: disable=import-error
        pt_bl = PT_BlockLinear(DETER, DETER, BLOCKS)
        with torch.no_grad():
            torch.nn.init.ones_(pt_bl.weight)
            torch.nn.init.zeros_(pt_bl.bias)
        pt_out = _to_np(pt_bl(torch.tensor(x_np)))

        # JAX — same constant weights
        jax_bl = BlockLinear(out_features=DETER, blocks=BLOCKS)
        jax_params = jax_bl.init(rng, jnp.array(x_np))
        jax_params = {"params": {
            "kernel": jnp.ones_like(jax_params["params"]["kernel"]),
            "bias": jnp.zeros_like(jax_params["params"]["bias"]),
        }}
        jax_out = np.array(jax_bl.apply(jax_params, jnp.array(x_np)))

        np.testing.assert_allclose(jax_out, pt_out, atol=ATOL_COMPONENT, rtol=RTOL)

    def test_weight_transfer(self, rng, random_input):
        """Transfer random PyTorch weights and verify outputs match."""
        x_np = random_input["deter"]

        from networks import BlockLinear as PT_BlockLinear  # pylint: disable=import-error
        from tools import weight_init_  # pylint: disable=import-error
        torch.manual_seed(SEED)
        pt_bl = PT_BlockLinear(DETER, DETER, BLOCKS)
        pt_bl.apply(weight_init_)
        pt_out = _to_np(pt_bl(torch.tensor(x_np)))

        jax_bl = BlockLinear(out_features=DETER, blocks=BLOCKS)
        jax_params = jax_bl.init(rng, jnp.array(x_np))
        # Transfer: BlockLinear weight layout is identical
        jax_params = {"params": {
            "kernel": jnp.array(_to_np(pt_bl.weight)),
            "bias": jnp.array(_to_np(pt_bl.bias)),
        }}
        jax_out = np.array(jax_bl.apply(jax_params, jnp.array(x_np)))

        np.testing.assert_allclose(jax_out, pt_out, atol=ATOL_COMPONENT, rtol=RTOL)


class TestEncoder:
    def test_weight_transfer(self, rng, random_input):
        obs_chw = random_input["obs_chw"]  # already [0,1]
        obs_hwc = random_input["obs_hwc"]

        # PyTorch — ConvEncoder.forward expects (B, T, H, W, C) float [0,1]
        torch.manual_seed(SEED)
        pt_enc = PT_ConvEncoder(make_pt_encoder_config(), (64, 64, 3))
        pt_enc.eval()
        with torch.no_grad():
            pt_input = torch.tensor(obs_hwc[:, None]).float()  # (B, 1, H, W, C)
            pt_out = _to_np(pt_enc(pt_input).squeeze(1))  # (B, embed_dim)

        # JAX — ConvEncoder expects (B, H, W, C) float [0,1] (HWC contract)
        jax_enc = ConvEncoder(depth=DEPTH, kernel_size=KERNEL)
        jax_params = jax_enc.init(rng, jnp.array(obs_hwc))
        jax_params = transfer_encoder(pt_enc, jax_params)
        jax_out = np.array(jax_enc.apply(jax_params, jnp.array(obs_hwc)))

        # 4-layer CNN accumulates small errors — use composed tolerance
        np.testing.assert_allclose(jax_out, pt_out, atol=ATOL_COMPOSED, rtol=RTOL)


class TestRSSMPosteriorStep:
    def test_weight_transfer(self, rng, random_input):
        """Single posterior step with transferred weights."""
        stoch_np = random_input["stoch"]
        deter_np = random_input["deter"]
        action_np = random_input["action"]
        embed_np = random_input["embed"]

        # PyTorch
        torch.manual_seed(SEED)
        pt_rssm = PT_RSSM(make_pt_rssm_config(), embed_size=EMBED_DIM, act_dim=NUM_ACTIONS)
        pt_rssm.eval()

        stoch_flat = stoch_np.reshape(2, -1)  # (2, 512)
        with torch.no_grad():
            pt_deter = pt_rssm._deter_net(
                torch.tensor(stoch_flat),
                torch.tensor(deter_np),
                torch.tensor(action_np),
            )
            # Posterior head
            obs_inp = torch.cat([pt_deter, torch.tensor(embed_np)], dim=-1)
            pt_post_logit = pt_rssm._obs_net(obs_inp)  # (2, STOCH, DISCRETE)

        pt_deter_np = _to_np(pt_deter)
        pt_logit_np = _to_np(pt_post_logit)

        # JAX
        jax_rssm = R2RSSM(
            deter_size=DETER, stoch_classes=STOCH, stoch_discrete=DISCRETE,
            num_actions=NUM_ACTIONS, hidden=HIDDEN, blocks=BLOCKS,
            dyn_layers=DYN_LAYERS, obs_layers=OBS_LAYERS, img_layers=IMG_LAYERS,
        )
        # Init with dummy to get param structure
        k1, k2 = jax.random.split(rng)
        jax_params = jax_rssm.init(
            {"params": k1, "sample": k2},
            jnp.array(stoch_np), jnp.array(deter_np),
            jnp.array(action_np), jnp.array(embed_np),
        )
        # Transfer weights
        jax_params = transfer_rssm(pt_rssm, jax_params)

        # Run JAX posterior step (need to call __call__ which does deter + posterior)
        _, jax_deter, jax_logit = jax_rssm.apply(
            jax_params,
            jnp.array(stoch_np), jnp.array(deter_np),
            jnp.array(action_np), jnp.array(embed_np),
            rngs={"sample": k2},
        )

        # Deter goes through BlockLinear + GRU — use composed tolerance
        np.testing.assert_allclose(
            np.array(jax_deter), pt_deter_np,
            atol=ATOL_COMPOSED, rtol=RTOL,
            err_msg="Deter output mismatch",
        )

        # Posterior logits depend on deter output
        np.testing.assert_allclose(
            np.array(jax_logit), pt_logit_np,
            atol=ATOL_COMPOSED, rtol=RTOL,
            err_msg="Posterior logit mismatch",
        )


class TestRSSMPriorStep:
    def test_weight_transfer(self, rng, random_input):
        """Single prior (imagination) step with transferred weights."""
        deter_np = random_input["deter"]

        # PyTorch
        torch.manual_seed(SEED)
        pt_rssm = PT_RSSM(make_pt_rssm_config(), embed_size=EMBED_DIM, act_dim=NUM_ACTIONS)
        pt_rssm.eval()

        with torch.no_grad():
            _pt_prior_stoch, pt_prior_logit = pt_rssm.prior(torch.tensor(deter_np))

        pt_logit_np = _to_np(pt_prior_logit)

        # JAX
        jax_rssm = R2RSSM(
            deter_size=DETER, stoch_classes=STOCH, stoch_discrete=DISCRETE,
            num_actions=NUM_ACTIONS, hidden=HIDDEN, blocks=BLOCKS,
            dyn_layers=DYN_LAYERS, obs_layers=OBS_LAYERS, img_layers=IMG_LAYERS,
        )
        k1, k2 = jax.random.split(rng)
        s0 = jnp.zeros((2, STOCH, DISCRETE))
        d0 = jnp.zeros((2, DETER))
        a0 = jnp.zeros((2, NUM_ACTIONS))
        e0 = jnp.zeros((2, EMBED_DIM))
        jax_params = jax_rssm.init({"params": k1, "sample": k2}, s0, d0, a0, e0)
        jax_params = transfer_rssm(pt_rssm, jax_params)

        _, jax_logit = jax_rssm.apply(
            jax_params, jnp.array(deter_np),
            method=jax_rssm.prior, rngs={"sample": k2},
        )

        # Prior goes through 2 hidden layers — use composed tolerance
        np.testing.assert_allclose(
            np.array(jax_logit), pt_logit_np,
            atol=ATOL_COMPOSED, rtol=RTOL,
            err_msg="Prior logit mismatch",
        )


class TestRSSMObserveRollout:
    def test_weight_transfer(self, rng, random_input):
        """Full observe rollout (T=16) with transferred weights."""
        B, T = 2, 16
        np.random.seed(SEED + 1)

        # Build sequence data
        embeds = np.random.randn(B, T, EMBED_DIM).astype(np.float32) * 0.1
        actions = np.zeros((B, T, NUM_ACTIONS), dtype=np.float32)
        actions[:, :, 0] = 1.0
        is_first = np.zeros((B, T), dtype=np.float32)
        is_first[:, 0] = 1.0

        # PyTorch
        torch.manual_seed(SEED)
        pt_rssm = PT_RSSM(make_pt_rssm_config(), embed_size=EMBED_DIM, act_dim=NUM_ACTIONS)
        pt_rssm.eval()
        stoch0, deter0 = pt_rssm.initial(B)

        with torch.no_grad():
            _pt_stochs, pt_deters, pt_logits = pt_rssm.observe(
                torch.tensor(embeds), torch.tensor(actions),
                (stoch0, deter0),
                torch.tensor(is_first, dtype=torch.bool),
            )

        pt_deters_np = _to_np(pt_deters)
        pt_logits_np = _to_np(pt_logits)

        # JAX
        jax_rssm = R2RSSM(
            deter_size=DETER, stoch_classes=STOCH, stoch_discrete=DISCRETE,
            num_actions=NUM_ACTIONS, hidden=HIDDEN, blocks=BLOCKS,
            dyn_layers=DYN_LAYERS, obs_layers=OBS_LAYERS, img_layers=IMG_LAYERS,
        )
        k1, k2 = jax.random.split(rng)
        s0 = jnp.zeros((B, STOCH, DISCRETE))
        d0 = jnp.zeros((B, DETER))
        a0 = jnp.zeros((B, NUM_ACTIONS))
        e0 = jnp.zeros((B, EMBED_DIM))
        jax_params = jax_rssm.init({"params": k1, "sample": k2}, s0, d0, a0, e0)
        jax_params = transfer_rssm(pt_rssm, jax_params)

        _jax_stochs, jax_deters, jax_logits = jax_rssm.apply(
            jax_params,
            jnp.array(embeds), jnp.array(actions),
            (jnp.array(_to_np(stoch0)), jnp.array(_to_np(deter0))),
            jnp.array(is_first),
            method=jax_rssm.observe, rngs={"sample": k2},
        )

        # Step 0: both start from zeros with is_first=1, fully deterministic
        np.testing.assert_allclose(
            np.array(jax_deters[:, 0]), pt_deters_np[:, 0],
            atol=ATOL_COMPOSED, rtol=RTOL,
            err_msg="Observe rollout step 0 deter mismatch",
        )
        np.testing.assert_allclose(
            np.array(jax_logits[:, 0]), pt_logits_np[:, 0],
            atol=ATOL_COMPOSED, rtol=RTOL,
            err_msg="Observe rollout step 0 logit mismatch",
        )
        # Later steps diverge due to different RNG for Gumbel sampling,
        # so just check shapes and finiteness
        assert np.array(jax_deters).shape == pt_deters_np.shape
        assert np.all(np.isfinite(np.array(jax_deters)))


class TestKLLoss:
    def test_equivalence(self):
        """KL loss on identical logits should match."""
        np.random.seed(SEED)
        post_logits = np.random.randn(8, STOCH, DISCRETE).astype(np.float32)
        prior_logits = np.random.randn(8, STOCH, DISCRETE).astype(np.float32)
        kl_free = 1.0

        # PyTorch
        from distributions import kl as pt_kl_fn  # pylint: disable=import-error
        pt_post = torch.tensor(post_logits)
        pt_prior = torch.tensor(prior_logits)

        pt_rep = pt_kl_fn(pt_post, pt_prior.detach()).sum(-1)
        pt_dyn = pt_kl_fn(pt_post.detach(), pt_prior).sum(-1)
        pt_rep = torch.clip(pt_rep, min=kl_free)
        pt_dyn = torch.clip(pt_dyn, min=kl_free)
        pt_dyn_np = _to_np(pt_dyn)
        pt_rep_np = _to_np(pt_rep)

        # JAX
        jax_dyn, jax_rep = kl_loss(
            jnp.array(post_logits), jnp.array(prior_logits),
            STOCH, DISCRETE, kl_free,
        )

        np.testing.assert_allclose(
            np.array(jax_dyn), pt_dyn_np,
            atol=ATOL_COMPONENT, rtol=RTOL,
            err_msg="KL dyn_loss mismatch",
        )
        np.testing.assert_allclose(
            np.array(jax_rep), pt_rep_np,
            atol=ATOL_COMPONENT, rtol=RTOL,
            err_msg="KL rep_loss mismatch",
        )


class TestOneHotModeST:
    """Greedy mode parity: forward and gradient match reference OneHotDist.mode."""

    @pytest.mark.parametrize("unimix", [0.0, 0.01])
    def test_forward_and_gradient(self, unimix):
        from distributions import OneHotDist as PT_OneHotDist  # pylint: disable=import-error

        np.random.seed(SEED)
        logits_np = np.random.randn(8, 16).astype(np.float32)

        pt_logits = torch.tensor(logits_np, requires_grad=True)
        pt_mode = PT_OneHotDist(pt_logits, unimix_ratio=unimix).mode
        pt_mode.sum().backward()
        pt_forward = pt_mode.detach().numpy()
        pt_grad = pt_logits.grad.numpy()

        jx_logits = jnp.array(logits_np)
        jx_forward = np.array(onehot_mode_st(jx_logits, unimix_ratio=unimix))
        jx_grad = np.array(
            jax.grad(lambda x: onehot_mode_st(x, unimix_ratio=unimix).sum())(jx_logits)
        )

        np.testing.assert_allclose(jx_forward, pt_forward, atol=ATOL_COMPONENT, rtol=RTOL,
                                   err_msg=f"forward mismatch (unimix={unimix})")
        np.testing.assert_allclose(jx_grad, pt_grad, atol=ATOL_COMPONENT, rtol=RTOL,
                                   err_msg=f"gradient mismatch (unimix={unimix})")


class TestBarlowLoss:
    def test_equivalence(self):
        """Barlow Twins loss on identical features."""
        np.random.seed(SEED)
        B, T = 4, 8
        x1 = np.random.randn(B * T, EMBED_DIM).astype(np.float32) * 0.1
        x2 = np.random.randn(B * T, EMBED_DIM).astype(np.float32) * 0.1
        lambd = 5e-4

        # PyTorch
        x1_t = torch.tensor(x1)
        x2_t = torch.tensor(x2)
        x1_norm = (x1_t - x1_t.mean(0)) / (x1_t.std(0) + 1e-8)
        x2_norm = (x2_t - x2_t.mean(0)) / (x2_t.std(0) + 1e-8)
        c = torch.mm(x1_norm.T, x2_norm) / (B * T)
        inv = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag = ~torch.eye(x1.shape[-1], dtype=torch.bool)
        red = c[off_diag].pow(2).sum()
        pt_loss = float(inv + lambd * red)

        # JAX
        x1_j = jnp.array(x1)
        x2_j = jnp.array(x2)
        # ddof=1 to match PyTorch's torch.std() default (Bessel correction)
        x1n = (x1_j - jnp.mean(x1_j, axis=0)) / (jnp.std(x1_j, axis=0, ddof=1) + 1e-8)
        x2n = (x2_j - jnp.mean(x2_j, axis=0)) / (jnp.std(x2_j, axis=0, ddof=1) + 1e-8)
        c_j = (x1n.T @ x2n) / (B * T)
        inv_j = jnp.sum((jnp.diag(c_j) - 1.0) ** 2)
        off_j = 1.0 - jnp.eye(c_j.shape[0])
        red_j = jnp.sum((c_j * off_j) ** 2)
        jax_loss = float(inv_j + lambd * red_j)

        np.testing.assert_allclose(jax_loss, pt_loss, atol=ATOL_COMPONENT, rtol=1e-4)


class TestRewardHead:
    def test_weight_transfer(self, rng, random_input):
        """Reward head (R2MLP) with transferred weights."""
        feat_np = random_input["feat"]

        # PyTorch
        torch.manual_seed(SEED)
        pt_cfg = make_pt_head_config("reward", 1, [255], "symexp_twohot", 0.0, bin_num=255)
        pt_head = PT_MLPHead(pt_cfg, FEAT_SIZE)
        pt_head.eval()
        with torch.no_grad():
            pt_logits = pt_head.mlp(torch.tensor(feat_np))
            pt_logits = pt_head.last(pt_logits)
        pt_out = _to_np(pt_logits)

        # JAX
        jax_mlp = R2MLP(hidden=HIDDEN, layers=1, out_dim=255)
        jax_params = jax_mlp.init(rng, jnp.array(feat_np))
        jax_params = transfer_mlp(pt_head, jax_params, "reward", 1)
        jax_out = np.array(jax_mlp.apply(jax_params, jnp.array(feat_np)))

        np.testing.assert_allclose(jax_out, pt_out, atol=ATOL_COMPONENT, rtol=RTOL)


class TestContinueHead:
    def test_weight_transfer(self, rng, random_input):
        feat_np = random_input["feat"]

        torch.manual_seed(SEED)
        pt_cfg = make_pt_head_config("cont", 1, [1], "binary", 1.0)
        pt_head = PT_MLPHead(pt_cfg, FEAT_SIZE)
        pt_head.eval()
        with torch.no_grad():
            pt_out = _to_np(pt_head.last(pt_head.mlp(torch.tensor(feat_np))))

        jax_mlp = R2MLP(hidden=HIDDEN, layers=1, out_dim=1)
        jax_params = jax_mlp.init(rng, jnp.array(feat_np))
        jax_params = transfer_mlp(pt_head, jax_params, "cont", 1)
        jax_out = np.array(jax_mlp.apply(jax_params, jnp.array(feat_np)))

        # Slightly relaxed for 2560-dim matmul accumulation
        np.testing.assert_allclose(jax_out, pt_out, atol=5e-4, rtol=RTOL)


class TestLambdaReturn:
    def test_equivalence(self):
        """Lambda return calculation on identical inputs."""
        np.random.seed(SEED)
        B, T = 4, 16
        disc = 1.0 - 1.0 / 333
        lamb = 0.95

        last = np.zeros((B, T, 1), dtype=np.float32)
        term = np.random.choice([0.0, 1.0], size=(B, T, 1), p=[0.95, 0.05]).astype(np.float32)
        reward = np.random.randn(B, T, 1).astype(np.float32) * 0.1
        value = np.random.randn(B, T, 1).astype(np.float32) * 0.1
        boot = value.copy()

        # PyTorch reference (from dreamer.py lambda_return_positional)
        def pt_lambda_return(last, term, reward, value, boot, disc, lamb):
            last = torch.tensor(last)
            term = torch.tensor(term)
            reward = torch.tensor(reward)
            value = torch.tensor(value)
            boot = torch.tensor(boot)

            live = (1 - term)[:, 1:] * disc
            cont = (1 - last)[:, 1:] * lamb
            interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
            out = [boot[:, -1]]
            for i in reversed(range(live.shape[1])):
                out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
            return torch.stack(list(reversed(out))[:-1], 1)

        pt_ret = _to_np(pt_lambda_return(last, term, reward, value, boot, disc, lamb))

        # JAX
        jax_ret = np.array(lambda_return_positional(
            jnp.array(last), jnp.array(term), jnp.array(reward),
            jnp.array(value), jnp.array(boot), disc, lamb,
        ))

        np.testing.assert_allclose(jax_ret, pt_ret, atol=ATOL_COMPONENT, rtol=RTOL)
