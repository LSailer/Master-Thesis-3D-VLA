import torch
from src.action_heads import (
    DiscreteActionHead,
    DDPMActionHead,
    FlowMatchingActionHead,
    FusionModule,
    EMA,
    cosine_beta_schedule,
    compute_alpha_bar,
)

B, COND_DIM, CHUNK, ADIM = 2, 256, 4, 4


def test_discrete_head_shape():
    head = DiscreteActionHead(cond_dim=COND_DIM)
    cond = torch.randn(B, COND_DIM)
    logits = head(cond)
    assert logits.shape == (B, 6)


def test_ddpm_forward_shape():
    head = DDPMActionHead(action_dim=ADIM, cond_dim=COND_DIM, chunk_size=CHUNK)
    cond = torch.randn(B, COND_DIM)
    noisy = torch.randn(B, CHUNK, ADIM)
    t = torch.randint(0, 100, (B,))
    out = head(cond, noisy, t)
    assert out.shape == (B, CHUNK, ADIM)


def test_ddpm_sample_shape():
    head = DDPMActionHead(
        action_dim=ADIM, cond_dim=COND_DIM, chunk_size=CHUNK, n_steps=10
    )
    cond = torch.randn(B, COND_DIM)
    out = head.sample(cond)
    assert out.shape == (B, CHUNK, ADIM)


def test_fm_forward_shape():
    head = FlowMatchingActionHead(action_dim=ADIM, cond_dim=COND_DIM, chunk_size=CHUNK)
    cond = torch.randn(B, COND_DIM)
    x_0 = torch.randn(B, CHUNK, ADIM)
    x_1 = torch.randn(B, CHUNK, ADIM)
    t = torch.rand(B)
    out = head(cond, x_0, x_1, t)
    assert out.shape == (B, CHUNK, ADIM)


def test_fm_sample_shape():
    head = FlowMatchingActionHead(action_dim=ADIM, cond_dim=COND_DIM, chunk_size=CHUNK)
    cond = torch.randn(B, COND_DIM)
    out = head.sample(cond, n_steps=5)
    assert out.shape == (B, CHUNK, ADIM)


def test_heads_swappable():
    """All heads accept same cond shape."""
    cond = torch.randn(B, COND_DIM)
    d = DiscreteActionHead(cond_dim=COND_DIM)
    ddpm = DDPMActionHead(
        action_dim=ADIM, cond_dim=COND_DIM, chunk_size=CHUNK, n_steps=5
    )
    fm = FlowMatchingActionHead(action_dim=ADIM, cond_dim=COND_DIM, chunk_size=CHUNK)
    # All should accept same cond without error
    d(cond)
    ddpm.sample(cond)
    fm.sample(cond, n_steps=3)


def test_fusion_module_shape():
    fm = FusionModule()
    wp = torch.randn(B, 3, 64, 64)
    depth = torch.randn(B, 1, 64, 64)
    rel = torch.randn(B, 1, 64, 64)
    pose = torch.randn(B, 9)
    goal = torch.randn(B, 768)
    out = fm(wp, depth, rel, pose, goal)
    assert out.shape == (B, 256)


def test_ddpm_noise_schedule():
    betas = cosine_beta_schedule(100)
    alpha_bar = compute_alpha_bar(betas)
    # Monotonically decreasing
    assert torch.all(alpha_bar[1:] <= alpha_bar[:-1])
    # Boundary conditions
    assert alpha_bar[0] > 0.99
    assert alpha_bar[-1] < 0.1


def test_fm_interpolation():
    """x_t at t=0 is x_0, at t=1 is x_1."""
    x_0 = torch.randn(B, CHUNK, ADIM)
    x_1 = torch.randn(B, CHUNK, ADIM)
    t0 = torch.zeros(B)
    t1 = torch.ones(B)
    t0_exp = t0[:, None, None]
    t1_exp = t1[:, None, None]
    x_at_0 = (1 - t0_exp) * x_0 + t0_exp * x_1
    x_at_1 = (1 - t1_exp) * x_0 + t1_exp * x_1
    assert torch.allclose(x_at_0, x_0)
    assert torch.allclose(x_at_1, x_1)


def test_ddpm_overfit_toy():
    """DDPM overfits single action in <200 steps."""
    target = torch.tensor([[[1.0, 0.0, -1.0, 0.5]]])  # [1, 1, 4]
    head = DDPMActionHead(
        action_dim=4, cond_dim=8, chunk_size=1, n_steps=20, hidden_dim=64, n_layers=2
    )
    cond = torch.zeros(1, 8)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(200):
        t = torch.randint(0, 20, (1,))
        noise = torch.randn_like(target)
        ab = head.alpha_bar[t].sqrt().view(-1, 1, 1)
        ab_comp = (1 - head.alpha_bar[t]).sqrt().view(-1, 1, 1)
        noisy = ab * target + ab_comp * noise
        pred = head(cond, noisy, t)
        loss = torch.nn.functional.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1.0


def test_fm_overfit_toy():
    """FM overfits single action in <200 steps."""
    target = torch.tensor([[[1.0, 0.0, -1.0, 0.5]]])  # [1, 1, 4]
    head = FlowMatchingActionHead(
        action_dim=4, cond_dim=8, chunk_size=1, hidden_dim=64, n_layers=2
    )
    cond = torch.zeros(1, 8)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(200):
        x_0 = torch.randn_like(target)
        t = torch.rand(1)
        v_target = target - x_0
        v_pred = head(cond, x_0, target, t)
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1.0


def test_ema_tracks_weights():
    """EMA shadow moves toward model params."""
    head = DiscreteActionHead(cond_dim=8, hidden_dim=16)
    ema = EMA(head, decay=0.9)
    old_shadow = {k: v.clone() for k, v in ema.shadow.items()}
    # Change model weights
    with torch.no_grad():
        for p in head.parameters():
            p.add_(1.0)
    ema.update(head)
    # Shadow should have moved toward new weights
    for k in ema.shadow:
        diff_old = (old_shadow[k] - head.state_dict()[k]).abs().mean()
        diff_new = (ema.shadow[k] - head.state_dict()[k]).abs().mean()
        assert diff_new < diff_old
