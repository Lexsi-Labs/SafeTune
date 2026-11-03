"""Audit H02: the EMA shadow is stored in the parameter dtype, so under bf16 it barely moves.

Eq. 3 of Kim et al. (arXiv:2508.12531): theta_EMA <- eta * theta_EMA + (1 - eta) * theta_t.
With decay 0.995 each step adds 0.005 * (theta_t - shadow). For a bf16 weight of size ~0.02
(spacing ~1.2e-4) any increment below ~6e-5 rounds to nothing. After a 1e-3 fine-tuning shift
the increments are ~5e-6, so almost no element of the shadow moves, and `apply_ema_weights` at
the end of training writes (nearly) the initial weights back over the fine-tuned ones.
"""
import torch

from safetune.core.optim.ema import EMAOptimizerWrapper


def test_ema_shadow_tracks_parameters_in_bf16():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_((0.02 * torch.randn(64, 64)).to(torch.bfloat16))
    ema = EMAOptimizerWrapper(lin, decay=0.995)
    w0 = lin.weight.detach().float().clone()
    with torch.no_grad():
        lin.weight.add_(1e-3)                                   # accumulated fine-tuning drift
    for _ in range(200):
        ema.step(lin)
    moved = ema.shadow_params["weight"].float() - w0
    frozen = (moved == 0).float().mean().item()                # an fp32 EMA moves every element by ~6e-4
    assert frozen < 0.10, f"{100 * frozen:.0f}% of the bf16 EMA shadow never moved"


def test_ema_fp32_behaviour_is_unchanged():
    """fp32 parameters follow exactly the previous update shadow -= (1 - decay) * (shadow - param)."""
    torch.manual_seed(1)
    lin = torch.nn.Linear(16, 16)
    ema = EMAOptimizerWrapper(lin, decay=0.9)
    ref = {n: p.detach().clone() for n, p in lin.named_parameters()}
    for step in range(20):
        with torch.no_grad():
            for p in lin.parameters():
                p.add_(0.01 * torch.randn_like(p))
        ema.step(lin)
        for n, p in lin.named_parameters():
            ref[n].sub_((1.0 - 0.9) * (ref[n] - p.detach()))
    for n in ref:
        assert torch.equal(ema.shadow_params[n], ref[n])
    ema.apply_ema_weights(lin)
    assert all(torch.equal(p.detach(), ref[n]) for n, p in lin.named_parameters())
    assert all(p.dtype == torch.float32 for p in lin.parameters())


def test_ema_apply_casts_back_to_the_parameter_dtype():
    lin = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    ema = EMAOptimizerWrapper(lin, decay=0.9)
    assert ema.shadow_params["weight"].dtype == torch.float32
    ema.apply_ema_weights(lin)
    assert lin.weight.dtype == torch.bfloat16


# --- review of #111: float64 parameters keep their precision ----------------------------------

def _steps(dtype, decay=0.9, n=5, delta=1e-9):
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 4, bias=False).to(dtype)
    ema = EMAOptimizerWrapper(lin, decay=decay)
    s0 = ema.shadow_params["weight"].clone()
    for _ in range(n):
        with torch.no_grad():
            lin.weight.add_(delta)
        ema.step(lin)
    return lin, ema, s0


def test_float64_shadow_keeps_a_1e9_change():
    lin, ema, s0 = _steps(torch.float64, n=1)
    assert ema.shadow_params["weight"].dtype == torch.float64
    moved = (ema.shadow_params["weight"] - s0).abs().max().item()
    assert abs(moved - 1e-10) < 1e-12, moved                 # (1 - decay) * 1e-9
    ema.apply_ema_weights(lin)
    assert lin.weight.dtype == torch.float64


def test_float32_matches_the_plain_dtype_update_bit_for_bit():
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 4, bias=False)
    ema = EMAOptimizerWrapper(lin, decay=0.9)
    ref = lin.weight.detach().clone()                          # main: shadow = param.clone()
    for i in range(5):
        with torch.no_grad():
            lin.weight.add_(0.01 * (i + 1))
        ema.step(lin)
        ref.sub_((1.0 - 0.9) * (ref - lin.weight.detach()))
    assert ema.shadow_params["weight"].dtype == torch.float32
    assert torch.equal(ema.shadow_params["weight"], ref)


def test_half_precision_shadow_is_still_float32():
    for dt in (torch.bfloat16, torch.float16):
        lin = torch.nn.Linear(4, 4, bias=False).to(dt)
        assert EMAOptimizerWrapper(lin).shadow_params["weight"].dtype == torch.float32
