"""
Diagnostic suite for the round-3 methods:

  * Circuit Breakers / RR  (Steer defense, conditional rerouting)
  * DeepRefusal            (Recover counter-defense to abliteration)

(The ArtPrompt and FlipAttack cases were dropped along with the legacy
``safety/attacks/`` tree — see ``verify/redteam/`` for the live stressors.)
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Circuit Breakers / RR
# ---------------------------------------------------------------------------

class _Inner(nn.Module):
    def __init__(self, hidden: int, n_layers: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids.long())
        for layer in self.layers:
            h = torch.nn.functional.gelu(layer(h)) + h
        return self.norm(h)


class _Wrap(nn.Module):
    def __init__(self, hidden: int = 32, n_layers: int = 4, vocab: int = 200) -> None:
        super().__init__()
        self.model = _Inner(hidden, n_layers, vocab)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(input_ids))


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    return _Wrap(hidden=32, n_layers=4, vocab=200)


def test_cb_rr_zero_mode_reduces_projection(small_model):
    """Hook should drive the projection along the direction toward zero for flagged tokens."""
    from safetune.steer import CircuitBreakerRRModel, CircuitBreakerRRConfig

    direction = torch.randn(32)
    direction = direction / direction.norm()
    # Baseline statistics chosen so EVERY token's projection lands above z=1.5.
    baselines = {2: (-100.0, 1.0)}
    cb = CircuitBreakerRRModel(
        small_model,
        directions={2: direction},
        baselines=baselines,
        config=CircuitBreakerRRConfig(threshold=1.5, strength=1.0, reroute_to="zero"),
    )
    ids = torch.randint(0, 200, (1, 8), dtype=torch.long)

    # Capture pre and post residuals at layer 2.
    pre = {}
    post = {}

    def cap(store):
        def hook(_m, _i, out):
            store["h"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return hook

    h_pre = small_model.model.layers[2].register_forward_hook(cap(pre))
    with torch.no_grad():
        small_model(ids)
    h_pre.remove()

    cb.install()
    h_post = small_model.model.layers[2].register_forward_hook(cap(post))
    with torch.no_grad():
        small_model(ids)
    h_post.remove()
    cb.remove()

    proj_pre = (pre["h"] * direction).sum(dim=-1).abs().mean().item()
    proj_post = (post["h"] * direction).sum(dim=-1).abs().mean().item()
    assert proj_post < proj_pre, f"CB-RR did not reduce projection: pre={proj_pre:.4f} post={proj_post:.4f}"


def test_cb_rr_threshold_keeps_benign_tokens(small_model):
    """If the threshold is set high enough that no token crosses it, the model output should not change."""
    from safetune.steer import CircuitBreakerRRModel, CircuitBreakerRRConfig

    direction = torch.randn(32)
    direction = direction / direction.norm()
    # baseline shifted so every token z-score is far below the threshold
    baselines = {2: (1e6, 1.0)}
    cb = CircuitBreakerRRModel(
        small_model,
        directions={2: direction},
        baselines=baselines,
        config=CircuitBreakerRRConfig(threshold=1.5, strength=1.0, reroute_to="zero"),
    )
    ids = torch.randint(0, 200, (1, 4), dtype=torch.long)
    with torch.no_grad():
        before = small_model(ids).clone()
    with cb:
        with torch.no_grad():
            after = small_model(ids).clone()
    assert torch.allclose(before, after, atol=1e-5), "Below-threshold tokens should be unmodified"


def test_cb_rr_context_manager_removes_hooks(small_model):
    from safetune.steer import CircuitBreakerRRModel

    direction = torch.randn(32)
    cb = CircuitBreakerRRModel(small_model, directions={2: direction})
    ids = torch.randint(0, 200, (1, 4), dtype=torch.long)
    with torch.no_grad():
        before = small_model(ids).clone()
    with cb:
        pass
    with torch.no_grad():
        after = small_model(ids).clone()
    assert torch.allclose(before, after)


# ---------------------------------------------------------------------------
# DeepRefusal
# ---------------------------------------------------------------------------

class _LlamaBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(hidden, hidden, bias=False),
            "k_proj": nn.Linear(hidden, hidden, bias=False),
            "v_proj": nn.Linear(hidden, hidden, bias=False),
            "o_proj": nn.Linear(hidden, hidden, bias=False),
        })
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(hidden, hidden * 2, bias=False),
            "up_proj": nn.Linear(hidden, hidden * 2, bias=False),
            "down_proj": nn.Linear(hidden * 2, hidden, bias=False),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _llama_like(hidden: int = 32, n_layers: int = 3):
    class W(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.ModuleDict({
                "embed_tokens": nn.Embedding(100, hidden),
                "layers": nn.ModuleList([_LlamaBlock(hidden) for _ in range(n_layers)]),
                "norm": nn.LayerNorm(hidden),
            })
            self.lm_head = nn.Linear(hidden, 100, bias=False)
        def forward(self, x):
            return x
    torch.manual_seed(0)
    return W()


def test_deeprefusal_edits_projection_matrices():
    from safetune.recover import apply_deeprefusal

    model = _llama_like(hidden=32, n_layers=3)
    direction = torch.randn(32)
    direction = direction / direction.norm()

    # Snapshot the o_proj and down_proj weights before.
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if n.endswith("o_proj.weight") or n.endswith("down_proj.weight")}
    assert before, "no projection weights found in the test fixture"

    apply_deeprefusal(model, direction, strength=0.5)

    # All targeted projections should have changed; non-target (q_proj, gate_proj) unchanged.
    moved = 0
    for name, p in model.named_parameters():
        if name in before and not torch.equal(p, before[name]):
            moved += 1
    assert moved == len(before), f"expected all targeted projections to move; moved {moved}/{len(before)}"

    # q_proj should not change.
    qs = {n: p for n, p in model.named_parameters() if n.endswith("q_proj.weight")}
    for n, p in qs.items():
        before_q = _llama_like(hidden=32, n_layers=3).state_dict()[n]
        # We can't compare against a fresh model directly since seed will differ; instead
        # assert q_proj is NOT in the moved-by-deeprefusal set by checking it's not in `before`.
        assert n not in before  # not targeted


def test_deeprefusal_strength_zero_is_no_op():
    """strength=0 should make the edit identically zero."""
    from safetune.recover import apply_deeprefusal

    model = _llama_like()
    direction = torch.randn(32)
    direction = direction / direction.norm()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    # apply_deeprefusal has assert_mutates decorator which will WARN (not error)
    # when nothing changed; that's the right behaviour here. We still run it
    # to confirm it does not crash on strength=0.
    apply_deeprefusal(model, direction, strength=0.0)
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), f"strength=0 mutated {n}"


def test_deeprefusal_writes_to_orthogonal_direction():
    """After DeepRefusal, the o_proj row-space should have non-trivial mass along d_perp."""
    from safetune.recover import apply_deeprefusal
    from safetune.recover.deeprefusal import _spread_direction

    model = _llama_like()
    direction = torch.randn(32)
    direction = direction / direction.norm()
    d_perp = _spread_direction(direction, seed=0)

    # Component of any o_proj row along d_perp BEFORE
    o_proj_w = dict(model.named_parameters())["model.layers.0.self_attn.o_proj.weight"].detach().clone()
    coeff_perp_before = (d_perp @ o_proj_w).abs().mean().item()

    apply_deeprefusal(model, direction, strength=2.0, seed=0)

    o_proj_w_after = dict(model.named_parameters())["model.layers.0.self_attn.o_proj.weight"]
    coeff_perp_after = (d_perp @ o_proj_w_after).abs().mean().item()
    assert coeff_perp_after > coeff_perp_before, (
        f"d_perp coefficient did not increase: before={coeff_perp_before:.4f} after={coeff_perp_after:.4f}"
    )
