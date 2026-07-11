"""
Diagnostic suite for round-4 methods:

  * AgentHarm/CARES/AIR/SALAD loaders (presence and signature)
  * EAP safety-circuit runner (graceful fallback when circuitkit absent)
  * Decoding-time processors (Contrastive, ProxyTuning, SafeDecoding, Nudging)
  * HARDEN SOTA (Vaccine loss, Booster projection, SaLoRA subspace, TAR outer)
  * SCRUB unlearning (TracIn influence + scrub_unlearn step)

(The ReNeLLM and CodeChameleon cases were dropped along with the legacy
``safety/attacks/`` tree — see ``verify/redteam/`` for the live stressors.)
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Eval pipeline: new loaders are registered
# ===========================================================================

def test_loader_registry_includes_new_benchmarks():
    from safetune.core.eval.pipeline import LOADERS

    for name in ("agentharm", "cares", "airbench", "saladbench"):
        assert name in LOADERS, f"{name} not registered in LOADERS"


# ===========================================================================
# Interpret: EAP runner
# ===========================================================================

def test_eap_safety_circuit_does_not_require_circuitkit():
    """EAP/EAP-IG no longer depends on the (non-existent) ``circuitkit``.

    Updated for the faithfulness fix (see ``audit_faithfulness/fix/eap.md``):
    ``eap.py`` was rewritten as a self-contained, dependency-free EAP / EAP-IG
    implementation traced to Syed et al. (arXiv:2310.10348) and hannamw/EAP-IG.
    The pre-fix code always hit a ``raise ImportError`` for a vaporware
    ``circuitkit`` package; that branch was correctly removed, so the function
    must NOT raise ImportError when ``circuitkit`` is absent. It validates its
    arguments instead.
    """
    from safetune.core.interpret import eap_safety_circuit, EAPSafetyCircuitConfig
    import sys

    # Force circuitkit absent — the rewritten module never references it.
    saved = sys.modules.pop("circuitkit", None)
    try:
        assert "circuitkit" not in sys.modules
        # Argument validation still happens (mismatched contrast list lengths)
        # — but it is a ValueError, not the removed ImportError.
        with pytest.raises(ValueError):
            eap_safety_circuit(
                "meta-llama/Llama-3.2-1B-Instruct",
                harmful_prompts=["harm 1"],
                harmless_prompts=["harmless 1", "harmless 2"],
                config=EAPSafetyCircuitConfig(),
            )
    finally:
        if saved is not None:
            sys.modules["circuitkit"] = saved


def test_eap_safety_circuit_validates_pair_lengths():
    from safetune.core.interpret import eap_safety_circuit, EAPSafetyCircuitConfig

    with pytest.raises(ValueError):
        eap_safety_circuit(
            "any",
            harmful_prompts=["a"],
            harmless_prompts=["b", "c"],
            config=EAPSafetyCircuitConfig(),
        )


# ===========================================================================
# Steer: decoding-time processors
# ===========================================================================

class _ToyGuide(nn.Module):
    """1-layer LM stub with a known logit bias toward a single token."""

    def __init__(self, vocab: int = 16, bias_token: int = 0, bias_value: float = 10.0) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab, bias=False)
        with torch.no_grad():
            # Reset the head to small random and set one column to push for bias_token.
            self.head.weight.zero_()
            self.head.weight[bias_token].fill_(bias_value)
        self.vocab = vocab
        self.bias_token = bias_token

    def forward(self, input_ids=None, inputs_embeds=None, use_cache: bool = False, **kw):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids.long())
        return type("O", (), {"logits": self.head(inputs_embeds)})


class _Tok:
    def get_vocab(self):
        return {f"<t{i}>": i for i in range(16)}


@pytest.fixture
def toy_setup():
    # Seed the global RNG: _ToyGuide.embed is randomly initialised, so without
    # a fixed seed the contrastive/proxy decoding assertions are flaky.
    torch.manual_seed(0)
    guide = _ToyGuide(vocab=16, bias_token=3, bias_value=10.0)
    tok = _Tok()
    # 1 batch, 4-token prompt
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    # Random target scores, neutral toward bias_token.
    scores = torch.zeros((1, 16))
    return guide, tok, input_ids, scores


def test_contrastive_decoding_amplifies_target_argmax(toy_setup):
    from safetune.steer.decoding import ContrastiveDecodingConfig, ContrastiveDecodingProcessor

    guide, tok, input_ids, scores = toy_setup
    # Target prefers token 5; the guide is biased toward token 3.
    scores[0, 5] = 5.0
    cd = ContrastiveDecodingProcessor(
        guide=guide, tokenizer_target=tok, tokenizer_guide=tok,
        config=ContrastiveDecodingConfig(alpha=0.5, adaptive_eps=0.0),
    )
    out = cd(input_ids, scores)
    # token 3 should be penalized (was high in guide); token 5 should remain high.
    assert out[0, 5] > out[0, 3]


def test_proxy_tuning_shifts_toward_tuned_minus_base(toy_setup):
    from safetune.steer.decoding import ProxyTuningConfig, ProxyTuningProcessor

    guide, tok, input_ids, scores = toy_setup
    base = _ToyGuide(vocab=16, bias_token=3, bias_value=0.0)  # neutral base
    pt = ProxyTuningProcessor(
        proxy_tuned=guide, proxy_base=base, tokenizer_target=tok, tokenizer_proxy=tok,
        config=ProxyTuningConfig(scale=1.0),
    )
    out = pt(input_ids, scores)
    # The delta (tuned - base) for token 3 is ~10 (tuned) - 0 (base); other tokens have ~0.
    assert out[0, 3] > out[0, 1]


def test_safedecoding_blends_with_decaying_alpha(toy_setup):
    from safetune.steer.decoding import SafeDecodingConfig, SafeDecodingProcessor

    guide, tok, input_ids, scores = toy_setup
    # Make target prefer 5, but also place 3 in its top-k so the intersection
    # with the guide (which prefers 3) is non-empty.
    scores[0, 5] = 3.0
    scores[0, 3] = 1.0
    sd = SafeDecodingProcessor(
        guide=guide, tokenizer_target=tok, prompt_length=4, tokenizer_guide=tok,
        config=SafeDecodingConfig(alpha0=1.0, decay_steps=2, top_k=8),
    )
    # Step 0 (alpha=1.0): pure guide; token 3 should dominate the blend.
    out0 = sd(input_ids, scores)
    # After one decoded token, step=1, alpha=0.5.
    decoded = torch.cat([input_ids, torch.tensor([[3]], dtype=torch.long)], dim=1)
    out1 = sd(decoded, scores)
    # Token 3 is in the intersection and at alpha=1.0 the guide dominates -> token 3 wins.
    assert not torch.isinf(out0[0, 3])
    assert out0[0, 3] >= out0[0, 5]
    # Step 1: distribution differs from step 0 (alpha decayed).
    assert not torch.allclose(out0, out1)


def test_nudging_high_entropy_uses_guide(toy_setup):
    from safetune.steer.decoding import NudgingConfig, NudgingProcessor

    guide, tok, input_ids, _ = toy_setup
    flat_scores = torch.zeros((1, 16))  # max entropy
    nd = NudgingProcessor(guide=guide, tokenizer_target=tok, tokenizer_guide=tok,
                          config=NudgingConfig(entropy_threshold=0.5, soft_blend=False))
    out = nd(input_ids, flat_scores)
    # Entropy of uniform is log(16) ~ 2.77 > 0.5, so we should pick guide; argmax is token 3.
    assert int(out[0].argmax().item()) == 3


def test_nudging_low_entropy_keeps_target(toy_setup):
    from safetune.steer.decoding import NudgingConfig, NudgingProcessor

    guide, tok, input_ids, _ = toy_setup
    sharp = torch.full((1, 16), -100.0)
    sharp[0, 7] = 50.0  # extremely sharp distribution on token 7
    nd = NudgingProcessor(guide=guide, tokenizer_target=tok, tokenizer_guide=tok,
                          config=NudgingConfig(entropy_threshold=0.5, soft_blend=False))
    out = nd(input_ids, sharp)
    # Entropy is ~0 < 0.5, so the target's token 7 should win.
    assert int(out[0].argmax().item()) == 7


# ===========================================================================
# HARDEN: Vaccine
# ===========================================================================

class _TinyAttention(nn.Module):
    """Minimal module whose class name marks it as a transformer attention
    block. Vaccine (post-fix, faithfully to git-disl/Vaccine) perturbs the
    *output hidden states of every attention layer* — not the input embedding —
    so the toy LM must route its forward pass through a real attention module.
    """

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.self_attn = _TinyAttention(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, **kw):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.long())
        hidden = self.self_attn(inputs_embeds)
        return type("O", (), {"logits": self.head(hidden)})


def _toy_task_loss(model, batch):
    if "inputs_embeds" in batch and batch["inputs_embeds"] is not None:
        out = model(inputs_embeds=batch["inputs_embeds"])
    else:
        out = model(input_ids=batch["input_ids"])
    return F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)),
                           batch["labels"].reshape(-1))


def test_vaccine_loss_returns_scalar_with_grad():
    from safetune.harden.vaccine import VaccineConfig, vaccine_loss

    model = _TinyLM(vocab=32, hidden=16)
    batch = {
        "input_ids": torch.randint(0, 32, (2, 8), dtype=torch.long),
        "labels": torch.randint(0, 32, (2, 8), dtype=torch.long),
    }
    loss = vaccine_loss(model, batch, _toy_task_loss, VaccineConfig(rho=1e-3, inner_steps=1))
    assert loss.dim() == 0 and loss.requires_grad
    loss.backward()
    # At least one trainable param should have a non-zero grad.
    found = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert found


# ===========================================================================
# HARDEN: Booster
# ===========================================================================

def test_booster_project_adds_harmful_regularizer():
    """Booster's gradient combination is *additive*, not a projection.

    Updated for the faithfulness fix (see ``audit_faithfulness/fix/booster.md``):
    the pre-fix code implemented a PCGrad/SafeGrad-style orthogonal projection
    ``g - coef*g_h``. Booster (Huang et al., ICLR 2025, arXiv:2409.01586) is not
    a projection — per ``BoosterAlignmentTrainer.training_step`` the final
    gradient is ``g_align + lambda*(g_h - g_h^perturbed)``, or, with no
    perturbed gradient supplied, the simplified ``g_align + lambda*g_h``.
    """
    from safetune.harden.booster import BoosterConfig, booster_project

    g = {"layer.weight": torch.tensor([1.0, 1.0, 0.0])}
    gh = {"layer.weight": torch.tensor([1.0, 0.0, 0.0])}
    # No perturbed-harmful grads -> simplified variant: g_align + lamb*g_h.
    projected = booster_project(g, gh, config=BoosterConfig(lamb=2.0))
    out = projected["layer.weight"]
    assert torch.allclose(out, torch.tensor([3.0, 1.0, 0.0]), atol=1e-5)


def test_booster_param_filter_skips_unmatched_names():
    from safetune.harden.booster import BoosterConfig, booster_project

    g = {"layer.weight": torch.tensor([1.0, 1.0])}
    gh = {"layer.weight": torch.tensor([1.0, 0.0])}
    cfg = BoosterConfig(param_filter=["other"])
    projected = booster_project(g, gh, cfg)
    assert torch.equal(projected["layer.weight"], g["layer.weight"])


# ===========================================================================
# HARDEN: SaLoRA
# ===========================================================================

def test_salora_subspace_has_expected_rank():
    from safetune.harden.salora import compute_safety_subspace

    base = _TinyLM(vocab=32, hidden=16)
    aligned = copy.deepcopy(base)
    with torch.no_grad():
        # Add a deterministic delta along a few directions.
        aligned.head.weight += torch.randn_like(aligned.head.weight) * 0.1
    sub = compute_safety_subspace(aligned, base, rank=4, param_filter=["head"])
    assert "head.weight" in sub
    # subspace shape is (in_dim, rank). Head in_dim=16, rank=4.
    assert sub["head.weight"].shape == (16, 4)


# ===========================================================================
# HARDEN: TAR
# ===========================================================================

def test_tar_outer_loss_returns_scalar_and_restores_params():
    from safetune.harden.tar import TARConfig, tar_outer_loss

    model = _TinyLM(vocab=32, hidden=16)
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    batch = lambda: {
        "input_ids": torch.randint(0, 32, (2, 6), dtype=torch.long),
        "labels": torch.randint(0, 32, (2, 6), dtype=torch.long),
    }
    loss = tar_outer_loss(
        model,
        retain_batch=batch(),
        harm_batch=batch(),
        safety_batch=batch(),
        task_loss_fn=_toy_task_loss,
        config=TARConfig(inner_steps=2, inner_lr=1e-3, lambda_tar=0.5),
    )
    assert loss.dim() == 0
    # Parameters must be restored exactly after the inner loop.
    for n, p in model.named_parameters():
        assert torch.equal(p, snapshot[n]), f"TAR did not restore {n}"


# ===========================================================================
# Recover.Unlearn: SCRUB + TracIn
# ===========================================================================

def test_tracin_influence_signs_make_sense():
    from safetune.core.unlearn import tracin_influence

    model = _TinyLM(vocab=16, hidden=8)
    train = [
        {"input_ids": torch.randint(0, 16, (1, 4), dtype=torch.long),
         "labels": torch.randint(0, 16, (1, 4), dtype=torch.long)}
        for _ in range(3)
    ]
    test = {"input_ids": torch.randint(0, 16, (1, 4), dtype=torch.long),
            "labels": torch.randint(0, 16, (1, 4), dtype=torch.long)}
    influences = tracin_influence(model, train, test, _toy_task_loss)
    assert len(influences) == 3
    # Influences are floats and not all zero (random init makes some non-trivial gradient).
    assert any(abs(x) > 0 for x in influences)


def test_scrub_unlearn_runs_steps_in_place():
    from safetune.core.unlearn import SCRUBConfig, scrub_unlearn

    model = _TinyLM(vocab=16, hidden=8)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    def _forward(m, batch):
        return m(input_ids=batch["input_ids"]).logits

    retain = [{"input_ids": torch.randint(0, 16, (1, 4), dtype=torch.long),
               "labels": torch.randint(0, 16, (1, 4), dtype=torch.long)} for _ in range(5)]
    forget = [{"input_ids": torch.randint(0, 16, (1, 4), dtype=torch.long),
               "labels": torch.randint(0, 16, (1, 4), dtype=torch.long)} for _ in range(5)]
    out = scrub_unlearn(model, retain, forget, forward_fn=_forward,
                        config=SCRUBConfig(max_steps=3, lr=1e-3, beta=0.1))
    assert out is model
    moved = any(not torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert moved, "scrub_unlearn did not update any parameter"
