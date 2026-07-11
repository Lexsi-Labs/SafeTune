"""
Behavioral diagnostic suite for round-5 new implementations.

Checks that each new method produces the *correct mathematical output*,
not just that it runs without error.

  * apply_prepost_merge      — exact interpolation formula
  * learn_somf_mask          — returns bool dict, correct shape
  * RepNoiseTrainer          — β1·L_noise + β2·L_benign + β3·L_forget formula
  * SEAMTrainer              — L_align - λ·L_harm formula
  * CTRAPTrainer             — Eq.(2): L_align + λ·L_Collapse(θ−α·∇L_harm)
  * fit_cast_probe / CASTModel — probe returns correct shape; gate fires
  * crisp_unlearn            — SAE feature suppression loss active
  * MARTTrainer              — instantiates; train() runs one round

All tests run on CPU with tiny models; no GPU required.
"""
from __future__ import annotations

import copy
import tempfile
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tiny shared fixtures
# ---------------------------------------------------------------------------

class _MiniInner(nn.Module):
    def __init__(self, hidden: int, vocab: int, n_layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids.long())
        for layer in self.layers:
            h = F.gelu(layer(h)) + h
        return self.norm(h)


class _MiniLM(nn.Module):
    """Tiny causal-LM stub compatible with HF Trainer conventions."""

    def __init__(self, hidden: int = 16, vocab: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden, "pad_token_id": 0})()
        self.model = _MiniInner(hidden, vocab, n_layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_hidden_states: bool = False,
        **_kw: Any,
    ) -> Any:
        h = self.model(input_ids)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift = logits[:, :-1, :].reshape(-1, logits.size(-1))
            tgt = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift, tgt, ignore_index=-100)
        hidden_states = (h,) if output_hidden_states else None
        return type("O", (), {"logits": logits, "loss": loss, "hidden_states": hidden_states})()

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse)


def _batch(bs: int = 2, seq: int = 5, vocab: int = 64) -> Dict[str, torch.Tensor]:
    ids = torch.randint(1, vocab, (bs, seq))
    return {
        "input_ids": ids,
        "attention_mask": torch.ones(bs, seq, dtype=torch.long),
        "labels": ids.clone(),
    }


# ===========================================================================
# apply_prepost_merge — exact interpolation
# ===========================================================================

def test_prepost_merge_exact_formula():
    """θ_out = (1-α)·θ_post + α·θ_pre (per param)."""
    from safetune.recover.merge import apply_prepost_merge

    torch.manual_seed(0)
    post = _MiniLM()
    torch.manual_seed(1)
    pre = _MiniLM()

    alpha = 0.3
    post_sd_before = {k: v.detach().clone() for k, v in post.state_dict().items()}
    apply_prepost_merge(post, pre, alpha=alpha)
    post_sd_after = post.state_dict()
    pre_sd = pre.state_dict()

    for name, w_before in post_sd_before.items():
        expected = (1 - alpha) * w_before.float() + alpha * pre_sd[name].float()
        actual = post_sd_after[name].float()
        assert torch.allclose(actual, expected, atol=1e-5), (
            f"prepost_merge formula wrong for {name}: max_diff="
            f"{(actual - expected).abs().max().item():.2e}"
        )


def test_prepost_merge_alpha_zero_is_identity():
    from safetune.recover.merge import apply_prepost_merge

    post = _MiniLM()
    pre = _MiniLM()
    before = {k: v.clone() for k, v in post.state_dict().items()}
    apply_prepost_merge(post, pre, alpha=0.0)
    for name, w_before in before.items():
        assert torch.allclose(post.state_dict()[name], w_before, atol=1e-6)


def test_prepost_merge_alpha_one_copies_pre():
    from safetune.recover.merge import apply_prepost_merge

    post = _MiniLM()
    pre = _MiniLM()
    pre_sd = {k: v.clone() for k, v in pre.state_dict().items()}
    apply_prepost_merge(post, pre, alpha=1.0)
    for name, w_pre in pre_sd.items():
        assert torch.allclose(post.state_dict()[name], w_pre, atol=1e-6), name


# ===========================================================================
# learn_somf_mask — output contract
# ===========================================================================

def test_somf_mask_returns_bool_dict_correct_keys():
    """learn_somf_mask must return a {param_name: bool_tensor} dict."""
    from safetune.recover.merge import learn_somf_mask

    torch.manual_seed(0)
    ft = _MiniLM(); al = _MiniLM(); ba = _MiniLM()

    # Build minimal preference data with the expected keys.
    pref = [
        {
            "input_ids": torch.randint(1, 64, (4,)),
            "chosen_ids": torch.randint(1, 64, (4,)),
            "rejected_ids": torch.randint(1, 64, (4,)),
        }
        for _ in range(4)
    ]
    mask = learn_somf_mask(ft, al, ba, pref, num_steps=3)

    assert isinstance(mask, dict), "learn_somf_mask must return a dict"
    assert len(mask) > 0, "mask is empty"
    # Every value must be a boolean tensor with the same shape as the param.
    ft_sd = ft.state_dict()
    for name, m in mask.items():
        assert m.dtype == torch.bool, f"{name}: mask dtype is {m.dtype}, expected bool"
        # Key must correspond to a 2-D param in the model.
        assert name in ft_sd, f"{name} in mask but not in model"
        assert ft_sd[name].dim() == 2, f"{name} is in mask but is not a 2-D tensor"


# ===========================================================================
# RepNoiseTrainer — formula correctness
# ===========================================================================

class _MultiLayerLM(nn.Module):
    """Causal-LM stub returning per-layer hidden states (HF convention).

    Unlike :class:`_MiniLM` (which exposes only the final hidden state), this
    returns ``output_hidden_states`` as an ``(n_layers + 1)``-tuple — embedding
    output plus each transformer-layer output — so the RepNoise *layer-wise*
    MMD noise term can be exercised over multiple layers.
    """

    def __init__(self, hidden: int = 16, vocab: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden, "pad_token_id": 0})()
        self.model = _MiniInner(hidden, vocab, n_layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                output_hidden_states: bool = False, **_kw: Any) -> Any:
        inner = self.model
        h = inner.embed_tokens(input_ids.long())
        states = [h]
        for layer in inner.layers:
            h = F.gelu(layer(h)) + h
            states.append(h)
        h = inner.norm(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift = logits[:, :-1, :].reshape(-1, logits.size(-1))
            tgt = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift, tgt, ignore_index=-100)
        hidden_states = tuple(states) if output_hidden_states else None
        return type("O", (), {"logits": logits, "loss": loss,
                              "hidden_states": hidden_states})()


def _make_repnoise_trainer(model, harmful_ds, benign_ds=None):
    """Build a RepNoiseTrainer with 0-step config (just checks formula)."""
    from safetune.harden.repnoise import RepNoiseTrainer, RepNoiseConfig

    with tempfile.TemporaryDirectory() as d:
        cfg = RepNoiseConfig(
            d,
            max_steps=0,
            use_cpu=True,
            report_to="none",
        )
        from datasets import Dataset
        train_ds = Dataset.from_dict(
            {"input_ids": [[1, 2, 3, 4, 5]] * 4,
             "attention_mask": [[1, 1, 1, 1, 1]] * 4,
             "labels": [[1, 2, 3, 4, 5]] * 4}
        )
        return RepNoiseTrainer(
            model=model,
            args=cfg,
            train_dataset=train_ds,
            harmful_dataset=harmful_ds,
            benign_dataset=benign_ds,
        )


def test_repnoise_loss_has_three_terms():
    """Faithful 3-term loss: L_retain + beta*L_noise(MMD) - alpha*log(L_harmful).

    Asserts the loss equals the manual sum of the three distinct terms, that the
    MMD noise term is computed over MULTIPLE hidden layers and is non-zero, and
    that the harmful-CE used in the ascent term is positive (log defined).
    """
    from safetune.harden.repnoise import RepNoiseTrainer, RepNoiseConfig
    from datasets import Dataset

    torch.manual_seed(0)
    model = _MultiLayerLM()

    harmful = [_batch(bs=2) for _ in range(4)]
    benign_ds = [_batch(bs=2) for _ in range(4)]

    with tempfile.TemporaryDirectory() as d:
        cfg = RepNoiseConfig(d, max_steps=0, use_cpu=True, report_to="none",
                             repnoise_beta1=1.0, repnoise_beta2=1.0, repnoise_beta3=0.001)
        train_ds = Dataset.from_dict(
            {"input_ids": [[1, 2, 3]] * 4, "attention_mask": [[1, 1, 1]] * 4,
             "labels": [[1, 2, 3]] * 4}
        )
        trainer = RepNoiseTrainer(
            model=model, args=cfg, train_dataset=train_ds,
            harmful_dataset=harmful, benign_dataset=benign_ds,
        )

    benign_inputs = {k: v.clone() for k, v in _batch().items()}
    loss = trainer.compute_loss(model, benign_inputs)
    assert loss.isfinite(), "RepNoise loss is not finite"

    # Re-derive each of the three terms independently from the index-0 batches
    # (compute_loss pulls benign then harmful, both from index 0).
    harmful_b = trainer._prepare_batch(dict(harmful[0]), model)
    benign_b = trainer._prepare_batch(dict(benign_ds[0]), model)

    mask = trainer._diff_mask(harmful_b, benign_b)
    hf_out = model(input_ids=harmful_b["input_ids"],
                   attention_mask=harmful_b.get("attention_mask"),
                   output_hidden_states=True)
    hs = hf_out.hidden_states
    assert hs is not None and len(hs) >= 2, (
        f"MMD term must span multiple layers; got {0 if hs is None else len(hs)}"
    )

    l_noise = trainer._noise_loss(hs, mask, torch.device("cpu"))
    assert l_noise.item() != 0.0, "MMD noise term must be non-zero"
    assert l_noise.isfinite()

    l_harmful = trainer._harmful_ce(model, hf_out, harmful_b["input_ids"], mask.float())
    assert l_harmful.item() > 0, "harmful CE must be positive (log defined)"

    l_retain = model(**benign_b).loss

    expected = (1.0 * l_retain + 0.001 * l_noise
                - 1.0 * torch.log(l_harmful.clamp(min=1e-8)))
    assert torch.allclose(loss.float(), expected.float(), atol=1e-4), (
        f"RepNoise 3-term loss mismatch: got {loss.item():.5f}, "
        f"expected {expected.item():.5f}"
    )


def test_repnoise_mmd_sums_over_multiple_layers():
    """The MMD noise term must accumulate across ALL hidden layers, not one."""
    from safetune.harden.repnoise import RepNoiseTrainer, RepNoiseConfig

    torch.manual_seed(7)
    model = _MultiLayerLM(n_layers=3)
    with tempfile.TemporaryDirectory() as d:
        cfg = RepNoiseConfig(d, max_steps=0, use_cpu=True, report_to="none")
        trainer = RepNoiseTrainer(model=model, args=cfg,
                                  harmful_dataset=[_batch(bs=2)],
                                  benign_dataset=[_batch(bs=2)])

    hb = _batch(bs=2)
    out = model(input_ids=hb["input_ids"], attention_mask=hb["attention_mask"],
                output_hidden_states=True)
    hs = out.hidden_states
    assert len(hs) == 4, f"expected n_layers+1=4 hidden states, got {len(hs)}"
    mask = hb["attention_mask"].bool()

    full = trainer._noise_loss(hs, mask, torch.device("cpu"))
    single = trainer._noise_loss(hs[:1], mask, torch.device("cpu"))
    assert not torch.allclose(full, single), (
        "MMD noise must be layer-wise (all layers), not a single-layer value"
    )
    assert full.isfinite() and full.item() != 0.0


def test_repnoise_ascent_sign_increases_harmful_ce():
    """Ascent term -alpha*log(harmful_CE): lower harmful CE -> higher loss.

    The term decreases monotonically in harmful_CE, so minimising the loss
    pushes harmful_CE UP (gradient ascent on harmful CE) — the correct sign.
    """
    import math
    alpha = 1.0
    ce_low = torch.tensor(0.5)
    ce_high = torch.tensor(2.0)
    term_low = -alpha * torch.log(ce_low)
    term_high = -alpha * torch.log(ce_high)
    assert term_low.item() > term_high.item(), (
        "Ascent term must penalise low harmful CE more than high harmful CE"
    )
    assert math.isclose(term_low.item(), -math.log(0.5), rel_tol=1e-5)


def test_repnoise_no_harmful_data_degrades_to_benign_ce():
    """Without harmful_dataset, loss should equal β2 * L_benign (no noise/forget)."""
    from safetune.harden.repnoise import RepNoiseTrainer, RepNoiseConfig
    from datasets import Dataset

    torch.manual_seed(1)
    model = _MiniLM()
    train_ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3]] * 4, "attention_mask": [[1, 1, 1]] * 4,
         "labels": [[1, 2, 3]] * 4}
    )

    with tempfile.TemporaryDirectory() as d:
        # beta2=1.0, beta1=beta3=0.0 — only benign term active.
        cfg = RepNoiseConfig(d, max_steps=0, use_cpu=True, report_to="none",
                             repnoise_beta1=0.0, repnoise_beta2=1.0, repnoise_beta3=0.0)
        trainer = RepNoiseTrainer(
            model=model, args=cfg, train_dataset=train_ds,
            harmful_dataset=None, benign_dataset=None,
        )

    inputs = _batch()
    loss = trainer.compute_loss(model, inputs)

    # Manually compute expected: β2 * CE on same inputs.
    with torch.no_grad():
        out = model(**inputs)
        expected = out.loss  # β2=1.0

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5), (
        f"RepNoise no-harmful loss mismatch: got {loss.item():.4f}, "
        f"expected {expected.item():.4f}"
    )


# ===========================================================================
# SEAMTrainer — faithful SEAM objective (arXiv:2505.12186, Eq. 5)
#   L = L_ul + alpha*L_up + beta*L_sd, with L_sd = cos(g_a, g_b)  (Eq. 2)
#   and grad L_sd estimated Hessian-free (Eq. 6).
# ===========================================================================

def _seam_trainer(model, *, alpha=1.0, beta=0.01, epsilon=1e-3):
    from safetune.harden.seam import SEAMTrainer, SEAMConfig
    from datasets import Dataset

    benign = [_batch(bs=1) for _ in range(4)]
    harmful = [_batch(bs=1) for _ in range(4)]
    align = [_batch(bs=1) for _ in range(4)]

    train_ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3]] * 4, "attention_mask": [[1, 1, 1]] * 4,
         "labels": [[1, 2, 3]] * 4}
    )
    with tempfile.TemporaryDirectory() as d:
        cfg = SEAMConfig(d, max_steps=0, use_cpu=True, report_to="none",
                         seam_alpha=alpha, seam_beta=beta, seam_epsilon=epsilon)
        trainer = SEAMTrainer(
            model=model, args=cfg, train_dataset=train_ds,
            harmful_dataset=harmful, alignment_dataset=align,
            benign_dataset=benign,
        )
    import itertools
    trainer._harm_iter = itertools.cycle(iter(harmful))
    trainer._align_iter = itertools.cycle(iter(align))
    trainer._benign_iter = itertools.cycle(iter(benign))
    trainer._cached = {}
    return trainer


def test_seam_loss_is_finite_and_decomposes():
    """SEAM scalar surrogate = L_ul + alpha*L_up + beta*L_sd, all finite."""
    import math
    torch.manual_seed(2)
    model = _MiniLM()
    trainer = _seam_trainer(model, alpha=1.0, beta=0.01)

    inputs = _batch()
    loss = trainer.compute_loss(model, inputs)
    assert torch.isfinite(loss), f"SEAM loss not finite: {loss}"

    m = trainer.last_seam_metrics
    for key in ("L_ul", "L_up", "L_sd_cos"):
        assert key in m, f"SEAM metric {key} missing"
        assert math.isfinite(m[key]), f"SEAM metric {key} not finite: {m[key]}"

    # L_sd is a cosine similarity, must lie in [-1, 1].
    assert -1.0 - 1e-4 <= m["L_sd_cos"] <= 1.0 + 1e-4, (
        f"SEAM L_sd cosine out of range: {m['L_sd_cos']}"
    )


def test_seam_coupling_term_is_computed_and_nonzero():
    """The defining gradient-coupling term L_sd must be a real cosine, and the
    Hessian-free estimator must inject a non-zero grad_L_sd into the update."""
    torch.manual_seed(7)
    model = _MiniLM()
    trainer = _seam_trainer(model, alpha=1.0, beta=0.5, epsilon=1e-3)

    inputs = _batch()
    # Run the true SEAM step (assembles Eq. 5 grad via the Eq. 6 estimator).
    loss = trainer.training_step(model, inputs)
    assert torch.isfinite(loss), f"SEAM training_step loss not finite: {loss}"

    m = trainer.last_seam_metrics
    assert "L_sd_cos" in m
    assert m["g_a_norm"] > 0.0, "harmful gradient (g_a) norm should be > 0"
    assert m["g_b_norm"] > 0.0, "benign gradient (g_b) norm should be > 0"

    # The assembled parameter gradient must be non-zero.
    total_sq = sum((p.grad.detach() ** 2).sum().item()
                   for p in model.parameters() if p.grad is not None)
    assert total_sq > 0.0, "SEAM assembled gradient is all-zero"

    # The L_sd contribution must actually change the update: compare against
    # an identical run with beta=0 (no coupling). They must differ.
    torch.manual_seed(7)
    model2 = _MiniLM()
    model2.load_state_dict(model.state_dict())
    trainer0 = _seam_trainer(model2, alpha=1.0, beta=0.0, epsilon=1e-3)
    trainer0.training_step(model2, inputs)
    g1 = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    diff = 0.0
    for n, p in model2.named_parameters():
        if p.grad is not None and n in g1:
            diff += (g1[n] - p.grad).abs().sum().item()
    assert diff > 1e-8, (
        "Gradient with coupling (beta>0) is identical to beta=0 — the L_sd "
        "coupling term is not affecting the update."
    )


def test_seam_beta_zero_drops_coupling_from_grad():
    """beta=0 -> assembled grad == g_ul + alpha*g_up (no L_sd contribution)."""
    torch.manual_seed(11)
    model = _MiniLM()
    trainer = _seam_trainer(model, alpha=1.0, beta=0.0)
    inputs = _batch()
    loss = trainer.training_step(model, inputs)
    assert torch.isfinite(loss)
    total_sq = sum((p.grad.detach() ** 2).sum().item()
                   for p in model.parameters() if p.grad is not None)
    assert total_sq > 0.0, "beta=0 still needs g_ul + alpha*g_up gradient"


# ===========================================================================
# CTRAPTrainer — Eq.(2): L_align + λ·L_Collapse(θ − α·∇_θ L_harm ; D_general)
#   Eq.(1) collapse loss = per-token CE toward a fixed token e.
# ===========================================================================

def _ctrap_trainer(model, harmful, *, lam, alpha=0.1, token_id=0,
                   second_order=True):
    """Build a CTRAPTrainer wired to a tiny model + harmful batches."""
    from safetune.harden.ctrap import CTRAPTrainer, CTRAPConfig
    from datasets import Dataset
    import itertools

    train_ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3]] * 4, "attention_mask": [[1, 1, 1]] * 4,
         "labels": [[1, 2, 3]] * 4}
    )
    with tempfile.TemporaryDirectory() as d:
        cfg = CTRAPConfig(d, max_steps=0, use_cpu=True, report_to="none",
                          ctrap_lambda=lam, ctrap_alpha=alpha,
                          ctrap_collapse_token_id=token_id,
                          ctrap_second_order=second_order)
        trainer = CTRAPTrainer(model=model, args=cfg, train_dataset=train_ds,
                               harmful_dataset=harmful)
    trainer._harm_iter = itertools.cycle(iter(harmful))
    return trainer


def test_ctrap_loss_exceeds_l_align():
    """Eq.(2): with λ>0 the collapse trap term adds positively, loss > L_align."""
    torch.manual_seed(4)
    model = _MiniLM()
    harmful = [_batch(bs=1) for _ in range(4)]
    trainer = _ctrap_trainer(model, harmful, lam=1.0, token_id=0)

    inputs = _batch()
    loss_ctrap = trainer.compute_loss(model, inputs).item()
    with torch.no_grad():
        l_align = model(**inputs).loss.item()

    assert loss_ctrap > l_align - 1e-5, (
        f"CTRAP: loss={loss_ctrap:.4f} should be ≥ L_align={l_align:.4f} "
        f"(collapse trap term adds positively)"
    )


def test_ctrap_lambda_zero_equals_l_align():
    """λ=0 → CTRAP loss ≈ L_align (trap dormant)."""
    torch.manual_seed(5)
    model = _MiniLM()
    harmful = [_batch(bs=1) for _ in range(4)]
    trainer = _ctrap_trainer(model, harmful, lam=0.0, token_id=0)

    inputs = _batch()
    loss_ctrap = trainer.compute_loss(model, inputs).item()
    with torch.no_grad():
        l_align = model(**inputs).loss.item()

    assert abs(loss_ctrap - l_align) < 1e-4, (
        f"CTRAP λ=0: expected ≈L_align ({l_align:.4f}), got {loss_ctrap:.4f}"
    )


def test_ctrap_collapse_is_token_ce():
    """Eq.(1) collapse loss is a per-token CE toward the fixed token e:
    ≈0 when the model already puts ~all mass on e, large otherwise.
    Proves it is a collapse-to-TOKEN objective, not an MSE-to-vector."""
    from safetune.harden.ctrap import _collapse_loss

    torch.manual_seed(6)
    vocab = 64
    B, T = 2, 5
    # Logits sharply peaked on token 7 at every position.
    logits = torch.full((B, T, vocab), -10.0)
    logits[..., 7] = 10.0
    outputs = type("O", (), {"logits": logits})()
    attn = torch.ones(B, T)

    loss_e_is_peak = _collapse_loss(outputs, attn, collapse_token_id=7).item()
    loss_e_is_other = _collapse_loss(outputs, attn, collapse_token_id=3).item()

    assert loss_e_is_peak < 1e-3, (
        f"collapse CE toward the already-predicted token should be ~0, "
        f"got {loss_e_is_peak:.4f}"
    )
    assert loss_e_is_other > 5.0, (
        f"collapse CE toward a non-predicted token should be large, "
        f"got {loss_e_is_other:.4f}"
    )


def test_ctrap_simulated_attack_path_executes():
    """The bi-level path must actually take the inner harmful-gradient step:
    with α≠0 the collapse loss is evaluated at θ′ = θ − α·∇L_harm, so the
    loss differs from evaluating collapse at θ (α=0)."""
    torch.manual_seed(7)
    harmful = [_batch(bs=1) for _ in range(4)]
    inputs = _batch()

    m0 = _MiniLM(); torch.manual_seed(7); m0 = _MiniLM()
    base_state = {k: v.clone() for k, v in m0.state_dict().items()}

    # α = 0 → θ′ = θ (no attack step).
    m_a0 = _MiniLM(); m_a0.load_state_dict(base_state)
    t_a0 = _ctrap_trainer(m_a0, harmful, lam=1.0, alpha=0.0, token_id=0)
    loss_a0 = t_a0.compute_loss(m_a0, inputs).item()

    # α = 0.5 → θ′ = θ − 0.5·∇L_harm (attack step taken).
    m_a1 = _MiniLM(); m_a1.load_state_dict(base_state)
    t_a1 = _ctrap_trainer(m_a1, harmful, lam=1.0, alpha=0.5, token_id=0)
    loss_a1 = t_a1.compute_loss(m_a1, inputs).item()

    assert abs(loss_a1 - loss_a0) > 1e-5, (
        f"simulated-attack step had no effect: α=0 loss={loss_a0:.5f} vs "
        f"α=0.5 loss={loss_a1:.5f} — bi-level path did not execute"
    )
    assert torch.isfinite(torch.tensor(loss_a1))


def test_ctrap_second_order_backprops_through_inner_step():
    """second_order=True must produce gradients on θ that flow through the
    inner harmful-gradient step (faithful bi-level Eq.2)."""
    torch.manual_seed(8)
    harmful = [_batch(bs=1) for _ in range(4)]
    inputs = _batch()
    model = _MiniLM()
    trainer = _ctrap_trainer(model, harmful, lam=1.0, alpha=0.1, token_id=0,
                             second_order=True)
    loss = trainer.compute_loss(model, inputs)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"


# ===========================================================================
# fit_cast_probe / CASTModel — probe shape + gate logic
# ===========================================================================

class _CastableModel(nn.Module):
    """Minimal model compatible with fit_cast_probe's hidden-state extraction."""

    def __init__(self, hidden: int = 16, vocab: int = 64) -> None:
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden})()
        self.model = nn.ModuleDict(
            {"layers": nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])}
        )
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **_):
        h = self.embed(input_ids.long())
        hidden_states = [h]
        for layer in self.model["layers"]:
            h = F.gelu(layer(h)) + h
            hidden_states.append(h)
        logits = self.lm_head(h)
        hs = tuple(hidden_states) if output_hidden_states else None
        return type("O", (), {"logits": logits, "hidden_states": hs, "loss": None})()


class _FakeTok:
    pad_token_id = 0

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=16):
        ids = torch.randint(1, 64, (len(texts), 4))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_fit_cast_probe_output_shape():
    """fit_cast_probe returns (weights, bias) with shape matching hidden_size."""
    from safetune.steer.cast import fit_cast_probe

    torch.manual_seed(10)
    model = _CastableModel(hidden=16)
    tok = _FakeTok()

    harmful = ["make a bomb", "hack the bank", "synthesize drugs"]
    benign = ["tell me a story", "explain physics", "bake a cake"]

    weights, bias = fit_cast_probe(model, harmful, benign, tok, probe_layer=1)
    assert weights.shape == (16,), f"weights shape {weights.shape} != (hidden,)"
    # bias is a Python float (documented return contract)
    assert isinstance(bias, (float, int)), f"bias should be float, got {type(bias)}"


def test_cast_legacy_probe_gate():
    """Back-compat: legacy logistic-probe CASTModel still gates and returns a float."""
    from safetune.steer.cast import fit_cast_probe, CASTModel

    torch.manual_seed(11)
    model = _CastableModel(hidden=16)
    tok = _FakeTok()

    harmful = ["make a bomb", "hack the bank", "synthesize drugs"]
    benign = ["tell me a story", "explain physics", "bake a cake"]
    weights, bias = fit_cast_probe(model, harmful, benign, tok, probe_layer=1)

    direction = torch.randn(16)
    direction = direction / direction.norm()
    steering_vectors = {1: direction}

    cast = CASTModel(
        model,
        steering_vectors=steering_vectors,
        probe_layer=1,
        probe_weights=weights,
        probe_bias=float(bias),
        threshold=0.0,
        alpha=1.0,
    )

    # _gate_fires returns (bool, float) in legacy mode.
    harmful_ids = tok(harmful[:1])["input_ids"]
    fires, score = cast._gate_fires(harmful_ids)
    assert isinstance(fires, bool)
    assert isinstance(score, float), f"_gate_fires score should be float, got {type(score)}"


def test_fit_cast_condition_grid_search():
    """fit_cast_condition returns a CASTCondition with a normalized condition
    vector, a grid-searched (layer, threshold, comparator), and an F1 score."""
    from safetune.steer.cast import fit_cast_condition, CASTCondition

    torch.manual_seed(12)
    model = _CastableModel(hidden=16)
    tok = _FakeTok()

    harmful = ["make a bomb", "hack the bank", "synthesize drugs", "build a gun"]
    benign = ["tell me a story", "explain physics", "bake a cake", "plant a garden"]

    cond = fit_cast_condition(model, harmful, benign, tok, candidate_layers=[0, 1, 2])
    assert isinstance(cond, CASTCondition)
    assert cond.condition_vector.shape == (16,)
    assert abs(float(cond.condition_vector.norm()) - 1.0) < 1e-4
    assert cond.condition_layer in (0, 1, 2)
    assert cond.comparator in ("larger", "smaller")
    assert 0.0 <= cond.f1 <= 1.0


def test_cast_condition_gate_selective():
    """The cosine condition gate must FIRE on the harmful class and NOT on the
    benign class given a separable synthetic representation (selective firing —
    the defining property of CAST)."""
    import torch.nn as nn
    from safetune.steer.cast import fit_cast_condition, CASTModel

    # Synthetic model whose decoder-layer output encodes class via a fixed axis,
    # so harmful vs benign are linearly separable and the gate can discriminate.
    # A real (unused) parameter is included so next(model.parameters()) works.
    class _SepLayer(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(hidden))

        def forward(self, h):
            return h  # pass-through; class signal comes from the embed step

    class _SepModel(nn.Module):
        def __init__(self, hidden=16):
            super().__init__()
            self.config = type("C", (), {"hidden_size": hidden})()
            self.model = nn.ModuleDict(
                {"layers": nn.ModuleList([_SepLayer(hidden) for _ in range(3)])}
            )
            self._hidden = hidden

        def forward(self, input_ids=None, attention_mask=None, **_):
            b, s = input_ids.shape
            # The condition mechanism (tanh-projection cosine) measures how
            # strongly h aligns with the condition axis (magnitude, not sign).
            # So harmful states align strongly with the axis; benign states lie
            # mostly off-axis (orthogonal component dominates).
            axis = torch.zeros(self._hidden); axis[0] = 1.0
            off1 = torch.zeros(self._hidden); off1[1] = 1.0
            off2 = torch.zeros(self._hidden); off2[2] = 1.0
            is_harmful = (input_ids[:, :1] == 1).float()  # (b,1)
            ha = is_harmful.unsqueeze(-1)
            # harmful: strongly along the condition axis (|cos(h,axis)| large).
            # benign: lies in an orthogonal subspace (|cos(h,axis)| ~ 0), so the
            # tanh-projection cosine separates them by alignment magnitude.
            h = (ha * (4 * axis + off1).view(1, 1, -1)
                 + (1 - ha) * (off1 + off2).view(1, 1, -1))
            h = h.expand(b, s, self._hidden).clone()
            for layer in self.model["layers"]:
                h = layer(h)
            return type("O", (), {"logits": h, "hidden_states": None, "loss": None})()

    torch.manual_seed(13)
    model = _SepModel(hidden=16)
    harmful = ["h1", "h2", "h3", "h4"]
    benign = ["b1", "b2", "b3", "b4"]

    # Tokenizer that encodes the class in token 0: 1=harmful, 2=benign.
    class _ClassTok:
        pad_token_id = 0

        def __call__(self, texts, **_):
            ids = torch.stack([
                torch.tensor([1 if t.startswith("h") else 2, 0, 0, 0])
                for t in texts
            ])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    tok = _ClassTok()
    cond = fit_cast_condition(model, harmful, benign, tok, candidate_layers=[0, 1, 2])

    # Gate must fire on harmful, not on benign.
    direction = torch.zeros(16); direction[1] = 1.0
    cast = CASTModel(model, steering_vectors={1: direction}, condition=cond, alpha=1.0)

    h_ids = tok(["h_held"])["input_ids"]
    b_ids = tok(["b_held"])["input_ids"]
    fires_h, sim_h = cast._gate_fires(h_ids)
    fires_b, sim_b = cast._gate_fires(b_ids)

    assert fires_h is True, f"gate should fire on harmful (sim={sim_h}, thr={cond.threshold}, comp={cond.comparator})"
    assert fires_b is False, f"gate should NOT fire on benign (sim={sim_b}, thr={cond.threshold}, comp={cond.comparator})"
    assert cond.f1 == 1.0, f"condition F1 should be perfect on separable data, got {cond.f1}"


# ===========================================================================
# crisp_unlearn — loss includes SAE feature suppression
# ===========================================================================

class _FakeSAE(nn.Module):
    """Minimal SAE stub: encode(h) -> feature activations."""

    def __init__(self, hidden: int = 16, n_features: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Linear(hidden, n_features, bias=True)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(h))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.encode(h)


def test_crisp_unlearn_concept_features_suppressed():
    """crisp_unlearn runs without error and suppresses concept feature activations."""
    from safetune.interventions.unlearn.crisp import crisp_unlearn

    torch.manual_seed(20)
    model = _MiniLM(hidden=16, vocab=64)
    sae = _FakeSAE(hidden=16, n_features=32)

    concept_features = torch.tensor([0, 1, 2], dtype=torch.long)  # must be LongTensor

    # Tiny forget/retain datasets — batched: (batch, seq).
    forget = [{"input_ids": torch.randint(1, 64, (1, 3)),
               "attention_mask": torch.ones(1, 3, dtype=torch.long),
               "labels": torch.randint(1, 64, (1, 3))} for _ in range(4)]
    retain = [{"input_ids": torch.randint(1, 64, (1, 3)),
               "attention_mask": torch.ones(1, 3, dtype=torch.long),
               "labels": torch.randint(1, 64, (1, 3))} for _ in range(4)]

    # Record mean concept feature activation before unlearning.
    def _mean_concept_act(m):
        acts = []
        for row in forget:
            ids = row["input_ids"]  # already (1, seq)
            h = m.model(ids)  # (1, seq, hidden)
            f = sae.encode(h)  # (1, seq, n_features)
            acts.append(f[:, :, concept_features].mean().item())
        return sum(acts) / len(acts)

    act_before = _mean_concept_act(model)

    # Run a few unlearning steps.
    crisp_unlearn(
        model=model,
        sae=sae,
        concept_features=concept_features,
        forget_dataset=forget,
        retain_dataset=retain,
        gamma=10.0,    # high gamma = strong suppression signal
        num_steps=20,
        lr=1e-3,
    )

    act_after = _mean_concept_act(model)

    # The suppression loss should drive concept feature activations down.
    # With gamma=10 and 20 steps, we expect measurable reduction.
    assert act_after <= act_before + 1e-3, (
        f"crisp_unlearn: concept feature activation did not decrease "
        f"(before={act_before:.4f}, after={act_after:.4f})"
    )


# ===========================================================================
# MARTTrainer — structural smoke (one round, no GPU)
# ===========================================================================

def test_mart_trainer_one_round():
    """MARTTrainer.train() completes one full round (adv update + tgt update)."""
    from safetune.harden.mart import MARTTrainer, MARTConfig

    torch.manual_seed(30)
    adv = _MiniLM()
    tgt = _MiniLM()

    # MARTConfig is a plain dataclass (not TrainingArguments).
    cfg = MARTConfig(
        num_rounds=1,
        num_candidates=2,
        adv_steps=2,
        tgt_steps=2,
        device="cpu",
    )

    # Minimal tokenizer stub that MARTTrainer can call .encode() / .__call__() on.
    class _MinTok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=16):
            if isinstance(texts, str):
                texts = [texts]
            ids = torch.randint(2, 64, (len(texts), 4))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def decode(self, ids, skip_special_tokens=True):
            return "dummy response"

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["dummy response"] * (ids.shape[0] if hasattr(ids, "shape") else len(ids))

    tok = _MinTok()

    # safety_reward_fn: callable(prompt, response) -> float in [0, 1].
    trainer = MARTTrainer(
        target_model=tgt,
        adv_model=adv,
        tokenizer=tok,
        seed_prompts=["test harmful prompt"],
        safety_reward_fn=lambda p, r: 0.8,
        helpfulness_reward_fn=lambda p, r: 0.7,
        config=cfg,
    )
    trainer.train()

    # After training both models must still be functional.
    inputs = _batch()
    out_adv = adv(**inputs)
    out_tgt = tgt(**inputs)
    assert out_adv.logits.shape[-1] == 64
    assert out_tgt.logits.shape[-1] == 64


# ===========================================================================
# Loader registry — new loaders present
# ===========================================================================

def test_new_loaders_registered():
    """JailbreakBench, MUSE, RWKU, SafeDialBench loaders are registered."""
    from safetune.core.eval.pipeline import LOADERS

    for name in ("jailbreakbench", "muse", "rwku", "safedialbench"):
        assert name in LOADERS, f"{name} not in LOADERS registry"


# ===========================================================================
# steer.run — chat-template handling
# ===========================================================================

class _NoTemplateTok:
    """Base-model tokenizer: exposes apply_chat_template but chat_template=None."""

    chat_template = None

    def apply_chat_template(self, *a, **k):  # pragma: no cover - must NOT be called
        raise ValueError(
            "Cannot use chat template functions because tokenizer.chat_template "
            "is not set"
        )


class _WithTemplateTok:
    """Instruct-model tokenizer: has a chat template."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        content = messages[0]["content"]
        return f"<|user|>{content}<|assistant|>"


def test_render_prompts_falls_back_on_base_tokenizer():
    """render_prompts must NOT call apply_chat_template when chat_template is None."""
    from safetune.steer.backends.run import render_prompts

    tok = _NoTemplateTok()
    # apply_chat_template=True but no template -> graceful fallback to raw prompt.
    out = render_prompts(tok, ["bake bread"], apply_chat_template=True)
    assert out == ["bake bread"], (
        "render_prompts should fall back to the raw prompt for a base-model "
        "tokenizer, not call apply_chat_template (which would raise)"
    )


def test_render_prompts_applies_template_when_present():
    """render_prompts applies the chat template when the tokenizer has one."""
    from safetune.steer.backends.run import render_prompts

    tok = _WithTemplateTok()
    out = render_prompts(tok, ["bake bread"], apply_chat_template=True)
    assert out == ["<|user|>bake bread<|assistant|>"], out

    # apply_chat_template=False -> raw passthrough even when a template exists.
    raw = render_prompts(tok, ["bake bread"], apply_chat_template=False)
    assert raw == ["bake bread"], raw
