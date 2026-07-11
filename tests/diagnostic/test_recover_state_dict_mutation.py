"""
Empirical diagnostic: every Recover method must mutate model weights.

A method that returns weights identical to the input is a no-op bug. This
harness exercises each Recover method on tiny synthetic Llama-shaped modules
and asserts the model's state_dict changes.

Run with:

    PYTHONPATH=src python -m pytest tests/diagnostic/test_recover_state_dict_mutation.py -v --no-cov
"""
from __future__ import annotations

import copy
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn


EPS = 1e-5


class _Block(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(hidden, hidden, bias=False),
                "k_proj": nn.Linear(hidden, hidden, bias=False),
                "v_proj": nn.Linear(hidden, hidden, bias=False),
                "o_proj": nn.Linear(hidden, hidden, bias=False),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden, hidden * 2, bias=False),
                "up_proj": nn.Linear(hidden, hidden * 2, bias=False),
                "down_proj": nn.Linear(hidden * 2, hidden, bias=False),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.self_attn["q_proj"](x)
        k = self.self_attn["k_proj"](x)
        v = self.self_attn["v_proj"](x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (x.shape[-1] ** 0.5), dim=-1) @ v
        attn = self.self_attn["o_proj"](attn) + x
        gate = torch.nn.functional.silu(self.mlp["gate_proj"](attn))
        up = self.mlp["up_proj"](attn)
        return self.mlp["down_proj"](gate * up) + attn


class _Wrap(nn.Module):
    def __init__(self, hidden: int = 32, n_layers: int = 4, vocab: int = 100) -> None:
        super().__init__()
        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(vocab, hidden),
                "layers": nn.ModuleList([_Block(hidden) for _ in range(n_layers)]),
            }
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.long, torch.int64):
            input_ids = input_ids.long()
        h = self.model["embed_tokens"](input_ids)
        for blk in self.model["layers"]:
            h = blk(h)
        return self.lm_head(h)


def _make(seed: int) -> _Wrap:
    torch.manual_seed(seed)
    return _Wrap()


def _perturb(model: nn.Module, seed: int, scale: float = 0.1) -> nn.Module:
    out = copy.deepcopy(model)
    with torch.no_grad():
        torch.manual_seed(seed)
        for p in out.parameters():
            p.add_(torch.randn_like(p) * scale)
    return out


def _max_abs_diff(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor]) -> float:
    diffs = []
    for k in sd_a:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            diffs.append((sd_a[k].float() - sd_b[k].float()).abs().max().item())
    return max(diffs) if diffs else 0.0


def _moved_toward(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
) -> float:
    db, dt = [], []
    for k in before:
        if k in after and k in target and before[k].shape == after[k].shape == target[k].shape:
            db.append((after[k].float() - before[k].float()).flatten())
            dt.append((target[k].float() - before[k].float()).flatten())
    if not db:
        return 0.0
    a = torch.cat(db)
    b = torch.cat(dt)
    if a.norm() < 1e-12 or b.norm() < 1e-12:
        return 0.0
    return float((a @ b) / (a.norm() * b.norm()))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_toxic() -> Tuple[nn.Module, nn.Module]:
    """Two-model fixture: clean reference + toxic (perturbed)."""
    clean = _make(seed=0)
    toxic = _perturb(clean, seed=1, scale=0.1)
    return clean, toxic


@pytest.fixture
def trio() -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Three-model fixture: base, aligned (base + safety drift), finetuned (aligned + task drift).

    This shape is required by RESTA / LoX / task arithmetic, which compute a
    safety vector (aligned - base) and add it to the fine-tuned model.
    """
    base = _make(seed=0)
    aligned = _perturb(base, seed=10, scale=0.1)
    finetuned = _perturb(aligned, seed=20, scale=0.1)
    return base, aligned, finetuned


# ---------------------------------------------------------------------------
# Control: RESTA (a Working ✅ method in the CSV)
# ---------------------------------------------------------------------------

def test_resta_mutates(trio):
    base, aligned, finetuned = trio
    from safetune.recover import apply_resta

    before = {k: v.detach().clone() for k, v in finetuned.state_dict().items()}
    apply_resta(finetuned, base=base, aligned=aligned, alpha=1.0)
    after = finetuned.state_dict()

    diff = _max_abs_diff(before, after)
    assert diff > EPS, f"RESTA was a no-op (max abs diff = {diff:.2e})"


def test_resta_adds_safety_vector(trio):
    base, aligned, finetuned = trio
    from safetune.recover import apply_resta

    before = {k: v.detach().clone() for k, v in finetuned.state_dict().items()}
    target = {k: (before[k].float() + (aligned.state_dict()[k].float() - base.state_dict()[k].float())) for k in before}
    apply_resta(finetuned, base=base, aligned=aligned, alpha=1.0)
    after = finetuned.state_dict()

    cos = _moved_toward(before, after, target)
    assert cos > 0.95, f"RESTA direction wrong (cos with safety vector = {cos:.3f}); expected ~1.0"


def test_task_arithmetic_mutates(trio):
    base, aligned, finetuned = trio
    from safetune.recover import task_arithmetic

    before = {k: v.detach().clone() for k, v in finetuned.state_dict().items()}
    out = task_arithmetic(finetuned, base=base, aligned=aligned, alpha=1.0)
    after = (out if hasattr(out, "state_dict") else finetuned).state_dict()

    diff = _max_abs_diff(before, after)
    assert diff > EPS, f"task_arithmetic was a no-op (max abs diff = {diff:.2e})"


# ---------------------------------------------------------------------------
# Tier 0: previously reported no-op trio
# ---------------------------------------------------------------------------

class _TinyTok:
    """Char-level tokenizer so PKE's DINM refusal-CE edit can tokenize."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1

    def __call__(self, text, add_special_tokens=True):
        ids = [(ord(c) % (self.vocab_size - 1)) for c in text]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


def test_pke_mutates(clean_toxic):
    clean, toxic = clean_toxic
    from safetune.recover import apply_pke

    before = {k: v.detach().clone() for k, v in toxic.state_dict().items()}
    apply_pke(toxic, clean=clean, toxic=copy.deepcopy(toxic), top_k_neurons=10,
              max_edit_magnitude=10.0, tokenizer=_TinyTok(), num_steps=8, lr=1e-2,
              max_len=48)
    after = toxic.state_dict()

    diff = _max_abs_diff(before, after)
    assert diff > EPS, f"PKE was a no-op (max abs diff = {diff:.2e})"


def test_pke_edit_loss_is_refusal_ce(clean_toxic):
    """The faithful PKE/DINM edit is driven by refusal cross-entropy, not by
    weight regression toward ``clean``. Assert the per-step edit CE decreases
    (the located rows learn to emit the refusal)."""
    clean, toxic = clean_toxic
    from safetune.recover.pke import PKEGradientEditor, PKEConfig
    from safetune.core.pke import ToxicNeuronLocator

    cfg = PKEConfig(top_k_neurons=10, num_steps=12, lr=1e-2,
                    max_edit_magnitude=None, max_len=48)
    loc = ToxicNeuronLocator(config=cfg)
    tn = loc.locate_by_weight_change(clean.state_dict(), toxic.state_dict())
    ed = PKEGradientEditor(toxic, tn, tokenizer=_TinyTok(), config=cfg)
    edited = ed.apply_edits()

    assert edited > 0
    assert len(ed.last_edit_losses) == cfg.num_steps
    assert ed.last_edit_losses[-1] < ed.last_edit_losses[0], (
        f"edit CE should decrease (refusal learned): "
        f"{ed.last_edit_losses[0]:.3f} -> {ed.last_edit_losses[-1]:.3f}"
    )
    # a logit-space KL locality term is computed every step
    assert len(ed.last_locality_losses) == cfg.num_steps


def test_pke_edits_only_located_rows(clean_toxic):
    """The edit gradient is masked to the located down_proj rows; no other
    weights move (DINM edits only the located toxic region)."""
    clean, toxic = clean_toxic
    from safetune.recover.pke import PKEGradientEditor, PKEConfig
    from safetune.core.pke import ToxicNeuronLocator

    before = {k: v.detach().clone() for k, v in toxic.state_dict().items()}
    cfg = PKEConfig(top_k_neurons=10, num_steps=6, lr=1e-2,
                    max_edit_magnitude=None, max_len=48)
    loc = ToxicNeuronLocator(config=cfg)
    tn = loc.locate_by_weight_change(clean.state_dict(), toxic.state_dict())
    PKEGradientEditor(toxic, tn, tokenizer=_TinyTok(), config=cfg).apply_edits()
    after = toxic.state_dict()

    for k in before:
        if "down_proj.weight" not in k:
            assert torch.allclose(before[k], after[k]), (
                f"non-down_proj param {k} changed but PKE must not touch it"
            )


def test_safereact_mutates(clean_toxic):
    clean, toxic = clean_toxic
    from safetune.recover import apply_safereact

    before = {k: v.detach().clone() for k, v in toxic.state_dict().items()}
    apply_safereact(toxic, reference_model=clean, top_k_neurons=8, reactivation_scale=1.0)
    after = toxic.state_dict()

    diff = _max_abs_diff(before, after)
    assert diff > EPS, f"SafeReact at scale=1.0 was a no-op (max abs diff = {diff:.2e})"


def test_safereact_moves_toward_reference(clean_toxic):
    clean, toxic = clean_toxic
    from safetune.recover import apply_safereact

    before = {k: v.detach().clone() for k, v in toxic.state_dict().items()}
    clean_sd = {k: v.detach().clone() for k, v in clean.state_dict().items()}
    apply_safereact(toxic, reference_model=clean, top_k_neurons=8, reactivation_scale=1.0)
    after = toxic.state_dict()

    cos = _moved_toward(before, after, clean_sd)
    assert cos > 0.0, f"SafeReact moved away from reference (cos = {cos:.3f})"


def test_nlsr_mutates(clean_toxic):
    clean, toxic = clean_toxic
    from safetune.recover import apply_nlsr

    donor_map: Dict[str, Dict[str, float]] = {}
    for name, p in clean.named_parameters():
        flat = p.detach().view(-1)
        donor_map[name] = {str(i): float(flat[i].item()) for i in range(min(8, flat.numel()))}

    before = {k: v.detach().clone() for k, v in toxic.state_dict().items()}
    apply_nlsr(toxic, donor_map=donor_map, blend=1.0)
    after = toxic.state_dict()

    diff = _max_abs_diff(before, after)
    assert diff > EPS, f"NLSR was a no-op (max abs diff = {diff:.2e})"


# ---------------------------------------------------------------------------
# Other Recover methods (smoke test, to catch any future regressions)
# ---------------------------------------------------------------------------

def test_lox_mutates(trio):
    base, aligned, finetuned = trio
    from safetune.recover import apply_lox

    before = {k: v.detach().clone() for k, v in finetuned.state_dict().items()}
    apply_lox(finetuned, base=base, aligned=aligned)
    after = finetuned.state_dict()

    diff = _max_abs_diff(before, after)
    assert diff > EPS, f"LoX was a no-op (max abs diff = {diff:.2e})"


def test_safe_lora_mutates(trio):
    base, aligned, finetuned = trio
    from safetune.recover import apply_safe_lora

    before = {k: v.detach().clone() for k, v in finetuned.state_dict().items()}
    # Updated for the faithfulness fix (see audit_faithfulness/fix/safe_lora.md):
    # the pre-fix code did a uniform alpha-interpolation merge that *always*
    # mutated every matched layer. Faithful Safe LoRA (Hsu et al., NeurIPS 2024,
    # arXiv:2405.16833 / IBM/SafeLoRA) is *selective*: it only projects layers
    # whose cosine(C@ΔW, ΔW) falls at or below the threshold. On this synthetic
    # random trio every layer's cosine is ~0.73-0.87, so the paper's
    # "threshold" mode at the default 0.5 correctly projects nothing. We
    # exercise the projection path with the paper's other selection mode,
    # "number", which projects the `num_proj_layers` lowest-cosine layers.
    apply_safe_lora(
        finetuned,
        aligned_state_dict=aligned.state_dict(),
        base_state_dict=base.state_dict(),
        select_layers_type="number",
        num_proj_layers=10,
    )
    after = finetuned.state_dict()

    diff = _max_abs_diff(before, after)
    assert diff > EPS, f"SafeLoRA was a no-op (max abs diff = {diff:.2e})"
