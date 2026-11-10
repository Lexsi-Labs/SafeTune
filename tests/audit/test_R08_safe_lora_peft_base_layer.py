"""Audit R08: Safe LoRA on a PEFT model finds the wrapped weights (``<module>.base_layer.weight``).

PEFT stores the weight a LoRA layer wraps under ``<module>.base_layer.weight``. apply_safe_lora
looked each module up as ``<module>.weight``, so on a PEFT model whose base weights were not passed
(the model's own are then used) no module matched: nothing was projected, and an
``aligned_adapter_path`` was never merged into the aligned weights either. The call returned with
only the "model was not mutated" warning. Taking the base from the PEFT model must give the same
adapter as passing the same base and aligned weights explicitly.
"""
import copy

import torch
from peft import LoraConfig, get_peft_model

from safetune.recover import apply_safe_lora


class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(32, 32, bias=False)
        self.k_proj = torch.nn.Linear(32, 32, bias=False)


def _adapter(base, seed, **lora_kwargs):
    torch.manual_seed(seed)
    model = get_peft_model(copy.deepcopy(base), LoraConfig(target_modules=["q_proj", "k_proj"], **lora_kwargs))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.copy_(0.1 * torch.randn_like(p))
    return model


def _lora_b(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}


def test_peft_model_with_aligned_adapter_and_no_base_is_projected(tmp_path):
    torch.manual_seed(0)
    base = _M()
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    lora = dict(r=8, lora_alpha=16, use_rslora=True)

    aligned = _adapter(base, 1, **lora)          # the aligned model: base + a safety adapter
    aligned.save_pretrained(tmp_path)
    aligned_sd = copy.deepcopy(aligned).merge_and_unload().state_dict()

    ft = _adapter(base, 2, **lora)               # the fine-tune: another adapter on the same base
    want_model, before = copy.deepcopy(ft), _lora_b(ft)

    apply_safe_lora(want_model, base_state_dict=base_sd, aligned_state_dict=aligned_sd, threshold=1.0)
    apply_safe_lora(ft, aligned_adapter_path=str(tmp_path), threshold=1.0)

    got, want = _lora_b(ft), _lora_b(want_model)
    assert all(not torch.equal(want[n], before[n]) for n in want)      # the reference call projects
    for n in want:
        diff = (got[n] - want[n]).abs().max().item()
        assert diff < 1e-5, f"{n}: max |diff| = {diff:.4f}"


def test_peft_keyed_state_dicts_are_matched():
    """State dicts taken from PEFT models (``base_layer`` keys) select the same weights."""
    torch.manual_seed(0)
    base = _M()
    aligned = copy.deepcopy(base)
    with torch.no_grad():
        for p in aligned.parameters():
            p.add_(0.3 * torch.randn_like(p))
    wrap = lambda m: {k: v.clone() for k, v in _adapter(m, 3, r=4).state_dict().items() if "lora_" not in k}

    ft = _adapter(aligned, 2, r=8)
    want_model = copy.deepcopy(ft)
    apply_safe_lora(want_model, base_state_dict=base.state_dict(), aligned_state_dict=aligned.state_dict(), threshold=1.0)
    apply_safe_lora(ft, base_state_dict=wrap(base), aligned_state_dict=wrap(aligned), threshold=1.0)

    got, want = _lora_b(ft), _lora_b(want_model)
    for n in want:
        assert torch.allclose(got[n], want[n], atol=1e-6), n
