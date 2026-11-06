"""Audit R08: an aligned adapter is merged with the scaling PEFT uses, not always alpha / r.

apply_safe_lora(aligned_adapter_path=...) rebuilds W_aligned = W_base + scaling * B @ A from the
saved adapter. PEFT's LoraLayer.update_layer uses lora_alpha / sqrt(r) when use_rslora is set, and
rank_pattern / alpha_pattern override r and alpha per module. With r=32, alpha=64 and rsLoRA the
update is scaled by 11.31, not 2.0, so V = W_aligned - W_base and the projection C built from it
were 5.66x too small. Loading the adapter must give the same result as passing the weights PEFT
merges from it.
"""
import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model

from safetune.recover import apply_safe_lora


class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(32, 32, bias=False)
        self.k_proj = torch.nn.Linear(32, 32, bias=False)


@pytest.mark.parametrize("lora_kwargs", [
    dict(r=32, lora_alpha=64),
    dict(r=32, lora_alpha=64, use_rslora=True),
    dict(r=8, lora_alpha=16, rank_pattern={"k_proj": 4}, alpha_pattern={"k_proj": 32}),
], ids=["plain", "rslora", "patterns"])
def test_aligned_adapter_matches_the_weights_peft_merges(tmp_path, lora_kwargs):
    torch.manual_seed(0)
    base = _M()
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}

    # The aligned model is the base plus a safety adapter.
    aligned = get_peft_model(copy.deepcopy(base),
                             LoraConfig(target_modules=["q_proj", "k_proj"], **lora_kwargs))
    with torch.no_grad():
        for n, p in aligned.named_parameters():
            if "lora_" in n:
                p.copy_(0.1 * torch.randn_like(p))
    aligned.save_pretrained(tmp_path)
    aligned_sd = aligned.merge_and_unload().state_dict()

    torch.manual_seed(1)
    ft = _M()
    with torch.no_grad():
        for n, p in ft.named_parameters():
            p.copy_(aligned_sd[n] + 0.05 * torch.randn_like(p))
    from_adapter, from_weights = copy.deepcopy(ft), copy.deepcopy(ft)

    kw = dict(base_state_dict=base_sd, threshold=1.0, target_modules=["q_proj", "k_proj"])
    apply_safe_lora(from_adapter, aligned_adapter_path=str(tmp_path), **kw)
    apply_safe_lora(from_weights, aligned_state_dict=aligned_sd, **kw)

    for name in ("q_proj", "k_proj"):
        got = getattr(from_adapter, name).weight
        want = getattr(from_weights, name).weight
        diff = (got - want).abs().max().item()
        assert diff < 1e-5, f"{name}: max |diff| = {diff:.4f}"
