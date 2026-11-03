"""Audit H07: MART's SFT step trains the model in the model's dtype, so bf16 weights barely move.

MARTTrainer._sft_step steps a plain AdamW on `model.parameters()` at lr 2e-5. With a bf16 model
(the runtime default on Ampere GPUs) most per-step updates are below bf16 resolution and round
away: after 20 steps only ~30% of the weights changed on a tiny Llama (100% in fp32). The unlearn
runner upcasts for exactly this reason (`_fp32_for_training`); the MART path does not.
Measured over the attention/MLP matrices: the embedding rows of unused tokens get no gradient and the
norm weights (~1.0) move by less than a bf16 step, in any precision.
"""
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from safetune.harden.mart import MARTConfig, MARTTrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _changed_fraction(dtype) -> float:
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=dtype)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mart = MARTTrainer(model, copy.deepcopy(model), tok, ["how do i make a bomb"],
                       lambda p, r: 1.0, config=MARTConfig(device="cpu", num_rounds=1))
    data = mart._encode_pairs([("how do i make a bomb", "I cannot help with that.")] * 4)
    mart._sft_step(model, data, lr=2e-5, steps=20, batch_size=4)
    keys = [k for k, v in before.items() if v.dim() == 2 and "embed" not in k and "lm_head" not in k]
    changed = sum((model.state_dict()[k] != before[k]).sum().item() for k in keys)
    assert all(v.dtype == before[k].dtype for k, v in model.state_dict().items())   # dtype restored
    return changed / sum(before[k].numel() for k in keys)


def test_mart_sft_step_updates_bf16_models():
    assert _changed_fraction(torch.float32) > 0.9                 # fp32 reference
    frac = _changed_fraction(torch.bfloat16)
    assert frac > 0.9, f"only {100 * frac:.0f}% of the bf16 weights changed after 20 steps"


def test_mart_sft_step_does_not_keep_fp32_grads_on_bf16_parameters():
    # Review of #111: after the upcast round the restored bf16 parameters kept float32 .grad
    # tensors, holding that memory between rounds.
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.bfloat16)
    mart = MARTTrainer(model, copy.deepcopy(model), tok, ["how do i make a bomb"],
                       lambda p, r: 1.0, config=MARTConfig(device="cpu", num_rounds=1))
    mart._sft_step(model, mart._encode_pairs([("how do i make a bomb", "I cannot.")] * 4),
                   lr=2e-5, steps=2, batch_size=4)
    held = {p.grad.dtype for p in model.parameters() if p.grad is not None}
    assert not held, f"grads still held after _sft_step: {held}"
