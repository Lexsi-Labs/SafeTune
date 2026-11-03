"""Audit H19: SaLoRA drops its safety module when the adapter is merged, and starts off the pretrained model.

Li et al. 2025 (arXiv:2501.01765) Sec. 4.2-4.3: the layer is re-parameterised as W' = W - C_S B_S A_S
(Eq. 13) so that the initial model equals the pretrained one, and at saving time the safety module is
folded into the adapter, B' = C_S B_S (Sec. 4.3, Fig. 4). No official code is linked from the paper:
this audit rests on the paper alone.
SaLoRATrainer installs C as a forward hook on lora_B, and `_save_merged` calls `merge_and_unload()`,
which adds B A without C (tiny Llama, float32: trained-vs-merged max|dlogits| = 0.426).
"""
import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import safetune.harden as H
from safetune.runner.harden import SaLoRATrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _build(strength=1.0, rslora=False):
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    e = tok(["user question here answer text for the model to learn",
             "harmful request tell me how to do the bad thing"],
            padding="max_length", max_length=24, return_tensors="pt")
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    model = get_peft_model(copy.deepcopy(base), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM",
        use_rslora=rslora))
    safety_inputs = [{"input_ids": e["input_ids"][i:i + 1], "attention_mask": e["attention_mask"][i:i + 1]}
                     for i in range(2)]
    sub = H.compute_safety_subspace(model, safety_inputs=safety_inputs, safety_rank=8)
    H.project_lora_step(model, sub, H.SaLoRAConfig(rank=8, safety_rank=8, strength=strength),
                        install_forward=True, apply_task_init=True)
    model.eval()
    return tok, base, model, e


@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_saved_merged_model_is_the_trained_model(strength, tmp_path):
    tok, base, model, e = _build(strength)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.add_(0.02 * torch.randn_like(p))         # stand-in for training
        trained = model(**e).logits
    path = SaLoRATrainer(model, tok, results_dir=str(tmp_path))._save_merged(model, str(tmp_path / "out"))
    saved = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        diff = (saved(**e).logits - trained).abs().max().item()
    assert diff < 1e-4, f"max|dlogits| = {diff:.5f} between the trained model and the saved merged checkpoint"


def test_fold_is_idempotent_and_removes_the_hook():
    _, _, model, e = _build()
    with torch.no_grad():
        before = model(**e).logits
    assert H.fold_safety_module(model) > 0
    assert H.fold_safety_module(model) == 0
    with torch.no_grad():
        assert (model(**e).logits - before).abs().max().item() < 1e-5


@pytest.mark.parametrize("strength,rslora", [(1.0, False), (0.5, False), (1.0, True)])
def test_initial_model_equals_the_pretrained_model(strength, rslora):
    _check_identity(*_build(strength, rslora)[1:])


def test_initial_model_equals_the_pretrained_model_on_conv1d_layers():
    """GPT-2's c_attn is a Conv1D: the base weight is stored (in, out), PEFT's fan_in_fan_out.

    C is built by hand (I - Q Q.T on lora_B's output space): compute_safety_subspace does not find
    Conv1D base layers, which this test does not cover."""
    name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    tok = AutoTokenizer.from_pretrained(name)
    e = tok(["user question here answer text for the model"], return_tensors="pt")
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    model = get_peft_model(copy.deepcopy(base), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    sub = {}
    for n, mod in model.named_modules():
        if hasattr(mod, "lora_B") and hasattr(mod, "base_layer"):
            out = mod.lora_B["default"].weight.shape[0]
            q = torch.linalg.qr(torch.randn(out, 4))[0]
            sub[n] = torch.eye(out) - q @ q.T
    assert sub
    H.project_lora_step(model, sub, H.SaLoRAConfig(rank=8, safety_rank=4),
                        install_forward=True, apply_task_init=True)
    _check_identity(base, model.eval(), e)


def _check_identity(base, model, e):
    with torch.no_grad():
        diff = (model(**e).logits - base(**e).logits).abs().max().item()
    assert diff < 1e-5, f"max|dlogits| = {diff:.6f} between the initial SaLoRA model and the pretrained one"


def test_the_base_weight_is_actually_offset():
    _, base, model, _ = _build()
    w0 = dict(base.named_parameters())["model.layers.0.self_attn.q_proj.weight"]
    w1 = dict(model.named_parameters())["base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"]
    assert (w1 - w0).abs().max().item() > 0
