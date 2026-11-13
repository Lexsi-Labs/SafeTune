"""Audit H03: SafeGrad adds the alignment gradient once per micro-batch under gradient accumulation.

Fixed by rejecting gradient_accumulation_steps > 1 unless surgery_per_optimizer_step=True, which runs
the surgery once per optimizer step on the accumulated gradient (#80).

Gradient Surgery for Safe LLM Fine-Tuning (Yi et al., arXiv:2508.07172) Algorithm 1: per update,
g_final = g'_user + rho * g_align, with one alignment gradient per optimizer step. The HF loss is
divided by gradient_accumulation_steps, but SafeGradHFTrainer.training_step runs the surgery after
every micro-batch and writes rho * g_align into .grad each time (unscaled), so with accumulation 2
the alignment term is added twice and the second conflict test sees an already modified gradient.
"""
import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from safetune.harden import SafeGradHFTrainer
from safetune.core.optim.safegrad import safegrad_kl_alignment_loss


def _model(seed):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=2))


def _batch(seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 32, (1, 6), generator=g)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _flat_grad(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.requires_grad])


def test_safegrad_rejects_gradient_accumulation(tmp_path):
    from transformers import TrainingArguments
    args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=1,
                             gradient_accumulation_steps=2, use_cpu=True, report_to=[])
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        SafeGradHFTrainer(model=_model(0), args=args, safety_dataset=[_batch(100)], reference_model=_model(1))


@pytest.mark.parametrize("accum", [1, 2])
def test_safegrad_final_gradient_matches_algorithm_1(tmp_path, accum):
    from transformers import TrainingArguments
    model, ref = _model(0), _model(1)                    # reference differs so the KL gradient is nonzero
    safety = _batch(100)
    micro = [_batch(1), _batch(2)][:accum]
    rho = 1.0

    # hand-computed Algorithm 1 for one optimizer step
    probe = copy.deepcopy(model)
    g_user = None
    for b in micro:
        probe.zero_grad(); (probe(**b).loss / accum).backward()
        g = _flat_grad(probe)
        g_user = g if g_user is None else g_user + g
    probe.zero_grad()
    out = probe(**safety)
    with torch.no_grad():
        ref_logits = ref(**safety).logits
    kl = safegrad_kl_alignment_loss(out.logits, ref_logits)
    kl.backward()
    g_align = _flat_grad(probe)
    dot = (g_user * g_align).sum()
    g_proj = g_user - (dot / (g_align * g_align).sum()) * g_align if dot < 0 else g_user
    expected = g_proj + rho * g_align

    args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=1,
                             gradient_accumulation_steps=accum, use_cpu=True, report_to=[])
    tr = SafeGradHFTrainer(model=model, args=args, safety_dataset=[safety], reference_model=ref, rho=rho,
                           surgery_per_optimizer_step=accum > 1)
    tr.current_gradient_accumulation_steps = accum   # normally set inside Trainer.train()
    model.train()
    for i, b in enumerate(micro):
        # Trainer.train() marks the last micro-batch of the window as the sync step
        tr.accelerator.gradient_state._set_sync_gradients(i == accum - 1)
        tr.training_step(model, b)
    got = _flat_grad(model)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-6), (
        f"accum={accum}: ||got - expected|| / ||expected|| = {((got - expected).norm() / expected.norm()).item():.3f}"
    )


def test_safegrad_surgery_once_per_optimizer_step_in_train(tmp_path):
    """With accumulation 2, train() runs the surgery (and draws a safety batch) once per update."""
    from datasets import Dataset
    from transformers import TrainingArguments
    rows = [{k: v[0] for k, v in _batch(i).items()} for i in range(8)]
    args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=1,
                             gradient_accumulation_steps=2, max_steps=3, use_cpu=True,
                             report_to=[], save_strategy="no")
    tr = SafeGradHFTrainer(model=_model(0), args=args, train_dataset=Dataset.from_list(rows),
                           safety_dataset=[_batch(100), _batch(101)], reference_model=_model(1),
                           surgery_per_optimizer_step=True)
    calls, drawn = [], []
    surgery, draw = tr._apply_global_surgery, tr._next_safety_batch
    tr._apply_global_surgery = lambda m, g: (calls.append(1), surgery(m, g))[1]
    tr._next_safety_batch = lambda: (drawn.append(1), draw())[1]
    tr.train()
    assert tr.state.global_step == 3
    assert len(calls) == 3 and len(drawn) == 3
