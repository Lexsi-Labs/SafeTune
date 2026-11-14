"""SafeGrad alignment pass: padding (#81), failures (#82), bf16 softmax (#83)."""
import tempfile

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
from safetune.harden import SafeGradHFTrainer


def _model(seed):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=2))


def _trainer(model, safety, ref):
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=1,
                             use_cpu=True, report_to=[])
    return SafeGradHFTrainer(model=model, args=args, safety_dataset=[safety], reference_model=ref)


def test_padding_adds_nothing_to_the_kl():
    torch.manual_seed(0)
    s, r = torch.randn(2, 4, 32), torch.randn(2, 4, 32)
    pad_s = torch.cat([s, torch.randn(2, 3, 32)], 1)
    pad_r = torch.cat([r, torch.randn(2, 3, 32)], 1)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]] * 2)
    want = safegrad_kl_alignment_loss(s, r)
    assert torch.allclose(safegrad_kl_alignment_loss(pad_s, pad_r, attention_mask=mask), want)
    # an all-ones mask keeps the old scale (sum over positions, mean over the batch)
    assert torch.allclose(safegrad_kl_alignment_loss(s, r, attention_mask=torch.ones(2, 4)), want)


def test_kl_between_identical_bf16_logits_is_zero():
    torch.manual_seed(0)
    x = (torch.randn(2, 64, 32000) * 3).to(torch.bfloat16)
    student = x.clone().requires_grad_(True)
    loss = safegrad_kl_alignment_loss(student, x.clone())
    loss.backward()
    assert abs(loss.item()) < 1e-4                      # main: -0.022
    assert student.grad.abs().max().item() < 1e-6      # main: ~1e-3, pure rounding noise
    # fp32 inputs give the same value as before
    y = torch.randn(2, 4, 32)
    z = torch.randn(2, 4, 32)
    want = torch.nn.functional.kl_div(torch.log_softmax(y, -1), torch.softmax(z, -1), reduction="batchmean")
    assert torch.equal(safegrad_kl_alignment_loss(y, z), want)


def test_trainer_alignment_gradient_ignores_padding():
    ids = torch.randint(0, 32, (1, 5), generator=torch.Generator().manual_seed(1))
    plain = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    padded = {"input_ids": torch.cat([ids, torch.zeros(1, 3, dtype=torch.long)], 1),
              "attention_mask": torch.cat([torch.ones_like(ids), torch.zeros(1, 3, dtype=torch.long)], 1)}
    model, ref = _model(0), _model(1)
    tr = _trainer(model, plain, ref)
    g_plain = torch.cat([g.reshape(-1) for g in tr._alignment_gradient(model, plain)])
    g_padded = torch.cat([g.reshape(-1) for g in tr._alignment_gradient(model, padded)])
    assert torch.allclose(g_padded, g_plain, rtol=1e-4, atol=1e-6)


def test_a_failing_alignment_pass_raises():
    """A broken alignment batch must stop training, not silently drop the surgery."""
    model, ref = _model(0), _model(1)
    bad = {"input_ids": torch.full((1, 4), 999)}          # outside the 32-token vocabulary
    tr = _trainer(model, bad, ref)
    tr.current_gradient_accumulation_steps = 1
    model.train()
    ids = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(IndexError):
        tr.training_step(model, {"input_ids": ids, "labels": ids})


def test_sft_mode_without_labels_raises():
    model = _model(0)
    tr = _trainer(model, {"input_ids": torch.ones(1, 4, dtype=torch.long)}, None)
    with pytest.raises(ValueError, match="labels"):
        tr._alignment_gradient(model, {"input_ids": torch.ones(1, 4, dtype=torch.long)})


def test_reference_model_without_safety_dataset_raises():
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), use_cpu=True, report_to=[])
    with pytest.raises(ValueError, match="safety_dataset"):
        SafeGradHFTrainer(model=_model(0), args=args, reference_model=_model(1))


def test_train_runs_the_surgery_every_step_with_padded_batchencoding_safety_batches():
    """The real Trainer loop, with safety batches as a tokenizer's collator returns them."""
    from transformers import BatchEncoding
    ids = torch.randint(0, 32, (2, 6), generator=torch.Generator().manual_seed(2))
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    safety = BatchEncoding({"input_ids": ids, "attention_mask": mask})
    train = [{"input_ids": r, "attention_mask": torch.ones_like(r), "labels": r} for r in ids]
    model = _model(0)
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=1, max_steps=2,
                             use_cpu=True, report_to=[], save_strategy="no", logging_strategy="no")
    tr = SafeGradHFTrainer(model=model, args=args, train_dataset=train, safety_dataset=[safety],
                           reference_model=_model(1))
    calls = []
    surgery = tr._apply_global_surgery
    tr._apply_global_surgery = lambda m, g: (calls.append(1), surgery(m, g))
    tr.train()
    assert len(calls) == 2
