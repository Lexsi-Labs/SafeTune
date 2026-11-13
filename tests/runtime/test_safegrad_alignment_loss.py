"""SafeGrad alignment pass: padding (#81)."""
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
