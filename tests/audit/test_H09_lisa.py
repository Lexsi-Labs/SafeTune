"""Audit H09: Lisa's proximal term is counted gradient_accumulation_steps times per optimizer step.

LisaHFTrainer.compute_loss adds rho/2*||w - w_other||^2 to the loss inside every micro-step.
transformers' training_step divides the loss by gradient_accumulation_steps only when it has no
`num_items_in_batch` (decoder LMs pass one, and then the CE is normalised by the whole step's token
count instead). The penalty is not scaled either way, so with GA=4 its gradient is 4x rho*(w - w_other).
The official trainer adds the penalty and the CE to the same unscaled backward, so their ratio is rho;
here the ratio becomes GA*rho (the penalty is GA times too strong relative to the task loss).
"""
import pytest
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM,
                          TrainerCallback)

from safetune.harden.lisa import LisaConfig, LisaHFTrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _penalty_grad_norm(ga: int, tmp_path) -> float:
    tok = AutoTokenizer.from_pretrained(TINY)
    ids = tok("hello world, how are you today?", return_tensors="pt")["input_ids"][0]
    rows = [{"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}] * 8
    out = {}
    for rho in (0.0, 1.0):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(TINY)
        args = LisaConfig(output_dir=str(tmp_path), per_device_train_batch_size=1,
                          gradient_accumulation_steps=ga, max_steps=1, report_to=[], use_cpu=True,
                          lisa_rho=rho, lisa_warmup_steps=0, lisa_alignment_step=0)
        tr = LisaHFTrainer(model=model, args=args, train_dataset=rows,
                           data_collator=lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]})
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))      # move off the anchor so the penalty is non-zero
        tr._lisa.current_step = 1
        model.train()
        model.zero_grad()
        batch = {k: v.unsqueeze(0) for k, v in rows[0].items()}
        for _ in range(ga):
            tr.training_step(model, batch, num_items_in_batch=torch.tensor(ga * ids.numel()))
        out[rho] = [p.grad.clone() for p in model.parameters()]
        drift = torch.sqrt(sum(((p - tr._lisa.alignment_weights[n]) ** 2).sum()
                               for n, p in model.named_parameters()))
    diff = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(out[1.0], out[0.0])))
    return float(diff / drift)          # expected ratio is rho = 1


def test_lisa_penalty_is_applied_once_per_optimizer_step(tmp_path):
    r1 = _penalty_grad_norm(1, tmp_path)
    r4 = _penalty_grad_norm(4, tmp_path)
    assert abs(r1 - 1.0) < 0.05, r1
    assert abs(r4 - 1.0) < 0.05, f"GA=4 penalty gradient is {r4:.2f}x rho*(w-w_anchor), expected 1x"


# Review of #111: HF divides the loss by the accumulation steps itself unless the loss was already
# normalised by num_items_in_batch (`Trainer.training_step`). On that branch the penalty must not be
# divided again, or it ends up 1/GA too weak. Measured through `train()`, so the Trainer decides the
# branch and the window size (6 rows at GA=4: windows of 4 and 2 micro-steps).

_CFG = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                   num_attention_heads=2, num_key_value_heads=2)


class _NoLossKwargs(LlamaForCausalLM):
    accepts_loss_kwargs = False          # ~50 transformers model classes set this


class _NoKwargsForward(torch.nn.Module):
    """forward without **kwargs: the Trainer cannot pass num_items_in_batch."""

    def __init__(self):
        super().__init__()
        self.m = LlamaForCausalLM(_CFG)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class _Grads(TrainerCallback):
    def __init__(self, model):
        self.model, self.steps = model, []

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self.steps.append([p.grad.clone() for p in self.model.parameters()])


def _train_ratios(make, tmp_path):
    ids = torch.arange(1, 9)
    rows = [{"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}] * 6
    grads = {}
    for rho in (0.0, 1.0):
        torch.manual_seed(0)
        model = make()
        args = LisaConfig(output_dir=str(tmp_path), per_device_train_batch_size=1,
                          gradient_accumulation_steps=4, num_train_epochs=1, learning_rate=0.0,
                          max_grad_norm=0.0, report_to=[], use_cpu=True, save_strategy="no",
                          lisa_rho=rho, lisa_warmup_steps=0, lisa_alignment_step=0)
        cb = _Grads(model)
        tr = LisaHFTrainer(model=model, args=args, train_dataset=rows, callbacks=[cb],
                           data_collator=lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]})
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))
        tr._lisa.current_step = 1           # past warmup from the first micro-step
        tr.train()
        grads[rho] = cb.steps
        drift = torch.sqrt(sum(((p - tr._lisa.alignment_weights[n]) ** 2).sum()
                               for n, p in model.named_parameters()))
    assert len(grads[1.0]) == 2
    return [float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(g1, g0))) / drift)
            for g1, g0 in zip(grads[1.0], grads[0.0])]


@pytest.mark.parametrize("make", [lambda: LlamaForCausalLM(_CFG), lambda: _NoLossKwargs(_CFG),
                                  _NoKwargsForward], ids=["loss_kwargs", "accepts_loss_kwargs_false",
                                                          "forward_without_kwargs"])
def test_lisa_penalty_counts_once_per_optimizer_step_on_both_trainer_branches(make, tmp_path):
    ratios = _train_ratios(make, tmp_path)
    assert all(abs(r - 1.0) < 0.05 for r in ratios), \
        f"penalty gradient / (rho * drift) per optimizer step = {ratios}, expected 1.0"
