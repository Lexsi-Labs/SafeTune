"""DeepRefusalTrainer must give the same gradient with and without gradient checkpointing.

Before the fix, the ablation hooks were switched off inside ``compute_loss``
(and the Bernoulli gate was redrawn on every call), so the checkpoint
recomputation during ``backward()`` ran without the ablation: the reentrant
mode silently produced a different gradient and the non-reentrant mode raised
``CheckpointError``.
"""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from safetune.interventions.harden.deeprefusal import DeepRefusalConfig, DeepRefusalTrainer  # noqa: E402

HIDDEN = 32


def _batch() -> dict:
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 128, (1, 16), generator=g)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _grads(tmp_path, checkpointing, ablation_prob=1.0):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=128, hidden_size=HIDDEN, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = transformers.LlamaForCausalLM(cfg)
    if checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": checkpointing == "reentrant"})
    args = DeepRefusalConfig(output_dir=str(tmp_path), ablation_prob=ablation_prob,
                             report_to=[], use_cpu=True)
    harmful, benign = _batch(), _batch()
    benign["input_ids"] = benign["labels"] = (harmful["input_ids"] + 7) % 128
    trainer = DeepRefusalTrainer(model=model, args=args, refusal_direction=torch.randn(HIDDEN),
                                 harmful_dataset=[harmful], benign_dataset=[benign])
    trainer.current_gradient_accumulation_steps = 1  # normally set by train()
    random.seed(0)
    loss = trainer.training_step(trainer.model, dict(harmful))
    grads = torch.cat([p.grad.flatten() for p in trainer.model.parameters() if p.grad is not None])
    assert all(not h.active for h in trainer._hooks)
    return float(loss), grads


@pytest.mark.parametrize("mode", ["reentrant", "non-reentrant"])
def test_gradient_matches_without_checkpointing(tmp_path, mode):
    loss_ref, g_ref = _grads(tmp_path / "ref", None)
    loss, g = _grads(tmp_path / mode, mode)
    assert loss == pytest.approx(loss_ref, rel=1e-5)
    assert torch.allclose(g, g_ref, rtol=1e-4, atol=1e-6)


def test_ablation_actually_changes_the_gradient(tmp_path):
    # Guard against a vacuous pass: with ablation off the gradient must differ.
    _, g_on = _grads(tmp_path / "on", None, ablation_prob=1.0)
    _, g_off = _grads(tmp_path / "off", None, ablation_prob=0.0)
    assert not torch.allclose(g_on, g_off, rtol=1e-4, atol=1e-6)


def test_gate_is_drawn_once_per_step():
    from safetune.interventions.harden.deeprefusal import _AblationHook

    hook = _AblationHook(torch.ones(HIDDEN) / HIDDEN ** 0.5, ablation_prob=0.5)
    hook.set_active(True)
    h = torch.randn(1, 3, HIDDEN)
    first = hook(None, None, (h,))[0]
    for _ in range(20):  # later calls in the same step reuse the decision
        assert torch.equal(hook(None, None, (h,))[0], first)
    hook.set_active(False)
    assert hook.gate is None
