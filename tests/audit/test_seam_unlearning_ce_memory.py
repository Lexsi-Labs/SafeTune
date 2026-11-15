"""SEAM's layer-wise unlearning loss projects only the counted positions through the LM head.

`_unlearning_loss` read out a next-token CE from every hidden state by projecting all of it through
the final norm and the LM head and upcasting to fp32, so a step kept (n_layers + 1) fp32
(batch, seq, vocab) tensors for backward. The value must not change.
"""
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from safetune.interventions.harden.seam import _unlearning_loss


def _full_projection_reference(model, batch):
    """The loss as computed before: every position projected, then CE with ignore_index=-100."""
    def ce(logits, labels):
        return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
    total = ce(out.logits, batch["labels"])
    for h in out.hidden_states:
        total = total + ce(model.lm_head(model.model.norm(h)), batch["labels"])
    return -torch.log(total / len(out.hidden_states) + 1.0)


def test_same_loss_and_grads_and_no_full_slab():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                         num_attention_heads=4, num_key_value_heads=4))
    ids = torch.randint(1, 64, (2, 12))
    labels = ids.clone()
    labels[:, :7] = -100                                  # prompt tokens do not count
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}

    ref = _full_projection_reference(model, batch)
    ref.backward()
    g_ref = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    shapes = []
    handle = model.lm_head.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))
    try:
        got = _unlearning_loss(model, batch)
    finally:
        handle.remove()
    got.backward()

    assert torch.allclose(got, ref, atol=1e-6), (got.item(), ref.item())
    for a, b in zip((p.grad for p in model.parameters()), g_ref):
        assert torch.allclose(a, b, atol=1e-6)
    # first call is the model's own forward; the per-layer read-outs see only the 2 x 5 counted positions
    assert shapes[0] == (2, 12, 64)
    assert shapes[1:] == [(10, 64)] * 4, shapes
