"""RepNoise's per-layer harmful CE no longer builds a (batch, seq, vocab) logits tensor per hidden state.

`RepNoiseHFTrainer._harmful_ce` projected every position of every hidden state through the final norm
and the output embeddings before masking, so a step kept (n_layers + 1) full-vocabulary logits tensors
for backward. With Tiny Aya (262k vocabulary, 36 layers) that alone exceeds a 16 GB T4 at batch 1.
`masked_token_ce_from_hidden` projects only the positions whose label is counted and must return the
same loss and the same gradients as `masked_token_ce_loss` on the full projection.
"""
import pytest
import torch

from safetune.interventions.harden.repnoise import masked_token_ce_from_hidden, masked_token_ce_loss

V, D = 50, 16


def _case(seed, batch=3, seq=12):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, seq, D, generator=g, requires_grad=True)
    labels = torch.randint(0, V, (batch, seq), generator=g)
    labels[0, 5] = 0                                   # a real token id 0 is ignored by the reference too
    mask = torch.zeros(batch, seq, dtype=torch.long)
    mask[0, 3:10] = 1                                  # response span with a masked-logit boundary
    mask[1, 0:12] = 1                                  # fully unmasked row
    mask[2, 4:6] = 1                                   # short span
    mask[2, 8] = 1                                     # isolated position
    return hidden, labels, mask


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_loss_and_gradients_as_full_projection(seed):
    torch.manual_seed(seed)
    norm = torch.nn.LayerNorm(D)
    head = torch.nn.Linear(D, V, bias=False)
    project = lambda x: head(norm(x))  # noqa: E731

    hidden, labels, mask = _case(seed)
    ref = masked_token_ce_loss(project(hidden), labels, mask)
    ref.backward()
    g_ref = (hidden.grad.clone(), head.weight.grad.clone(), norm.weight.grad.clone())

    hidden.grad = None; head.weight.grad = None; norm.weight.grad = None
    new = masked_token_ce_from_hidden(hidden, project, labels, mask)
    new.backward()

    assert torch.allclose(new, ref, atol=1e-6), (new.item(), ref.item())
    for a, b in zip((hidden.grad, head.weight.grad, norm.weight.grad), g_ref):
        assert torch.allclose(a, b, atol=1e-6)


def test_projects_only_counted_positions():
    head = torch.nn.Linear(D, V, bias=False)
    seen = []
    def project(x):
        seen.append(tuple(x.shape))
        return head(x)
    hidden, labels, mask = _case(0)
    masked_token_ce_from_hidden(hidden, project, labels, mask)
    (rows,) = {s[0] for s in seen}
    shift = (labels[..., 1:] * mask[..., 1:]) != 0
    assert rows == int((shift & mask[..., :-1].bool()).sum())     # not batch * seq
    assert rows < hidden.shape[0] * (hidden.shape[1] - 1)


def test_trainer_harmful_ce_matches_full_projection_on_a_real_model():
    """The trainer's L_harmful is unchanged, and the LM head never sees a full (batch, seq) slab."""
    import types
    from transformers import LlamaConfig, LlamaForCausalLM
    from safetune.interventions.harden.repnoise import RepNoiseHFTrainer as T

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                         num_attention_heads=4, num_key_value_heads=4))
    ids = torch.randint(1, 64, (2, 10))
    mask = torch.zeros(2, 10, dtype=torch.long); mask[:, 4:] = 1
    out = model(input_ids=ids, output_hidden_states=True)

    ref = masked_token_ce_loss(out.logits, ids, mask)
    for h in out.hidden_states:
        ref = ref + masked_token_ce_loss(model.lm_head(model.model.norm(h)), ids, mask)
    ref = ref / len(out.hidden_states)

    rows = []
    handle = model.lm_head.register_forward_hook(lambda m, i, o: rows.append(i[0].shape[:-1]))
    try:
        self = types.SimpleNamespace(_get_output_embeddings=T._get_output_embeddings, _get_final_norm=T._get_final_norm)
        got = T._harmful_ce(self, model, out, ids, mask)
    finally:
        handle.remove()

    assert torch.allclose(got, ref, atol=1e-5), (got.item(), ref.item())
    assert len(rows) == len(out.hidden_states)
    assert all(len(r) == 1 and r[0] < ids.numel() for r in rows), rows    # (positions,), never (batch, seq)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_sum_does_not_overflow(dtype):
    """A real step counts thousands of tokens; their CE summed in fp16 passes 65504 and turns to inf."""
    torch.manual_seed(0)
    v, d = 4096, 32
    norm, head = torch.nn.LayerNorm(d), torch.nn.Linear(d, v, bias=False)
    hidden = torch.randn(4, 2048, d)
    labels = torch.randint(1, v, (4, 2048))
    mask = torch.zeros(4, 2048, dtype=torch.bool)
    mask[:, 100:] = True                               # ~7.8k counted tokens at CE ~8.3
    truth = masked_token_ce_loss(head(norm(hidden)), labels, mask)

    norm, head = norm.to(dtype), head.to(dtype)
    got = masked_token_ce_from_hidden(hidden.to(dtype), lambda x: head(norm(x)), labels, mask)
    assert got.dtype == dtype
    assert torch.isfinite(got), got
    assert abs(got.float().item() - truth.item()) < 0.05 * truth.item(), (got.item(), truth.item())
