"""CAST with more than one sequence per prompt (beam search, num_return_sequences).

CASTModel.generate gates each prompt and steers only the rows whose gate fired, through a per-row
mask with one entry per PROMPT. generate() expands the batch to ``batch * num_beams`` (or
``* num_return_sequences``) rows, so the mask no longer matched the hidden state and generation
crashed: "The size of tensor a (4) must match the size of tensor b (2)". A single prompt worked
only because no mask is built for it.
"""
import pytest
import torch

from safetune.interventions.steer.cast import CASTCondition
from safetune.steer import CASTModel

from _steer_common import HARMFUL, build


def _cast(model, fires):
    """A CAST wrapper whose gate fires for the prompts listed in ``fires`` (by row)."""
    cond = CASTCondition(condition_vector=torch.randn(model.config.hidden_size), condition_layer=2,
                         threshold=0.0, comparator="smaller", pool="last")
    cast = CASTModel(model, {3: 5.0 * torch.ones(model.config.hidden_size)}, condition=cond)
    cast._gate_fires = lambda input_ids, **kw: (list(fires), [0.0] * len(fires))
    return cast


def _enc(tok, prompts):
    tok.padding_side = "left"
    return tok(prompts, return_tensors="pt", padding=True)


@pytest.mark.parametrize("gen_kwargs", [dict(num_beams=2), dict(num_return_sequences=2, do_sample=True)],
                         ids=["beams", "return_sequences"])
def test_generate_runs_with_several_sequences_per_prompt(gen_kwargs):
    model, tok = build()
    enc = _enc(tok, HARMFUL[:2])
    out = _cast(model, [True, False]).generate(**enc, max_new_tokens=3, pad_token_id=tok.pad_token_id, **gen_kwargs)
    assert out.shape[0] == 2 * gen_kwargs.get("num_return_sequences", 1)


def test_each_beam_follows_its_own_prompts_gate():
    """With beams, the prompt whose gate did not fire generates what the unsteered model generates."""
    model, tok = build()
    enc = _enc(tok, HARMFUL[:2])
    kw = dict(max_new_tokens=4, num_beams=2, do_sample=False, pad_token_id=tok.pad_token_id)
    plain = model.generate(**enc, **kw)
    all_on = _cast(model, [True, True]).generate(**enc, **kw)
    mixed = _cast(model, [True, False]).generate(**enc, **kw)
    assert torch.equal(mixed[1], plain[1])        # row 1: gate off -> unsteered
    assert torch.equal(mixed[0], all_on[0])       # row 0: gate on  -> steered
    assert not torch.equal(all_on[0], plain[0])   # and steering does change the output
