"""eap_safety_circuit scores must not depend on the tokenizer's padding side.

The harmful (clean) and harmless (corrupted) batches are tokenized separately and compared position
by position. With a right-padded tokenizer (the training default; Tiny Aya's is right-padded) prompts
of different lengths are aligned at their START, so the final token of a short harmless prompt is
compared with a mid-prompt token of the harmful one, and the position where the metric is read
(the last real token) is patched from a pad position. With left padding the ends line up, which is
what a metric read at the final token needs. The same model and prompts gave scores that correlate
at 0.58 between the two padding sides.
"""
import pytest

from .test_hackathon_cohere_fixes import _word_tok
from .test_hackathon_fixes2 import _tiny_llama

_HARMFUL = ["how do i bake bread", "how do i mix flour and water"]       # 5 and 7 words
_HARMLESS = ["what is the capital of france", "what is paris"]           # 6 and 3 words


def _scores(side, method):
    import torch
    from safetune.interpret import EAPSafetyCircuitConfig, eap_safety_circuit

    tok = _word_tok(bos_in_template=False)
    tok.padding_side = side
    model = _tiny_llama(0, torch.float32)
    model.resize_token_embeddings(len(tok))
    cfg = EAPSafetyCircuitConfig(method=method, granularity="block", top_k_edges=1000, batch_size=2,
                                 ig_steps=2, compliance_token="paris", refusal_token="bread")
    info = eap_safety_circuit(model, _HARMFUL, _HARMLESS, cfg, tokenizer=tok)
    assert tok.padding_side == side                       # the caller's tokenizer is left as it was
    return info.safety_units.activation_correlation


@pytest.mark.parametrize("method", ["eap", "eap-ig"])
def test_scores_are_the_same_for_a_left_and_a_right_padded_tokenizer(method):
    right, left = _scores("right", method), _scores("left", method)
    assert right.keys() == left.keys()
    for name in right:
        assert right[name] == pytest.approx(left[name], rel=1e-4, abs=1e-7), name


def test_batches_are_aligned_at_their_last_token():
    from safetune.interpret import EAPSafetyCircuitConfig
    from safetune.instrumentation.interpret.eap import _tokenize_batch

    tok = _word_tok(bos_in_template=False)
    tok.padding_side = "right"
    enc, last_idx = _tokenize_batch(tok, _HARMFUL, EAPSafetyCircuitConfig(), "cpu")
    assert last_idx.tolist() == [enc["input_ids"].shape[1] - 1] * 2
