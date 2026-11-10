"""A corrupted (harmless) batch longer than the clean (harmful) batch keeps its BOS and leading tokens.

The two batches are tokenized separately. The corrupted one used to be cut to the clean batch's length
from the left, which dropped the BOS and the first words of its longest prompt. The shorter batch is now
padded up instead, and nothing is truncated.
"""
import pytest

from .test_hackathon_cohere_fixes import _word_tok
from .test_hackathon_fixes2 import _tiny_llama

_HARMFUL = ["how do i bake bread", "how do i mix"]                              # 5 and 4 words
_HARMLESS = ["what is the capital of france and what is the capital", "what is paris"]   # 10 and 3 words


@pytest.mark.parametrize("side", ["left", "right"])
def test_longer_corrupted_batch_keeps_bos_and_leading_words(side):
    import torch
    from safetune.interpret import EAPSafetyCircuitConfig, eap_safety_circuit

    tok = _word_tok(bos_in_template=False)
    tok.padding_side = side
    model = _tiny_llama(0, torch.float32)
    model.resize_token_embeddings(len(tok))
    seen = []
    handle = model.register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append(kwargs["input_ids"].clone()) if "input_ids" in kwargs else None,
        with_kwargs=True)
    cfg = EAPSafetyCircuitConfig(method="eap", granularity="block", top_k_edges=1000, batch_size=2,
                                 compliance_token="paris", refusal_token="bread")
    eap_safety_circuit(model, _HARMFUL, _HARMLESS, cfg, tokenizer=tok)
    handle.remove()

    corrupted = seen[0]                      # the first forward of a batch is the corrupted pass
    bos = tok.bos_token_id
    assert (corrupted == bos).sum().item() == 2          # one BOS per row, none cut
    longest = _HARMLESS[0].split()
    real = [i for i in corrupted[0].tolist() if i != tok.pad_token_id]
    assert len(real) == len(longest) + 1                 # BOS plus every word of the longest prompt
