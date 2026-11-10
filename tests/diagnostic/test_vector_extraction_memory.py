"""SteeringVectorExtractor keeps one pooled vector per prompt, not every token's hidden state.

The forward hook stored each batch's full ``(batch, seq, hidden)`` hidden state for every target
layer on the CPU until the whole prompt set had run, and pooled only afterwards. Memory grew with
prompts * layers * sequence length. With a chat template that adds a long preamble (Tiny Aya: about
366 tokens, 36 layers, hidden 2048, bf16) 200 prompts hold about 11 GB that last_token pooling never
reads. extract_refusal_direction and extract_caa_vectors both run through this class.
"""
import pytest
import torch

from safetune.core.runtime.inference.vector_extraction import SteeringVectorExtractor, VectorExtractionConfig

from tests.runtime.test_hackathon_cohere_fixes import _word_tok
from tests.runtime.test_hackathon_fixes2 import _tiny_llama

_SAFE = ["how do i bake bread", "how do i mix flour and water", "do i bake flour"]
_UNSAFE = ["what is the capital of france", "what is paris", "is paris the capital"]


def _reference(model, tok, prompts, layers, pool):
    """The same statistic computed one prompt at a time, with no padding."""
    out = {li: [] for li in layers}
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        for li in layers:
            h = hs[li + 1][0]                      # output of decoder block li
            out[li].append({"last_token": h[-1], "mean": h.mean(0), "max": h.max(0).values}[pool])
    return {li: torch.stack(v).float().mean(0) for li, v in out.items()}


@pytest.mark.parametrize("pool", ["last_token", "mean", "max"])
def test_only_pooled_vectors_are_held_and_the_result_is_unchanged(pool):
    tok = _word_tok(preamble_words=40)
    model = _tiny_llama(0, torch.float32).eval()
    model.resize_token_embeddings(len(tok))
    layers = [1, 4]
    ext = SteeringVectorExtractor(model, tok, VectorExtractionConfig(target_layers=layers, pool_method=pool,
                                                                     normalize=False, batch_size=2))
    held = []
    hook = model.model.layers[4].register_forward_hook(
        lambda *_: held.append(max((t.dim() for acts in ext._activations.values() for t in acts), default=0)))
    try:
        got = ext.extract(_SAFE, _UNSAFE)
    finally:
        hook.remove()

    assert max(held) <= 2, f"a {max(held)}-D hidden state per batch is held during extraction"
    safe, unsafe = _reference(model, tok, _SAFE, layers, pool), _reference(model, tok, _UNSAFE, layers, pool)
    for li in layers:
        assert torch.allclose(got[li], safe[li] - unsafe[li], atol=1e-5), li
