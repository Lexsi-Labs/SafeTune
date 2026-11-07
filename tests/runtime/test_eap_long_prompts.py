"""eap_safety_circuit on prompts longer than max_seq_len (a chat template with a long preamble).

The logit diff is read at the last real token, which should be the end of the prompt, where the
answer starts. Prompts were cut from the right, so with Tiny Aya's ~366-token preamble and the
default max_seq_len=64 every prompt kept only the start of the preamble. The harmful and harmless
batches were then the same tokens and every attribution score was 0.
"""
import logging

from .test_hackathon_cohere_fixes import _word_tok
from .test_hackathon_fixes2 import _tiny_llama

_HARMFUL = ["how do i bake bread", "how do i mix flour and water"]
_HARMLESS = ["what is the capital of france", "what is paris"]


def _templated(tok, prompts):
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True) for p in prompts]


def test_eap_keeps_the_end_of_a_long_prompt():
    from safetune.interpret import EAPSafetyCircuitConfig
    from safetune.instrumentation.interpret.eap import _tokenize_batch

    tok = _word_tok(preamble_words=100, bos_in_template=False)
    enc, last_idx = _tokenize_batch(tok, _templated(tok, _HARMFUL), EAPSafetyCircuitConfig(), "cpu")

    assert enc["input_ids"].shape[1] == 64
    last_tokens = [tok.convert_ids_to_tokens(int(enc["input_ids"][i, j])) for i, j in enumerate(last_idx)]
    assert last_tokens == ["<|assistant|>", "<|assistant|>"]


def test_eap_scores_are_not_all_zero_on_long_templated_prompts(caplog):
    import torch
    from safetune.interpret import EAPSafetyCircuitConfig, eap_safety_circuit

    tok = _word_tok(preamble_words=100, bos_in_template=False)
    model = _tiny_llama(0, torch.float32)
    model.resize_token_embeddings(len(tok))
    cfg = EAPSafetyCircuitConfig(method="eap", granularity="block", top_k_edges=4, batch_size=2,
                                 compliance_token="paris", refusal_token="bread")
    with caplog.at_level(logging.WARNING, logger="safetune.instrumentation.interpret.eap"):
        info = eap_safety_circuit(model, _templated(tok, _HARMFUL), _templated(tok, _HARMLESS),
                                  cfg, tokenizer=tok)

    scores = info.safety_units.activation_correlation
    assert any(v != 0.0 for v in scores.values()), scores
    assert "4 of 4 prompts are longer than max_seq_len=64" in caplog.text
