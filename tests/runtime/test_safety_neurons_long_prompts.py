"""identify_safety_neurons(method="activation") on prompts longer than activation_max_tokens.

The same fault eap_safety_circuit had (test_eap_long_prompts): prompts were cut from the right, so
with a chat template whose preamble is longer than the default activation_max_tokens=64 (Tiny Aya:
~366 tokens) every prompt kept only the start of the preamble. The harmful and harmless sets were
then the same tokens, every contrast score was exactly 0, and the call raised "the selection scored
no unit" without naming the cause.
"""
import logging

from .test_hackathon_cohere_fixes import _word_tok
from .test_hackathon_fixes2 import _tiny_llama

_HARMFUL = ["how do i bake bread", "how do i mix flour and water"]
_HARMLESS = ["what is the capital of france", "what is paris"]


def _templated(tok, prompts):
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True) for p in prompts]


def test_long_templated_prompts_keep_their_end_and_score(caplog):
    import torch
    from safetune.interpret import SafetyNeuronConfig, identify_safety_neurons

    tok = _word_tok(preamble_words=100, bos_in_template=False)
    model = _tiny_llama(0, torch.float32)
    model.resize_token_embeddings(len(tok))
    with caplog.at_level(logging.WARNING, logger="safetune.instrumentation.interpret.safety_neurons"):
        report = identify_safety_neurons(model, {}, SafetyNeuronConfig(method="activation"), tokenizer=tok,
                                         harmful_prompts=_templated(tok, _HARMFUL),
                                         harmless_prompts=_templated(tok, _HARMLESS))

    scores = [s for units in report.per_layer.values() for _, s in units]
    assert scores and any(s != 0.0 for s in scores)
    assert "4 of 4 prompts are longer than activation_max_tokens=64" in caplog.text
    assert tok.truncation_side == "right"              # the caller's tokenizer is left as it was


def test_short_prompts_are_unchanged_and_do_not_warn(caplog):
    import torch
    from safetune.interpret import SafetyNeuronConfig, identify_safety_neurons

    tok = _word_tok()
    model = _tiny_llama(0, torch.float32)
    model.resize_token_embeddings(len(tok))
    with caplog.at_level(logging.WARNING, logger="safetune.instrumentation.interpret.safety_neurons"):
        identify_safety_neurons(model, {}, SafetyNeuronConfig(method="activation"), tokenizer=tok,
                                harmful_prompts=_HARMFUL, harmless_prompts=_HARMLESS)
    assert "longer than" not in caplog.text
