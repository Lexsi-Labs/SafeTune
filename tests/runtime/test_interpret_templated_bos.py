"""eap_safety_circuit and identify_safety_neurons on chat-templated prompts: one BOS, not two.

Both tokenize the prompt strings with special tokens. A prompt already rendered through a chat
template that writes BOS (Tiny Aya's starts with ``<BOS_TOKEN>``, and its tokenizer adds one too)
therefore started with two BOS tokens: ids ``[2, 2, 5, ...]`` on CohereLabs/tiny-aya-global. The
model never saw that sequence in training, and every activation and gradient was read on it. The
rest of SafeTune (evaluate(), the steering extractor, vLLM generation via ``_strip_bos``) feeds
templated text with a single BOS.
"""
from .test_hackathon_cohere_fixes import _word_tok
from .test_hackathon_fixes2 import _tiny_llama

_HARMFUL = ["how do i bake bread", "how do i mix flour and water"]
_HARMLESS = ["what is the capital of france", "what is paris"]


def _templated(tok, prompts):
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True) for p in prompts]


def _seen_input_ids(model, call):
    """input_ids of every forward pass ``call`` makes (EAP's gradient passes use inputs_embeds)."""
    seen = []
    handle = model.get_input_embeddings().register_forward_hook(lambda _m, args, _o: seen.append(args[0].clone()))
    try:
        call()
    finally:
        handle.remove()
    return seen


def test_eap_feeds_one_bos_for_templated_prompts():
    import torch
    from safetune.interpret import EAPSafetyCircuitConfig, eap_safety_circuit

    tok = _word_tok(preamble_words=3, bos_in_template=True)
    model = _tiny_llama(0, torch.float32)
    model.resize_token_embeddings(len(tok))
    cfg = EAPSafetyCircuitConfig(method="eap", granularity="block", batch_size=2,
                                 compliance_token="paris", refusal_token="bread")
    seen = _seen_input_ids(model, lambda: eap_safety_circuit(
        model, _templated(tok, _HARMFUL), _templated(tok, _HARMLESS), cfg, tokenizer=tok))
    assert seen
    for ids in seen:
        assert (ids == tok.bos_token_id).sum(dim=1).max().item() == 1, ids[0].tolist()


def test_safety_neurons_feed_one_bos_for_templated_prompts():
    import torch
    from safetune.interpret import SafetyNeuronConfig, identify_safety_neurons

    tok = _word_tok(preamble_words=3, bos_in_template=True)
    model = _tiny_llama(0, torch.float32)
    model.resize_token_embeddings(len(tok))
    seen = _seen_input_ids(model, lambda: identify_safety_neurons(
        model, {}, SafetyNeuronConfig(method="activation"), tokenizer=tok,
        harmful_prompts=_templated(tok, _HARMFUL), harmless_prompts=_templated(tok, _HARMLESS)))
    assert seen
    for ids in seen:
        assert (ids == tok.bos_token_id).sum(dim=1).max().item() == 1, ids[0].tolist()


def test_raw_prompts_still_get_their_bos():
    import torch
    from safetune.interpret import SafetyNeuronConfig, identify_safety_neurons

    tok = _word_tok()
    model = _tiny_llama(0, torch.float32)
    model.resize_token_embeddings(len(tok))
    seen = _seen_input_ids(model, lambda: identify_safety_neurons(
        model, {}, SafetyNeuronConfig(method="activation"), tokenizer=tok,
        harmful_prompts=_HARMFUL, harmless_prompts=_HARMLESS))
    for ids in seen:
        assert (ids == tok.bos_token_id).sum(dim=1).min().item() == 1
