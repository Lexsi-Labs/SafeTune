"""Forwards that only read hidden states ask the model for the last position's logits only.

Each of these passes runs ``model(..., output_hidden_states=True)`` and never reads the logits,
yet the LM head still built a ``(batch, seq, vocab)`` tensor per batch. With a 262k vocabulary
that tensor outweighs everything else the pass keeps. The LM head must now see one position,
including on a peft-wrapped model, whose forward takes ``logits_to_keep`` through ``**kwargs``.
"""
import pytest
import torch

from tests.runtime.test_hackathon_cohere_fixes import _word_tok
from tests.runtime.test_hackathon_fixes2 import _tiny_llama

_TEXTS = ["how do i bake bread", "what is the capital of france paris"]


def _model_and_tok():
    tok = _word_tok()
    model = _tiny_llama(0, torch.float32).eval()
    model.resize_token_embeddings(len(tok))
    return model, tok


def _enc(tok, text):
    return dict(tok(text, return_tensors="pt"))


def _extractor_on_peft(model, tok):
    from peft import LoraConfig, get_peft_model
    from safetune.core.runtime.inference.vector_extraction import SteeringVectorExtractor, VectorExtractionConfig
    peft_model = get_peft_model(model, LoraConfig(target_modules=["q_proj"]))
    cfg = VectorExtractionConfig(target_layers=[1], pool_method="last_token", normalize=False, batch_size=2)
    SteeringVectorExtractor(peft_model, tok, cfg)._collect_activations(_TEXTS)


def _filters_hidden(model, tok):
    from safetune.core.data_compiler.filters import compute_hidden_states
    compute_hidden_states(model, tok, _TEXTS, device="cpu")


def _filters_larf(model, tok):
    from safetune.core.data_compiler.filters import filter_by_larf
    filter_by_larf([{"text": t} for t in _TEXTS], model, tok, safety_layers=[1, 2])


def _subspace(model, tok):
    from safetune.core.extras.subspace import _encode
    _encode(model, _TEXTS, tokenizer=tok)


def _prober_train(model, tok):
    from safetune.core.runtime.inference.safeswitch import SafetyProber
    SafetyProber(hidden_size=64).train(model, [_enc(tok, _TEXTS[0])], [_enc(tok, _TEXTS[1])])


def _runner_prefill(model, tok, monkeypatch):
    import safetune.core.runtime.inference.safeswitch as S
    monkeypatch.setattr(S, "prober_probs", lambda *a, **k: [0.0])
    S.SafeSwitchRunner(model, S.SafetyProber(hidden_size=64))._check_prompt_safety(**_enc(tok, _TEXTS[0]))


def _switch_prefill(model, tok):
    from safetune.interventions.steer.safeswitch import SafeSwitchModel
    SafeSwitchModel(model, hidden_size=64)._prefill_hidden_states(**_enc(tok, _TEXTS[0]))


@pytest.mark.parametrize("run", [_extractor_on_peft, _filters_hidden, _filters_larf, _subspace,
                                 _prober_train, _runner_prefill, _switch_prefill])
def test_lm_head_sees_one_position(run, monkeypatch):
    model, tok = _model_and_tok()
    positions = []
    handle = model.lm_head.register_forward_hook(lambda m, i, o: positions.append(o.shape[1]))
    try:
        if run is _runner_prefill:
            run(model, tok, monkeypatch)
        else:
            run(model, tok)
    finally:
        handle.remove()
    assert positions and set(positions) == {1}, positions
