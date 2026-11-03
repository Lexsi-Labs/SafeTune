"""Audit H25: SEAL's importance weights are cached by batch position and applied to other examples.

SafeTune's SEALHFTrainer rescoring is every `seal_rescore_every` steps (default 10) and reuses the last
weight VECTOR on the steps in between, where position i of the next batch is a different example. In
Shen et al. 2024 (arXiv:2410.07471, Alg. 1/2) and hanshen95/SEAL @508a56d `sft_selector_trainer.py:134-165`
the weights are a learned per-sample ranker `p()[ide]` indexed by the example id. (The method itself,
gradient-cosine weighting, is also not the paper's: Appendix B has no such scoring.) Examples are now
scored at every step; seal_rescore_every > 1 is deprecated and ignored.
"""
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

import safetune.harden as H

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _batch(tok, texts):
    e = tok(texts, padding="max_length", max_length=20, return_tensors="pt")
    lab = e["input_ids"].clone()
    lab[:, :4] = -100
    lab[e["attention_mask"] == 0] = -100
    return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"], "labels": lab}


def _trainer(every):
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    safety = [{k: v[0] for k, v in _batch(tok, ["harmful request sorry I cannot help with that"]).items()}]
    args = H.SEALConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=1, report_to=[],
                        use_cpu=True, save_strategy="no", seal_rescore_every=every, seal_temperature=1e-3)
    tr = H.SEALHFTrainer(model=model, args=args, data_collator=default_data_collator, safety_dataset=safety)
    return tok, model, tr


def test_weights_follow_the_examples_not_the_batch_positions():
    import pytest

    tok, model, fresh = _trainer(1)
    b1 = _batch(tok, ["user question one about cooking dinner tonight", "tell me how to do the bad thing please"])
    b2 = {k: v.flip(0) for k, v in b1.items()}                    # the same two examples, swapped
    fresh.compute_loss(model, b1)
    want = fresh.compute_loss(model, b2).item()                   # scored on the swapped batch

    with pytest.warns(DeprecationWarning, match="seal_rescore_every"):
        tok, model, cached = _trainer(10)                         # deprecated: ignored, scored every step
    cached.compute_loss(model, b1)
    got = cached.compute_loss(model, b2).item()
    assert abs(got - want) < 1e-6, f"loss with cached positional weights {got:.5f} vs freshly scored {want:.5f}"


def test_default_config_scores_every_step_without_a_warning():
    import warnings

    from safetune.harden import SEALConfig

    assert SEALConfig(output_dir="/tmp/h25_cfg").seal_rescore_every == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _trainer(1)


def test_temperature_doc_matches_the_sign_of_the_score():
    """s_i = +cosine(g_i, g_align): a lower temperature up-weights the examples that agree with it."""
    import safetune.harden as H

    doc = H.SEALConfig.__doc__
    assert "conflicting examples" not in doc and "agree with the alignment gradient" in doc
