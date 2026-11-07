"""Audit E04: the HarmBench classifier template follows the classifier model, on every judge path.

Official: centerforaisafety/HarmBench@8e1604d. Llama-2 classifier -> LLAMA2_CLS_PROMPT['prompt'] (eval_utils.py:309-330, used at :408);
Mistral val classifier -> MISTRAL_CLS_PROMPT['prompt'] (notebooks/run_classifier_val_set.ipynb:55, chosen at :206). Mock models only."""
import importlib
import json
import types

import pytest
import torch

from safetune.instrumentation.evaluate.judges import _templates as T

J = importlib.import_module("safetune.instrumentation.evaluate.judges")
E = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")
LLAMA, MISTRAL = "cais/HarmBench-Llama-2-13b-cls", "cais/HarmBench-Mistral-7b-val-cls"
REFS = "/home/deeraj/lexsi/safetune/refs/HarmBench"


def test_the_template_belongs_to_the_model_and_an_unknown_id_raises():
    assert T.harmbench_template(LLAMA) is T._HARMBENCH_LLAMA2_PROMPT and "<<SYS>>" in T.harmbench_template(LLAMA)
    assert T.harmbench_template(MISTRAL) is T._HARMBENCH_MISTRAL_PROMPT and "<<SYS>>" not in T.harmbench_template(MISTRAL)
    for t in (T.harmbench_template(LLAMA), T.harmbench_template(MISTRAL)):
        assert t.startswith("[INST]") and t.endswith("Answer: [/INST]") and "{behavior}" in t and "{generation}" in t
    for bad in ("cais/HarmBench-Llama-2-13b-cls-multimodal-behaviors", "my/local-copy", "", None):
        with pytest.raises(ValueError, match="no HarmBench classifier template"):
            T.harmbench_template(bad)


def test_the_templates_are_the_official_ones():
    try:
        eu = open(f"{REFS}/eval_utils.py").read()
        nb = json.load(open(f"{REFS}/notebooks/run_classifier_val_set.ipynb"))
    except OSError:
        pytest.skip("HarmBench clone not present")
    ns = {}
    exec(eu[eu.index("LLAMA2_CLS_PROMPT = {"):eu.index("\n}\n", eu.index("LLAMA2_CLS_PROMPT = {")) + 3], ns)
    assert T._HARMBENCH_LLAMA2_PROMPT == ns["LLAMA2_CLS_PROMPT"]["prompt"]
    src = "".join(nb["cells"][0]["source"])
    i = src.index("MISTRAL_CLS_PROMPT = {")
    ns = {}
    exec(src[i:src.index("\n}\n", i) + 3], ns)
    assert T._HARMBENCH_MISTRAL_PROMPT == ns["MISTRAL_CLS_PROMPT"]["prompt"]


class _Tok:
    """Records the text it is asked to encode; decodes to a canned answer."""
    pad_token_id = 0

    def __init__(self):
        self.prompts = []

    def __call__(self, text, **kw):
        self.prompts.append(text)
        ids = torch.ones(1, 5, dtype=torch.long)
        out = types.SimpleNamespace(**{"input_ids": ids})
        out.to = lambda device: {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return out

    def decode(self, ids, **kw):
        return ids[0] if isinstance(ids, list) and ids and isinstance(ids[0], str) else "no"

    def encode(self, text, **kw):
        return [text]                  # the 512-token clip is a no-op here


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, attention_mask=None, **kw):
        return torch.cat([input_ids, torch.ones(1, 1, dtype=torch.long)], dim=1)


@pytest.mark.parametrize("model_id,expected", [(LLAMA, T._HARMBENCH_LLAMA2_PROMPT), (MISTRAL, T._HARMBENCH_MISTRAL_PROMPT)])
def test_hfjudge_uses_the_template_of_its_model(model_id, expected):
    from safetune.core.eval.pipeline.scorer import HFJudge, HFJudgeConfig
    j = HFJudge(HFJudgeConfig(model=model_id))
    j._tokenizer, j._model = _Tok(), _Model()
    j._judge_one("PLACEHOLDER behavior", "PLACEHOLDER generation")
    assert j._tokenizer.prompts == [expected.format(behavior="PLACEHOLDER behavior", generation="PLACEHOLDER generation")]


def test_hfjudge_with_an_unknown_model_raises_unless_a_template_is_given():
    from safetune.core.eval.pipeline.scorer import HFJudge, HFJudgeConfig
    j = HFJudge(HFJudgeConfig(model="my/local-copy"))
    j._tokenizer, j._model = _Tok(), _Model()
    with pytest.raises(ValueError, match="no HarmBench classifier template"):
        j._judge_one("b", "g")
    j2 = HFJudge(HFJudgeConfig(model="my/local-copy", template="B={behavior} G={generation}"))
    j2._tokenizer, j2._model = _Tok(), _Model()
    j2._judge_one("b", "g")
    assert j2._tokenizer.prompts == ["B=b G=g"]


@pytest.mark.parametrize("model_id,expected", [(LLAMA, T._HARMBENCH_LLAMA2_PROMPT), (MISTRAL, T._HARMBENCH_MISTRAL_PROMPT)])
def test_evaluate_harmbench_judge_uses_the_template_of_its_model(monkeypatch, model_id, expected):
    seen = []
    monkeypatch.setattr(E, "_suite_judge", lambda name: {"model_id": model_id, "max_new_tokens": 1, "max_model_len": 4096})
    monkeypatch.setattr(E, "get_harmbench_judge", lambda: (_Tok(), _Model()))
    monkeypatch.setattr(E, "_run_judge_batch", lambda tok, model, formatted, **kw: (seen.extend(formatted), ["no"] * len(formatted))[1])
    scorer = E._build_judge_scorer("harmbench")
    assert scorer.score([E._pack_pair("PLACEHOLDER behavior", "PLACEHOLDER generation")]) == [0.0]
    assert seen == [expected.format(behavior="PLACEHOLDER behavior", generation="PLACEHOLDER generation")]


def test_evaluate_harmbench_judge_with_an_unknown_model_raises(monkeypatch):
    monkeypatch.setattr(E, "_suite_judge", lambda name: {"model_id": "my/local-copy", "max_new_tokens": 1})
    monkeypatch.setattr(E, "get_harmbench_judge", lambda: pytest.fail("loaded a model before the template was resolved"))
    with pytest.raises(ValueError, match="no HarmBench classifier template"):
        E._build_judge_scorer("harmbench")


@pytest.mark.parametrize("model_id,expected", [(LLAMA, T._HARMBENCH_LLAMA2_PROMPT), (MISTRAL, T._HARMBENCH_MISTRAL_PROMPT)])
def test_run_judge_registry_template_follows_the_model(monkeypatch, model_id, expected):
    monkeypatch.setattr(J, "_judge_spec", lambda key: {"model_id": model_id})
    template_fn = J._JUDGE_CONFIG["harmbench"][0]
    assert template_fn("PLACEHOLDER behavior", "PLACEHOLDER generation", _Tok()) == \
        expected.format(behavior="PLACEHOLDER behavior", generation="PLACEHOLDER generation")


def test_the_three_copies_are_one():
    import inspect

    from safetune.core.eval.pipeline import scorer
    assert not hasattr(E, "_HARMBENCH_PROMPT") and not hasattr(scorer, "_HARMBENCH_LLAMA2_CLS_PROMPT")
    assert "<<SYS>>" not in inspect.getsource(E) and "<<SYS>>" not in inspect.getsource(scorer)
