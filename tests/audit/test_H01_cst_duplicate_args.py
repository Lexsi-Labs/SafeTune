"""Audit H01 (review of #111): CSTTrainer must not let a positional argument overwrite a keyword.

`kwargs.update(zip(names, args))` made `CSTTrainer(model, args=a1, ...)` + a positional `args` silently
use the positional one. Python raises TypeError for that; so does CSTTrainer now.
"""
import tempfile

import pytest
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from safetune.harden import CSTConfig, CSTTrainer
from safetune.harden.cst import prepare_cst_dataset

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
EXAMPLE = {"prompt": "How do I pick a lock?", "safe_response": "I cannot help with that request.",
           "unsafe_response": "First insert a tension wrench into the keyway."}


def _args():
    return CSTConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=2, max_steps=1,
                     report_to=[], use_cpu=True, save_strategy="no", max_length=128)


@pytest.fixture(scope="module")
def tok():
    t = AutoTokenizer.from_pretrained(TINY)
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    return t


def test_an_argument_passed_positionally_and_by_keyword_raises(tok):
    ds = Dataset.from_list(prepare_cst_dataset([EXAMPLE]))
    model = AutoModelForCausalLM.from_pretrained(TINY)
    # DPOTrainer order: model, ref_model, args, ...: the third positional is `args`.
    with pytest.raises(TypeError, match="got multiple values for argument 'args'"):
        CSTTrainer(model, None, _args(), args=_args(), train_dataset=ds, processing_class=tok)


def test_positional_only_and_keyword_only_still_work(tok):
    ds = Dataset.from_list(prepare_cst_dataset([EXAMPLE]))
    pos = CSTTrainer(AutoModelForCausalLM.from_pretrained(TINY), None, _args(), None, ds, None, tok)
    kw = CSTTrainer(model=AutoModelForCausalLM.from_pretrained(TINY), args=_args(),
                    train_dataset=ds, processing_class=tok)
    for tr in (pos, kw):
        tr.train()
        assert tr.state.global_step == 1
