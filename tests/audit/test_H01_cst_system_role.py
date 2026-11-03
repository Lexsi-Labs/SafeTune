"""Audit H01 (review of #111): a chat template that rejects the system role must not break CST.

SIMULATED template: a Gemma-style template written for this test (it raises on a leading system
message). It has not been run against a real Gemma tokenizer; none is available offline.
"""
import logging
import tempfile

import pytest
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from safetune.harden import CSTConfig, CSTTrainer
from safetune.harden.cst import prepare_cst_dataset

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
EXAMPLE = {"prompt": "How do I pick a lock?", "safe_response": "I cannot help with that request.",
           "unsafe_response": "First insert a tension wrench into the keyway."}
NO_SYSTEM_TEMPLATE = (
    "{{ bos_token }}{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}")


@pytest.fixture()
def tok():
    t = AutoTokenizer.from_pretrained(TINY)
    t.pad_token = t.pad_token or t.eos_token
    t.chat_template = NO_SYSTEM_TEMPLATE
    return t


def _trainer(tok, rows):
    model = AutoModelForCausalLM.from_pretrained(TINY)
    args = CSTConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=2, max_steps=1,
                     report_to=[], use_cpu=True, save_strategy="no", max_length=128)
    return CSTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                      processing_class=tok,
                      peft_config=LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))


def test_system_prompt_is_folded_into_the_user_turn_with_one_warning(tok, caplog):
    with caplog.at_level(logging.WARNING):
        rows = prepare_cst_dataset([EXAMPLE], tokenizer=tok)
    assert len([r for r in caplog.records if "folded" in r.getMessage()]) == 1
    safe = [r for r in rows if r["cst_mode"] == "safe"][0]
    unc = [r for r in rows if r["cst_mode"] == "uncensored"][0]
    assert [m["role"] for m in safe["prompt"]] == ["user"]
    assert "harmless" in safe["prompt"][0]["content"] and EXAMPLE["prompt"] in safe["prompt"][0]["content"]
    assert safe["prompt"] != unc["prompt"]                      # CST still sees different prompts
    tok.apply_chat_template(safe["prompt"], tokenize=False)     # and the template accepts it


def test_trainer_construction_does_not_raise_when_the_dataset_was_prepared_without_a_tokenizer(
        tok, caplog):
    rows = prepare_cst_dataset([EXAMPLE])                        # still [system, user]
    assert rows[0]["prompt"][0]["role"] == "system"
    with caplog.at_level(logging.WARNING):
        tr = _trainer(tok, rows)
    assert len([r for r in caplog.records if "folded" in r.getMessage()]) == 1
    assert tr.train_dataset is not None


def test_a_template_that_accepts_system_is_left_alone():
    t = AutoTokenizer.from_pretrained(TINY)
    t.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}")
    rows = prepare_cst_dataset([EXAMPLE], tokenizer=t)
    assert [m["role"] for m in rows[0]["prompt"]] == ["system", "user"]


# Review of #111: the fold looked only at train_dataset and at processing_class passed by keyword.
def _args():
    return CSTConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=2, max_steps=1,
                     report_to=[], use_cpu=True, save_strategy="no", max_length=128)


@pytest.mark.parametrize("as_dict", [False, True], ids=["dataset", "dict_of_datasets"])
def test_eval_dataset_with_a_system_turn_is_folded(tok, as_dict):
    ds = Dataset.from_list(prepare_cst_dataset([EXAMPLE]))
    tr = CSTTrainer(model=AutoModelForCausalLM.from_pretrained(TINY), args=_args(),
                    train_dataset=ds, eval_dataset={"cst": ds} if as_dict else ds,
                    processing_class=tok)
    assert tr.eval_dataset is not None


def test_positional_processing_class_is_folded_too(tok):
    ds = Dataset.from_list(prepare_cst_dataset([EXAMPLE]))
    # DPOTrainer order: model, ref_model, args, data_collator, train_dataset, eval_dataset, processing_class
    tr = CSTTrainer(AutoModelForCausalLM.from_pretrained(TINY), None, _args(), None, ds, None, tok)
    tr.train()
    assert tr.state.global_step == 1
