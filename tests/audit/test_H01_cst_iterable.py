"""Audit H01 (review of #111): CSTTrainer must accept an IterableDataset.

The system-role check called len(train_dataset) before looking at the template, so any
IterableDataset raised "TypeError: object of type 'IterableDataset' has no len()" at construction.
"""
import logging
import tempfile

from datasets import IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from safetune.harden import CSTConfig, CSTTrainer
from safetune.harden.cst import prepare_cst_dataset
from test_H01_cst_system_role import EXAMPLE, TINY, tok  # noqa: F401  (fixture)

ACCEPTS_SYSTEM = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                  "{% if add_generation_prompt %}assistant: {% endif %}")


def _iterable(rows):
    def gen():
        yield from rows
    return IterableDataset.from_generator(gen)


def _train(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(TINY)
    args = CSTConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=2, max_steps=1,
                     report_to=[], use_cpu=True, save_strategy="no", max_length=128)
    tr = CSTTrainer(model=model, args=args, processing_class=tokenizer,
                    train_dataset=_iterable(prepare_cst_dataset([EXAMPLE])))
    tr.train()
    return tr


def test_iterable_dataset_with_a_template_that_accepts_system():
    t = AutoTokenizer.from_pretrained(TINY)
    t.pad_token = t.pad_token or t.eos_token
    t.chat_template = ACCEPTS_SYSTEM
    assert _train(t).state.global_step == 1


def test_iterable_dataset_is_folded_when_the_template_rejects_system(tok, caplog):  # noqa: F811
    with caplog.at_level(logging.WARNING):
        tr = _train(tok)
    assert tr.state.global_step == 1
    assert len([r for r in caplog.records if "folded" in r.getMessage()]) == 1
