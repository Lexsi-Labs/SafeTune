"""Audit H06: ASRT's discovered rows must reach the next epoch's batches.

ASRTCallback.on_epoch_end appends (prompt, refusal) rows to `trainer.train_dataset` "so the next
epoch trains on the holes the red-team found". For a `datasets.Dataset` it assigns a *new*
concatenated dataset to `trainer.train_dataset`, but HF's Trainer builds its DataLoader once at
the start of `train()` and re-iterates it every epoch, so the loader keeps the old dataset and the
appended rows are never trained on.
"""
import logging
import tempfile

import pytest
import torch
from datasets import Dataset
from transformers import (LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments,
                          default_data_collator)

from safetune.harden.asrt import ASRTCallback, ASRTConfig

MARK = 31   # token id that only appended rows contain


class _Attacker:
    def generate(self, prompts):
        return ["Sure, here is how to do that."] * len(prompts)   # never refuses -> every probe succeeds


def _row(token):
    return {"input_ids": [token] * 6, "attention_mask": [1] * 6, "labels": [token] * 6}


def _run(as_list: bool) -> int:
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    rows = [_row(i % 8) for i in range(8)]
    train = rows if as_list else Dataset.from_list(rows)
    seen = []

    def collate(features):
        seen.append(sum(1 for f in features if f["input_ids"][0] == MARK))
        return {k: torch.tensor([f[k] for f in features]) for k in ("input_ids", "attention_mask", "labels")}

    cb = ASRTCallback(attacker=_Attacker(), adversarial_prompts=["p1", "p2"],
                      config=ASRTConfig(eval_steps=1, probes_per_eval=2),
                      contrastive_builder=lambda prompt: _row(MARK))
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=4,
                             num_train_epochs=2, use_cpu=True, report_to=[], save_strategy="no",
                             remove_unused_columns=False, dataloader_drop_last=False)
    tr = Trainer(model=model, args=args, train_dataset=train, data_collator=collate, callbacks=[cb])
    cb.bind_trainer(tr)
    tr.train()
    return sum(seen)


def test_asrt_rows_appended_in_epoch_1_are_trained_on_in_epoch_2_datasets_dataset():
    assert _run(as_list=False) > 0, "appended rows never reached a training batch"


def test_asrt_datasets_dataset_matches_list_behaviour():
    # 4 appended rows (2 probes x 2 steps); max_steps is fixed by epoch 1, so epoch 2 is cut at the
    # same point for both containers and the two runs see the same marked rows.
    assert _run(as_list=True) == _run(as_list=False) == 4


# Review of #111: the default builder returns prompt/completion rows with no input_ids. Appended to a
# tokenized train set they reached the default collator in epoch 2 and raised KeyError 'input_ids'.


@pytest.mark.parametrize("as_list", [False, True], ids=["datasets_dataset", "list"])
@pytest.mark.parametrize("remove_unused", [True, False])
def test_default_builder_rows_that_do_not_fit_the_train_set_are_kept_not_trained(
        as_list, remove_unused, caplog):
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    rows = [_row(i % 8) for i in range(8)]
    cb = ASRTCallback(attacker=_Attacker(), adversarial_prompts=["p1", "p2"],
                      config=ASRTConfig(eval_steps=1, probes_per_eval=2))
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=4,
                             num_train_epochs=2, use_cpu=True, report_to=[], save_strategy="no",
                             remove_unused_columns=remove_unused)
    tr = Trainer(model=model, args=args, train_dataset=rows if as_list else Dataset.from_list(rows),
                 data_collator=default_data_collator, callbacks=[cb])
    cb.bind_trainer(tr)
    with caplog.at_level(logging.WARNING):
        tr.train()
    assert len(tr.train_dataset) == 8
    assert len(cb.buffer) == 8                      # kept: 2 probes x 4 steps
    assert any("input_ids" in r.getMessage() for r in caplog.records)


def _collate_tokens(features):          # module level: DataLoader workers must pickle it
    return {k: torch.tensor([f[k] for f in features]) for k in ("input_ids", "attention_mask", "labels")}


def _marked_row(prompt):
    return _row(MARK)


def test_persistent_workers_keep_the_buffer_instead_of_crashing(caplog):
    # Review of #111: a persistent worker keeps its own copy of the dataset, so rows appended in the
    # main process are not there and the sampler's new indices raised IndexError in the worker.
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    cb = ASRTCallback(attacker=_Attacker(), adversarial_prompts=["p1", "p2"],
                      config=ASRTConfig(eval_steps=1, probes_per_eval=2), contrastive_builder=_marked_row)
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=4,
                             num_train_epochs=2, use_cpu=True, report_to=[], save_strategy="no",
                             remove_unused_columns=False, dataloader_num_workers=1,
                             dataloader_persistent_workers=True)
    tr = Trainer(model=model, args=args, train_dataset=Dataset.from_list([_row(i % 8) for i in range(8)]),
                 data_collator=_collate_tokens, callbacks=[cb])
    cb.bind_trainer(tr)
    with caplog.at_level(logging.WARNING):
        tr.train()
    assert len(tr.train_dataset) == 8 and len(cb.buffer) == 8
    assert any("persistent" in r.getMessage() for r in caplog.records)
