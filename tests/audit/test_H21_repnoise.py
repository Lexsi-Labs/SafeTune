"""Audit H21: RepNoise's token mask covers the harmful PROMPT when the retain batch is unrelated.

Rosati et al. 2024 (arXiv:2405.14577) / official domenicrosati/representation-noising @03689ff
`representation_noising/loss.py::rep_noise_loss`: the harmful and harmless batches are PAIRED (same
prompt, different response), and `mask = ~eq(harmful_ids, harmless_ids)` therefore selects only the
response tokens. The runner pairs the harmful batch with an unrelated training batch, so the mask also
selects most of the harmful prompt: the MMD noise term and the log-ascent term act on the question
text (tiny Llama: 104 of 239 in-mask tokens are prompt tokens, 87% of the prompt tokens).
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

import safetune.harden as H
from safetune.runner.utils.data_utils import build_sft_dataset, harden_contamination_sets

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
KEEP = ("input_ids", "attention_mask", "labels")


def _batches():
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    if not tok.chat_template:
        tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                             "{% if add_generation_prompt %}assistant: {% endif %}")
    harm = harden_contamination_sets(tok, n=16, max_len=64)[0].with_format("torch")
    train = build_sft_dataset(tok, n_total=16, harmful_frac=0.0, max_len=64, domain="gsm8k").with_format("torch")

    def first(ds):
        ds = ds.remove_columns([c for c in ds.column_names if c not in KEEP])
        return next(iter(DataLoader(ds, batch_size=4, collate_fn=default_data_collator)))

    return first(harm), first(train)


def test_mask_excludes_the_harmful_prompt_tokens():
    harmful, retain = _batches()
    mask = H.RepNoiseHFTrainer._diff_mask(harmful, retain)
    prompt = harmful["attention_mask"].bool() & (harmful["labels"] == -100)
    assert prompt.any()
    assert not (mask & prompt).any(), (
        f"{int((mask & prompt).sum())} of {int(mask.sum())} masked-in tokens are harmful prompt tokens")


def test_response_mask_has_zero_prompt_share_and_covers_the_response_tokens():
    harmful, retain = _batches()
    mask = H.RepNoiseHFTrainer._diff_mask(harmful, retain)
    response = (harmful["labels"] != -100) & harmful["attention_mask"].bool()
    assert (mask & ~response).sum().item() == 0                    # no prompt (or pad) token
    assert torch.equal(mask, response)


def test_paired_batches_keep_the_reference_difference_mask_bit_for_bit():
    harmful, _ = _batches()
    ids = harmful["input_ids"].clone()
    resp = harmful["labels"] != -100
    harmless = dict(harmful)
    harmless["input_ids"] = torch.where(resp, (ids + 7) % 1000 + 5, ids)   # same prompt, other response
    mask = H.RepNoiseHFTrainer._diff_mask(harmful, harmless)
    assert torch.equal(mask, ~torch.eq(harmful["input_ids"], harmless["input_ids"]))
    assert (mask & ~resp).sum().item() == 0                                # and it is the response
