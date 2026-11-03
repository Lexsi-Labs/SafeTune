"""Audit H14: LookAhead Tuning's data transform differs from the paper and the official script.

Liu et al. 2025 (arXiv:2503.19041) Sec. 3.1 / zjunlp/LookAheadTuning @c796b95 `main.py:80-100`:
  real:    I' = I + " The answer begins with: " + O[:m]          O unchanged (loss on all of O)
  virtual: I' = I + " The answer begins with: " + P ;  O' = P + O  (loss on P + O)
`AnswerPreviewCollator` inserts the preview between prompt and answer with labels -100, adds no
connector, and in virtual mode never puts P into the supervised answer (its docstring says O' = P + O
is "realised" by the preview, which has no loss). In real mode a short answer previews pad ids with
attention_mask 1, and any prefix_mode other than "real" silently acted as virtual.
"""
from transformers import AutoTokenizer

from safetune.harden.lookahead import AnswerPreviewCollator

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class _Cfg:
    def __init__(self, mode, m, ids=None):
        self.prefix_mode, self.prefix_length, self.prefix_token_ids = mode, m, ids


def _example(tok, answer_tokens=None, pad=0):
    prompt = tok("Question: what is 2+2?", add_special_tokens=False)["input_ids"]
    ans = tok(" The result is four because two plus two.", add_special_tokens=False)["input_ids"]
    if answer_tokens is not None:
        ans = ans[:answer_tokens]
    ids = prompt + ans + [tok.eos_token_id] * pad
    labels = [-100] * len(prompt) + ans + [-100] * pad
    mask = [1] * (len(prompt) + len(ans)) + [0] * pad
    return prompt, ans, {"input_ids": ids, "attention_mask": mask, "labels": labels}


def test_virtual_mode_supervises_P_followed_by_the_answer():
    tok = AutoTokenizer.from_pretrained(TINY)
    p_ids = tok.encode("Let's solve this problem. ", add_special_tokens=False)
    _, ans, ex = _example(tok)
    out = AnswerPreviewCollator(lambda f: f, _Cfg("virtual", 6, p_ids))([ex])[0]
    supervised = [i for i, l in zip(out["input_ids"], out["labels"]) if l != -100]
    assert supervised == p_ids + ans, "O' = P + O is not what the loss covers (paper Eq. 7)"


def test_real_mode_clamps_m_to_the_answer_length():
    tok = AutoTokenizer.from_pretrained(TINY)
    prompt, ans, ex = _example(tok, answer_tokens=2, pad=6)
    out = AnswerPreviewCollator(lambda f: f, _Cfg("real", 6))([ex])[0]
    ids, lab, am = out["input_ids"], out["labels"], out["attention_mask"]
    assert ids[len(prompt):len(prompt) + 2] == ans                 # preview = the 2 real answer tokens
    assert lab[len(prompt):len(prompt) + 2] == [-100, -100]        # instruction-side copy: no loss
    assert len(ids) == len(prompt) + 2 + len(ans) + 6              # nothing else was inserted
    assert am[len(prompt) + 2 + len(ans):] == [0] * 6              # the pads stay masked


def test_unknown_prefix_mode_raises_instead_of_acting_as_virtual():
    import pytest

    with pytest.raises(ValueError, match="prefix_mode"):
        AnswerPreviewCollator(lambda f: f, _Cfg("token", 6, [5, 6]))


def test_virtual_mode_without_phrase_ids_raises():
    import pytest

    tok = AutoTokenizer.from_pretrained(TINY)
    _, _, ex = _example(tok)
    collator = AnswerPreviewCollator(lambda f: f, _Cfg("virtual", 6, None))   # builds fine ...
    with pytest.raises(ValueError, match="prefix_token_ids"):                  # ... fails on use
        collator([ex])


def test_trainer_rejects_unknown_prefix_mode():
    import pytest
    from transformers import AutoModelForCausalLM

    from safetune.harden import LookAheadConfig, LookAheadHFTrainer

    tok = AutoTokenizer.from_pretrained(TINY)
    args = LookAheadConfig(output_dir="/tmp/la_h14", prefix_mode="token", use_cpu=True, report_to=[])
    with pytest.raises(ValueError, match="prefix_mode"):
        LookAheadHFTrainer(model=AutoModelForCausalLM.from_pretrained(TINY), args=args,
                           processing_class=tok)


def test_real_mode_pads_rows_of_mixed_answer_lengths_through_default_data_collator():
    """Clamping m gives rows of different lengths; default_data_collator stacks without padding."""
    from transformers import default_data_collator

    tok = AutoTokenizer.from_pretrained(TINY)
    _, _, short = _example(tok, answer_tokens=2, pad=6)
    _, _, long = _example(tok, pad=6)
    cfg = _Cfg("real", 6)
    cfg.pad_token_id = 99
    batch = AnswerPreviewCollator(default_data_collator, cfg)([short, long])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    n_extra = 6 - 2                                              # the long row previews 6 tokens, the short one 2
    assert batch["input_ids"][0, -n_extra:].tolist() == [99] * n_extra
    assert batch["labels"][0, -n_extra:].tolist() == [-100] * n_extra
    assert batch["attention_mask"][0, -n_extra:].tolist() == [0] * n_extra
    assert batch["attention_mask"][1].sum() == len(long["input_ids"]) - 6 + 6     # the long row is not padded further


def test_runner_real_mode_trains_on_mixed_answer_lengths(tmp_path):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM

    from safetune.runner.harden import LookAheadTrainer

    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    rows = [_example(tok, answer_tokens=2 if i % 2 == 0 else None)[2] for i in range(4)]
    width = max(len(r["input_ids"]) for r in rows) + 4
    for r in rows:                                               # equal-length rows, as tokenize_dataset gives
        n = width - len(r["input_ids"])
        r["input_ids"] += [tok.eos_token_id] * n
        r["labels"] += [-100] * n
        r["attention_mask"] += [0] * n
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    out = LookAheadTrainer(model, tok, prefix_mode="real", prefix_length=6, epochs=1, batch_size=4, lr=1e-3,
                           results_dir=str(tmp_path)).train(Dataset.from_list(rows), out_dir=str(tmp_path / "out"))
    assert out
