"""Audit H12: SAP's contrastive batch scores prompt and padding tokens as targets.

Official ChengcanWu/SAP @487edc3 `datas.py::load_contrastive_data` sets the prompt positions of
`chosen_labels` / `rejected_labels` to -100 and every pad id to -100 before the contrastive loss
(`train.py::compute_contrastive_gradients`). `sap_contrastive_dataset` (runner/utils/data_utils.py)
used to keep raw token ids, so the prompt (identical in both labels) and the pad ids (different counts
in the two completions) entered L_safe and changed the harmful direction (cosine 0.72 to the masked
loss on a tiny model).
"""
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from safetune.core.optim.sap import SafetyAwareProbingWrapper
from safetune.runner.utils.data_utils import harden_contamination_pairs, sap_contrastive_dataset

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
MAX_LEN = 96


def _tok(side):
    t = AutoTokenizer.from_pretrained(TINY)
    t.pad_token = t.pad_token or t.eos_token
    t.padding_side = side
    if not t.chat_template:
        t.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    return t


def _reference_labels(tok, n):
    """Independent re-implementation of the official masking (right padding)."""
    contamination, _, refusal = harden_contamination_pairs(n)
    saved, tok.padding_side = tok.padding_side, "right"
    try:
        rows = []
        for (u, bad), (_, ref) in zip(contamination, refusal):
            prompt = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                             add_generation_prompt=True)
            enc = tok(prompt, truncation=True, max_length=MAX_LEN, padding="max_length",
                      add_special_tokens=False)
            input_len = sum(1 for t in enc["input_ids"] if t != tok.pad_token_id)
            lab = []
            for completion in (ref, bad):
                ids = torch.tensor(tok(prompt + completion, truncation=True, max_length=MAX_LEN,
                                       padding="max_length", add_special_tokens=False)["input_ids"])
                ids[:input_len] = -100
                ids[ids == tok.pad_token_id] = -100
                lab.append(ids)
            rows.append(lab)
        return rows
    finally:
        tok.padding_side = saved


@pytest.mark.parametrize("side", ["right", "left"])
def test_contrastive_labels_mask_prompt_and_padding(side):
    tok = _tok(side)
    ds = sap_contrastive_dataset(tok, n=4, max_len=MAX_LEN)
    assert tok.padding_side == side                      # the caller's tokenizer is left as it was
    for r, (ref_chosen, ref_rejected) in zip(ds, _reference_labels(tok, 4)):
        n_prompt = int(sum(r["attention_mask"]))
        for key, ref in (("chosen_labels", ref_chosen), ("rejected_labels", ref_rejected)):
            lab = torch.tensor(r[key])
            assert (lab[:n_prompt] == -100).all(), f"{key}: prompt tokens are loss targets"
            assert not (lab == tok.pad_token_id).any(), f"{key}: pad ids are loss targets"
            assert torch.equal(lab, ref), f"{key} differs from the official masking"


@pytest.mark.parametrize("side", ["right", "left"])
def test_harmful_direction_equals_the_masked_loss_direction(side):
    tok = _tok(side)
    ds = sap_contrastive_dataset(tok, n=4, max_len=MAX_LEN)
    ref = _reference_labels(tok, 4)
    r0 = ds[0]
    ids = torch.tensor([r0["input_ids"]])
    am = torch.tensor([r0["attention_mask"]])

    def direction(chosen, rejected):
        torch.manual_seed(0)
        m = get_peft_model(AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32),
                           LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
        with torch.no_grad():
            for n_, p in m.named_parameters():
                if "lora_B" in n_:
                    p.normal_(0, 0.05)
        g = SafetyAwareProbingWrapper.compute_contrastive_safety_gradient(
            m, ids, am, chosen[None], rejected[None], temperature=200.0)
        return torch.cat([v.flatten() for v in g.values()])

    got = direction(torch.tensor(r0["chosen_labels"]), torch.tensor(r0["rejected_labels"]))
    want = direction(ref[0][0], ref[0][1])
    cos = float(got @ want / (got.norm() * want.norm()))
    assert abs(cos - 1.0) < 1e-6, cos
