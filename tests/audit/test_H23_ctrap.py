"""Audit H23 (Minor, for a decision): CTRAP's collapse loss averages over every attended token.

Yi et al. 2025 (arXiv:2505.16559) Eq. 1: l_Collapse(theta; D) = E_(x,y)~D [ -1/|y| sum_{t=1..|y|} log p(e | x o y_<t; theta) ],
i.e. the mean over the |y| RESPONSE positions of the context x o y_<t. `_collapse_loss` averages the
-log p(e | .) over all attended positions (prompt included). No verified official code (anonymous artifact).
"""
import tempfile

import torch
from transformers import AutoModelForCausalLM, default_data_collator

import safetune.harden as H

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_collapse_loss_is_the_mean_over_the_response_positions():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    args = H.CTRAPConfig(output_dir=tempfile.mkdtemp(), use_cpu=True, report_to=[], per_device_train_batch_size=2)
    tr = H.CTRAPHFTrainer(model=model, args=args, data_collator=default_data_collator,
                          harmful_dataset=[{"input_ids": torch.ones(8, dtype=torch.long),
                                            "attention_mask": torch.ones(8, dtype=torch.long),
                                            "labels": torch.ones(8, dtype=torch.long)}])
    ids = torch.randint(5, 200, (2, 16))
    labels = ids.clone()
    labels[:, :8] = -100                                     # 8 prompt tokens, 8 response tokens
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    got = tr._collapse_loss_at(model, {}, batch).item()

    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=batch["attention_mask"]).logits.float(), -1)
    nll_e = -logp[..., tr._collapse_token_id]                # position p predicts token p+1
    resp_positions = labels[:, 1:] != -100                   # positions whose next token is a response token
    want = nll_e[:, :-1][resp_positions].mean().item()
    assert abs(got - want) < 1e-5, f"collapse loss {got:.4f} (all attended tokens) vs mean over y {want:.4f}"


def test_without_a_prompt_the_loss_is_bit_identical_to_the_attention_mask_mean():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    args = H.CTRAPConfig(output_dir=tempfile.mkdtemp(), use_cpu=True, report_to=[], per_device_train_batch_size=2)
    tr = H.CTRAPHFTrainer(model=model, args=args, data_collator=default_data_collator,
                          harmful_dataset=[{"input_ids": torch.ones(8, dtype=torch.long),
                                            "attention_mask": torch.ones(8, dtype=torch.long),
                                            "labels": torch.ones(8, dtype=torch.long)}])
    ids = torch.randint(5, 200, (2, 16))
    am = torch.ones_like(ids)
    am[1, 12:] = 0                                            # padding, but no masked prompt
    batch = {"input_ids": ids, "attention_mask": am, "labels": ids.clone()}
    got = tr._collapse_loss_at(model, {}, batch)
    with torch.no_grad():
        nll_e = -torch.log_softmax(model(input_ids=ids, attention_mask=am).logits.float(), -1)[..., tr._collapse_token_id]
    old = (nll_e * am.float()).sum() / am.float().sum()
    assert torch.equal(got.detach(), old)
