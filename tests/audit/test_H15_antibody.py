"""Audit H15: Antibody's fine-tuning-stage weight scores the refusal under the wrong prompt.

Paper Eq. (7): r(x_i, y_i) = log pi(y_i | x_i) - log pi(y_r | x_i), the refusal y_r being scored after
the SAME prompt x_i. Official minhquoc0712/Antibody @6c557b1 `train.py:124-170` builds a paired
`refusal_input_ids` / `refusal_labels` view for every instance. `AntibodyTrainer._refusal_labels` takes
the labels of an unrelated batch from `refusal_dataset`, and `_refusal_logprob` splices them into the
user's `input_ids` at the positions where they are not -100, so (x_i, y_r) is only right when the two
prompts have the same length.
"""
import tempfile

import torch
from transformers import AutoModelForCausalLM

import safetune.harden as H

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
T = 20


def _seq(prompt_len, ans):
    g = torch.Generator().manual_seed(prompt_len * 7 + len(ans))
    p = torch.randint(5, 200, (prompt_len,), generator=g)
    ids = torch.cat([p, torch.tensor(ans)])
    lab = torch.cat([torch.full((prompt_len,), -100), torch.tensor(ans)])
    pad = T - len(ids)
    return torch.cat([ids, torch.zeros(pad, dtype=torch.long)]), torch.cat([lab, torch.full((pad,), -100)]), p


def test_refusal_logprob_is_scored_after_the_samples_own_prompt():
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    args = H.AntibodyConfig(output_dir=tempfile.mkdtemp(), use_cpu=True, report_to=[],
                            per_device_train_batch_size=2, max_steps=1)
    tr = H.AntibodyTrainer(model=m, args=args, mode="finetune")
    ans_user, ans_ref = [50, 51, 52, 53, 54, 55], [90, 91, 92, 93, 94, 95]
    user = [_seq(6, ans_user), _seq(9, ans_user)]               # prompts of length 6 and 9
    refusal_labels = torch.stack([_seq(12, ans_ref)[1]] * 2)    # refusal batch with its own 12-token prompts
    ids = torch.stack([a for a, _, _ in user])
    inputs = {"input_ids": ids, "attention_mask": (ids != 0).long(), "labels": torch.stack([b for _, b, _ in user])}
    with torch.no_grad():
        logits = m(**{k: v for k, v in inputs.items() if k != "labels"}).logits
        got = tr._refusal_logprob(m, inputs, logits, refusal_labels)
        want = []
        for _, _, p in user:
            full = torch.cat([p, torch.tensor(ans_ref)])
            fi = torch.cat([full, torch.zeros(T - len(full), dtype=torch.long)])
            fl = torch.cat([torch.full((len(p),), -100), torch.tensor(ans_ref), torch.full((T - len(full),), -100)])
            lg = m(input_ids=fi[None], attention_mask=(fi != 0).long()[None]).logits
            want.append(tr._sequence_logprob(lg, fl[None]).item())
    assert torch.allclose(got, torch.tensor(want), atol=1e-4), f"splice {got.tolist()} vs (x_i, y_r) {want}"


def test_finetune_stage_runs_with_a_refusal_set_of_another_shape():
    """A refusal dataset with a different length and prompts than the user batch is accepted."""
    from datasets import Dataset
    from transformers import AutoTokenizer, default_data_collator

    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token

    def rows(texts, length, mask):
        out = []
        for t in texts:
            e = tok(t, padding="max_length", max_length=length, return_tensors="pt")
            ids = e["input_ids"][0]
            lab = ids.clone()
            lab[:mask] = -100
            lab[e["attention_mask"][0] == 0] = -100
            out.append({"input_ids": ids.tolist(), "attention_mask": e["attention_mask"][0].tolist(),
                        "labels": lab.tolist()})
        return Dataset.from_list(out).with_format("torch")

    benign = rows([f"user question {i} answer text here for the model" for i in range(8)], 24, 6)
    refusal = rows([f"a much longer harmful prompt number {i} I cannot help with that request" for i in range(6)], 40, 12)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    args = H.AntibodyConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=4, max_steps=2,
                            learning_rate=1e-3, report_to=[], use_cpu=True, save_strategy="no")
    tr = H.AntibodyTrainer(model=m, args=args, train_dataset=benign, data_collator=default_data_collator,
                           processing_class=tok, mode="finetune", refusal_dataset=refusal)
    tr.train()
    assert torch.isclose(tr._last_weights.sum(), torch.tensor(1.0), atol=1e-5)
