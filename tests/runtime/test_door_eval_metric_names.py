"""SafetyDOORTrainer names the DPO eval metrics after each named eval set, as CSTTrainer does (#84)."""
import tempfile

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from safetune.harden import DOORConfig, SafetyDOORTrainer

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _pairs(chosen, rejected):
    return Dataset.from_list([{"prompt": f"Q{i}?", "chosen": chosen, "rejected": rejected} for i in range(4)])


def test_each_eval_set_keeps_its_own_dpo_metrics():
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY)
    # door_pure_mode=False: the DPO loss (and its metrics) runs on batches without DOOR fields
    args = DOORConfig(output_dir=tempfile.mkdtemp(), per_device_eval_batch_size=2, report_to=[],
                      use_cpu=True, save_strategy="no", max_length=32, door_pure_mode=False)
    tr = SafetyDOORTrainer(model=model, args=args, processing_class=tok, train_dataset=_pairs(" yes", " no"),
                           eval_dataset={"alpha": _pairs(" yes", " no"), "beta": _pairs(" no", " yes")})

    metrics = tr.evaluate()
    # before: both sets wrote eval_rewards/..., so the second overwrote the first
    assert "eval_alpha_rewards/accuracies" in metrics and "eval_beta_rewards/accuracies" in metrics
    assert "eval_rewards/accuracies" not in metrics

    single = tr.evaluate(_pairs(" yes", " no"))
    assert "eval_rewards/accuracies" in single
