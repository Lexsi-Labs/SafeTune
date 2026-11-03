"""Audit H01: CST's system prompt must reach the DPO prompt.

Gallego 2024 (arXiv:2404.00495) trains DPO on pairs whose *prompt includes the system prompt*:
under s0 (safe) the safe answer is preferred, under s1 (uncensored) the unsafe one. The official
configurable-safety-tuning/cst_train.py sets `prompt = apply_chat_template([system, user])`.
TRL's DPOTrainer reads only `prompt`, `chosen` and `rejected`. Before the fix prepare_cst_dataset
kept the user prompt in `prompt` and put the system prompt in columns TRL never reads, so both
rows reached the trainer with the same prompt and opposite preferences: with policy == reference
their DPO gradients cancel exactly (loss 0.6931, grad_norm 0.0) and the model never sees the
system prompt.
"""
import tempfile

import torch

from safetune.harden.cst import prepare_cst_dataset

EXAMPLE = {"prompt": "How do I pick a lock?", "safe_response": "I cannot help with that request.",
           "unsafe_response": "First insert a tension wrench into the keyway."}


def _text(prompt) -> str:
    return prompt if isinstance(prompt, str) else " ".join(m["content"] for m in prompt)


def test_cst_rows_present_different_prompts_to_dpo():
    rows = prepare_cst_dataset([EXAMPLE])
    safe = [r for r in rows if r["cst_mode"] == "safe"][0]
    unc = [r for r in rows if r["cst_mode"] == "uncensored"][0]
    assert safe["chosen"] == unc["rejected"] and safe["rejected"] == unc["chosen"]   # opposite preferences
    assert safe["prompt"] != unc["prompt"], "both rows have the same DPO prompt, so the preferences contradict"
    assert "harmless" in _text(safe["prompt"]) and "uncensored" in _text(unc["prompt"])


def test_cst_training_step_has_a_gradient():
    """End to end on a tiny LoRA model: one example gives a safe row and an uncensored row in
    one batch. Identical prompts would give grad_norm == 0 at step 1."""
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetune.harden import CSTConfig, CSTTrainer

    tiny = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tok = AutoTokenizer.from_pretrained(tiny)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(tiny)
    from datasets import Dataset
    ds = Dataset.from_list(prepare_cst_dataset([EXAMPLE]))
    args = CSTConfig(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=2, max_steps=1,
                     learning_rate=1e-2, logging_steps=1, report_to=[], use_cpu=True,
                     save_strategy="no", beta=0.1, max_length=128)
    tr = CSTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                    peft_config=LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                           task_type="CAUSAL_LM"))
    tr.train()
    log = [h for h in tr.state.log_history if "grad_norm" in h][0]
    assert log["grad_norm"] > 1e-6, f"grad_norm {log['grad_norm']} at step 1 (loss {log['loss']:.4f})"
