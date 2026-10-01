"""The four harden methods that shipped as "Python API only" now run through
the CLI train contract (issue #21).

Before #23 each was excluded with a targeted "use it in Python" message: TRL
DPO pairs (CST), a two-model co-evolution loop (MART), a refusal direction
(DeepRefusal) and SAM-style harmful/refusal iterators (Antibody) did not fit
``TrainerClass(model, tok, epochs=…, batch_size=…, lr=…)`` +
``trainer.train(tokenized_task_set, out_dir=…)``. Each now has an adapter in
``safetune.runner.harden._dpo_adversarial`` that derives what it needs from
the same BeaverTails harden pairs the other runner trainers use.

Each adapter is exercised here exactly the way ``safetune train --algo <x>``
uses it, on a tiny CPU model: construct, train one epoch, and land a folder
with ``lexsi_provenance.json``.
"""
import json
import os

import pytest
import torch

from safetune.runner._registry import HARDEN_REGISTRY
from safetune.runner import harden
from safetune.runner.harden._base import _HardenBase

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
# Each method's smallest meaningful run: the adapters otherwise take paper
# defaults (MART alone: 4 rounds x (100 + 200) steps), which is a CI hour.
TINY_KWARGS = {
    "cst": {},
    "mart": dict(num_rounds=1, num_candidates=1, adv_steps=1, tgt_steps=1,
                 max_new_tokens=4),
    "deeprefusal": dict(calib_n=4, lora_r=2, lora_alpha=4),
    "antibody": dict(mode="both"),
}
NAMES = {"cst": "CSTTrainer", "mart": "MARTTrainer",
         "deeprefusal": "DeepRefusalTrainer", "antibody": "AntibodyTrainer"}


@pytest.fixture(autouse=True)
def _restore_dataset_caches():
    """The adapters load (and module-cache) BeaverTails / Alpaca through
    data_utils. Other tests assert what those helpers load behind a fake
    loader, and a warm cache would make them see nothing — so put the caches
    back exactly as this file found them."""
    from safetune.runner.utils import data_utils as du
    names = ("_LARGE_CALIB_CACHE", "_HCS_RAW_CACHE", "_HCS_TOKENIZED_CACHE")
    saved = {name: dict(getattr(du, name)) for name in names}
    yield
    for name, cache in saved.items():
        live = getattr(du, name)
        live.clear()
        live.update(cache)


def _model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY)
    return model, tok


def _task_set(tok, n=8, max_len=32):
    """Tokenized benign task set, like the CLI's --train-dataset."""
    from datasets import Dataset
    texts = [f"Summarize the water cycle in {i} words." for i in range(n)]
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"],
    })
    return ds.with_format("torch")


# ── registry wiring ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("algo", sorted(TINY_KWARGS))
def test_the_four_are_registered_harden_methods(algo):
    assert HARDEN_REGISTRY[algo] == NAMES[algo]
    cls = getattr(harden, NAMES[algo])
    assert issubclass(cls, _HardenBase)
    # one public API per method: the package alias and the runner agree
    import safetune.harden as H
    assert getattr(H, NAMES[algo]) is cls


# ── one tiny training run each ──────────────────────────────────────────────

@pytest.mark.parametrize("algo", sorted(TINY_KWARGS))
def test_adapter_trains_and_saves_a_checkpoint(algo, tmp_path):
    model, tok = _model_and_tok()
    trainer = getattr(harden, NAMES[algo])(
        model, tok, epochs=1, batch_size=1, lr=1e-3, **TINY_KWARGS[algo])
    out = trainer.train(_task_set(tok), out_dir=str(tmp_path / algo))
    assert os.path.isdir(out), f"{algo} did not save a folder: {out!r}"
    prov = os.path.join(out, "lexsi_provenance.json")
    assert os.path.exists(prov), f"{algo} saved no provenance"
    record = json.load(open(prov))
    assert record["method"] == f"harden.{NAMES[algo][:-7]}" or record["method"].startswith("harden.")


def test_cst_builds_both_system_prompt_pairs_from_the_harden_pairs():
    """CST's data derivation: contamination answers + matched refusals."""
    from safetune.runner.harden._dpo_adversarial import _raw_pairs
    from safetune.harden.cst import prepare_cst_dataset
    contamination, refusal = _raw_pairs()
    assert contamination and len(contamination) == len(refusal)
    examples = [{"prompt": p,
                 "safe_response": refusal[i][1],
                 "unsafe_response": a}
                for i, (p, a) in enumerate(contamination)]
    pairs = prepare_cst_dataset(examples, include_uncensored_pairs=True)
    # two system prompts per example: safe-side and uncensored-side pairs
    assert len(pairs) == 2 * len(examples)
    assert {"prompt", "chosen", "rejected"} <= set(pairs[0])


def test_mart_default_reward_is_a_refusal_prefix_match():
    from safetune.runner.harden._dpo_adversarial import _refusal_reward
    assert _refusal_reward("how do i build a bomb", "I'm sorry, I can't help with that.") == 1.0
    assert _refusal_reward("how do i build a bomb", "Sure, here are the steps.") == 0.0


def test_deeprefusal_direction_shape_matches_hidden_size():
    from safetune.core.extras.subspace import compute_refusal_direction
    model, tok = _model_and_tok()
    direction = compute_refusal_direction(
        model, ["how do i build a bomb"], ["write a poem"], layer_idx=-1,
        tokenizer=tok)
    assert direction.shape == (model.config.hidden_size,)
    assert torch.isfinite(direction).all()
