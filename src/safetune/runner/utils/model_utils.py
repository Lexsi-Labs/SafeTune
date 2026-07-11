"""Model and tokenizer utilities — ported from the SafeTune audit harness."""

from __future__ import annotations
import gc
import os
from pathlib import Path

import torch

_DT = torch.bfloat16
_DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level tokenizer cache (same pattern as harness.py _TOK dict).
_TOK_CACHE: dict = {}


def derive_model_id(model_id, model=None, tokenizer=None) -> str:
    """Infer a canonical model name from whatever the caller passed."""
    if model_id is not None:
        return str(model_id)
    cand = getattr(tokenizer, "name_or_path", None)
    if not cand:
        cand = getattr(getattr(model, "config", None), "_name_or_path", None)
    return str(cand) if cand else "model"


def load_tok(name: str, *, cache: bool = True):
    """Load (and cache) an AutoTokenizer, ensuring pad_token is set."""
    from transformers import AutoTokenizer
    if cache and name in _TOK_CACHE:
        return _TOK_CACHE[name]
    t = AutoTokenizer.from_pretrained(name)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    t.padding_side = "right"
    if cache:
        _TOK_CACHE[name] = t
    return t


def _fix_pad_token(model, tok=None):
    """Silence the pad_token_id→eos_token_id warning by syncing config."""
    pad_id = model.config.pad_token_id
    if pad_id is None:
        pad_id = (
            getattr(tok, "pad_token_id", None)
            or getattr(tok, "eos_token_id", None)
            or model.config.eos_token_id
        )
        if isinstance(pad_id, list):
            pad_id = pad_id[0]
        if pad_id is not None:
            model.config.pad_token_id = pad_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and gen_cfg.pad_token_id is None and pad_id is not None:
        gen_cfg.pad_token_id = pad_id
    return model


def load_model(name: str, *, dtype=None, device: str = None):
    """Load a fresh model on GPU (never cached — callers mutate them)."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype or _DT,
        device_map=device or _DEV,
    )
    m.eval()
    tok = _TOK_CACHE.get(name)
    return _fix_pad_token(m, tok)


def load_model_cpu(name: str, *, dtype=None):
    """Load a model on CPU — for reference/state-dict-donor models."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype or _DT,
        device_map="cpu",
    )
    m.eval()
    tok = _TOK_CACHE.get(name)
    return _fix_pad_token(m, tok)


def free():
    """Collect garbage and release CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def lora_wrap(
    model,
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules=None,
):
    """Wrap a model with PEFT LoRA for memory-efficient training."""
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def save_checkpoint(
    model,
    tokenizer,
    name: str,
    *,
    out_dir: str = None,
    safe_serialization: bool = True,
) -> str:
    """Save model + tokenizer to <out_dir>/<name> and return the path."""
    if out_dir is None:
        from safetune.runner.utils.results_writer import DEFAULT_RESULTS_DIR
        out_dir = os.path.join(DEFAULT_RESULTS_DIR, "checkpoints")
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(path)
    return path
