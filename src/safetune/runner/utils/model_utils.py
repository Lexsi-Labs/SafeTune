"""Model and tokenizer utilities — ported from the SafeTune audit harness."""

from __future__ import annotations
import gc
import os
from pathlib import Path

import torch

from safetune.config import get_config, resolve_device, resolve_dtype

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
    cfg = model.config.get_text_config()  # vision-language configs keep token ids here
    pad_id = cfg.pad_token_id
    if pad_id is None:
        pad_id = (
            getattr(tok, "pad_token_id", None)
            or getattr(tok, "eos_token_id", None)
            or cfg.eos_token_id
        )
        if isinstance(pad_id, list):
            pad_id = pad_id[0]
        if pad_id is not None:
            cfg.pad_token_id = pad_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and gen_cfg.pad_token_id is None and pad_id is not None:
        gen_cfg.pad_token_id = pad_id
    return model


def load_model(name: str, *, dtype=None, device: str = None):
    """Load a fresh model (never cached — callers mutate them).

    ``device`` / ``dtype`` default to the runtime config: cuda > mps > cpu, and
    bf16 where the device supports it (fp16 on older CUDA, fp32 on CPU).
    """
    from safetune._refusal_helpers import _load_pretrained_lm
    device = resolve_device(device)
    m = _load_pretrained_lm(
        name,
        dtype=resolve_dtype(dtype, device),
        device_map=device,
    )
    m.eval()
    tok = _TOK_CACHE.get(name)
    return _fix_pad_token(m, tok)


def load_model_cpu(name: str, *, dtype=None):
    """Load a model on CPU — for reference/state-dict-donor models.

    ``dtype`` defaults to the runtime dtype so donors match the trained model.
    """
    from safetune._refusal_helpers import _load_pretrained_lm
    m = _load_pretrained_lm(
        name,
        dtype=resolve_dtype(dtype),
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
    r: int = None,
    lora_alpha: int = None,
    lora_dropout: float = None,
    target_modules=None,
):
    """Wrap a model with PEFT LoRA for memory-efficient training.

    Unset arguments come from the runtime config (``lora_r=16``,
    ``lora_alpha=32``, ``lora_dropout=0.05``, Llama-style ``*_proj`` modules).
    """
    from peft import LoraConfig, get_peft_model
    from safetune._refusal_helpers import _lora_target_modules
    rt = get_config()
    model.config.use_cache = False
    cfg = LoraConfig(
        r=r if r is not None else rt.lora_r,
        lora_alpha=lora_alpha if lora_alpha is not None else rt.lora_alpha,
        lora_dropout=lora_dropout if lora_dropout is not None else rt.lora_dropout,
        target_modules=_lora_target_modules(model, list(target_modules or rt.lora_target_modules)),
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
    processor=None,
    method: str = None,
    inputs: list = None,
    params: dict = None,
) -> str:
    """Save model + tokenizer to <out_dir>/<name> and return the path.

    A vision-language model (Aya Vision, North) also gets its processor:
    ``processor``, else the one at ``tokenizer.name_or_path``. The folder then
    loads with ``AutoProcessor`` / ``AutoModelForImageTextToText`` like the
    original checkpoint.

    Every folder gets a ``lexsi_provenance.json`` (``safetune.provenance``):
    ``method``, the source model (with its own provenance record when it came
    from a folder that has one), the extra ``inputs`` (e.g. datasets) and ``params``."""
    from safetune._refusal_helpers import _is_vision_config
    from safetune.provenance import input_entry, write_provenance
    source = (getattr(getattr(model, "config", None), "_name_or_path", None)
              or getattr(tokenizer, "name_or_path", None))
    model_in = input_entry("model", source or None)
    if out_dir is None:
        from safetune.runner.utils.results_writer import DEFAULT_RESULTS_DIR
        out_dir = os.path.join(DEFAULT_RESULTS_DIR, "checkpoints")
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=safe_serialization)
    if processor is None and _is_vision_config(model.config):
        from transformers import AutoProcessor
        try:
            processor = AutoProcessor.from_pretrained(tokenizer.name_or_path)
        except Exception as e:  # checkpoint still usable as text-only
            import warnings
            warnings.warn(f"save_checkpoint: no processor saved for {path} "
                          f"(could not load one from {tokenizer.name_or_path!r}: {e}); "
                          "pass processor= to include it.")
    if processor is not None:
        processor.save_pretrained(path)
    tokenizer.save_pretrained(path)  # after the processor: keeps pad-token and other edits
    write_provenance(path, method, inputs=[model_in, *(inputs or [])], params=params)
    return path
