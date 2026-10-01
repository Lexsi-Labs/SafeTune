"""Judge registry and top-level run_judge() entry point.

Each judge is loaded via vLLM (advbench uses StringMatchJudge, no model),
scored, then immediately unloaded to free GPU memory before the next judge
runs. Models are never kept alive across calls.

Score convention: 0.0 = refused/safe, 1.0 = complied/jailbroken/unsafe.
"""
from __future__ import annotations

import gc
import logging
from typing import List

from safetune.config import dtype_name, get_config

from ._parsers import (
    parse_ailuminate,
    parse_harmbench,
    parse_hexphi,
    parse_orbench,
    parse_sorrybench,
    parse_wildguard,
)
from ._templates import (
    _AILUMINATE_PROMPT,
    _HARMBENCH_PROMPT,
    _HEXPHI_PROMPT,
    _ORBENCH_PROMPT,
    _SORRYBENCH_TEMPLATE,
    _WILDGUARD_TEMPLATE,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

JUDGE_REGISTRY: dict[str, dict] = {
    "harmbench": {
        "model_id": "cais/HarmBench-Mistral-7b-val-cls",
        "max_new_tokens": 1,
        "max_model_len": 4096,
    },
    "wildguard": {
        "model_id": "allenai/wildguard",
        "max_new_tokens": 32,
        "max_model_len": 4096,
    },
    "sorrybench": {
        "model_id": "sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
        "max_new_tokens": 10,
        "max_model_len": 4096,
    },
    "hexphi": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_new_tokens": 10,
        "max_model_len": 4096,
    },
    "orbench": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_new_tokens": 256,
        "max_model_len": 8192,
    },
    "ailuminate": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "max_new_tokens": 10,
        "max_model_len": 4096,
    },
    "advbench": {  # string-match, no model
        "model_id": None,
        "max_new_tokens": 0,
        "max_model_len": 0,
    },
}


def _judge_spec(judge_key: str) -> dict:
    """Registry entry merged with ``configure(judges={judge_key: {...}})``."""
    return {**JUDGE_REGISTRY[judge_key], **get_config().judges.get(judge_key, {})}


# Maps each judge key to its (template_fn, parse_fn, needs_chat_template) triple.
#
# needs_chat_template=True  → raw prompt text is wrapped in Llama chat template
#                              via tokenizer.apply_chat_template() before passing
#                              to vLLM.  Matches cthetha-eval build_*_inputs().
# needs_chat_template=False → template already embeds format tokens verbatim
#                              (harmbench [INST]<<SYS>>, wildguard <s><|user|>,
#                              sorrybench [INST]).
#
# harmbench additionally truncates the generation to 512 tokens right-side
# per HarmBench evaluate_completions.py.
_JUDGE_CONFIG: dict[str, tuple] = {
    "harmbench": (
        lambda p, r, tok: _HARMBENCH_PROMPT.format(
            behavior=p,
            generation=_truncate_tokens(r, tok, max_tokens=512),
        ),
        parse_harmbench,
        False,  # template has [INST]<<SYS>> baked in
    ),
    "wildguard": (
        lambda p, r, tok: _WILDGUARD_TEMPLATE.format(prompt=p, response=r),
        parse_wildguard,
        False,  # template has <s><|user|>[INST] baked in
    ),
    "sorrybench": (
        lambda p, r, tok: _SORRYBENCH_TEMPLATE.format(question=p, answer=r),
        parse_sorrybench,
        False,  # template has [INST] baked in
    ),
    "hexphi": (
        lambda p, r, tok: _HEXPHI_PROMPT.format(prompt=p, response=r),
        parse_hexphi,
        True,   # Llama-3.1-8B judge: needs chat template
    ),
    "orbench": (
        lambda p, r, tok: _ORBENCH_PROMPT.format(question=p, response=r),
        parse_orbench,
        True,   # Llama-3.1-8B judge: needs chat template
    ),
    "ailuminate": (
        lambda p, r, tok: _AILUMINATE_PROMPT.format(prompt=p, response=r),
        parse_ailuminate,
        True,   # Llama-3.1-8B judge: needs chat template
    ),
}


def _truncate_tokens(text: str, tokenizer, max_tokens: int = 512) -> str:
    """Right-truncate text to max_tokens using the given tokenizer."""
    tokenizer.truncation_side = "right"
    ids = tokenizer.encode(text, max_length=max_tokens, truncation=True)
    return tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# AdvBench string-match scorer
# ---------------------------------------------------------------------------

def _score_advbench(responses: List[str]) -> List[float]:
    """0.0 (refused/safe) or 1.0 (complied/unsafe) per response, from
    StringMatchJudge in the runtime ``advbench_scorer`` mode (default "prefix")."""
    from safetune.core.eval.pipeline.scorer import StringMatchJudge
    judge = StringMatchJudge(mode=get_config().advbench_scorer)
    return [judge.score_one(r) for r in responses]


# ---------------------------------------------------------------------------
# vLLM judge runner (transformers fallback)
# ---------------------------------------------------------------------------

_warned_no_vllm = False


def _run_hf_judge(model_id: str, token_ids: List[List[int]], max_new_tokens: int) -> List[str]:
    """Greedy-decode pre-tokenized judge inputs with plain transformers."""
    import torch
    from safetune.evaluate.suite.evaluate import _load_hf_model

    tok, model = _load_hf_model(model_id)  # left padding
    texts: List[str] = []
    bs = get_config().gen_batch_size
    try:
        for i in range(0, len(token_ids), bs):
            enc = tok.pad({"input_ids": token_ids[i:i + bs]}, padding=True,
                          return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
    finally:
        del model
        gc.collect()
    return texts


def _run_vllm_judge(
    judge_key: str,
    prompts: List[str],
    responses: List[str],
    gpu_memory_utilization: float,
) -> List[float]:
    """Load the judge model via vLLM, score all (prompt, response) pairs, then
    unload the model and free GPU memory before returning.

    Uses plain transformers instead when vLLM is not installed or the runtime
    ``eval_backend`` is ``"hf"``.
    """
    global _warned_no_vllm
    use_hf = get_config().eval_backend == "hf"
    if not use_hf:
        try:
            import vllm  # noqa: F401
        except ImportError:
            use_hf = True
            if not _warned_no_vllm:
                _warned_no_vllm = True
                log.warning("vLLM is not installed; running judges with transformers "
                            "(slower). Install vllm for faster judging.")

    spec = _judge_spec(judge_key)
    model_id: str = spec["model_id"]
    max_new_tokens: int = spec["max_new_tokens"]
    max_model_len: int = spec["max_model_len"]

    template_fn, parse_fn, needs_chat_template = _JUDGE_CONFIG[judge_key]

    # Load tokenizer for harmbench truncation and chat-template judges.
    from transformers import AutoTokenizer
    from safetune.utils.errors import hf_access_errors
    with hf_access_errors(model_id, kind="judge model"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build judge inputs then pre-tokenize to token IDs before passing to vLLM.
    # This matches cthetha-eval's input pipeline exactly:
    #   1. harmbench: right-truncates generation to 512 tokens first
    #   2. hexphi/orbench/ailuminate: wraps in Llama chat template
    #   3. wildguard: add_special_tokens=False — template already has <s> baked in;
    #      letting the tokenizer add another BOS would corrupt scoring
    #   4. all: explicit right-truncation to (max_model_len - max_new_tokens)
    #      before passing token IDs to vLLM (no silent vLLM-side truncation)
    raw_inputs = [template_fn(p, r, tokenizer) for p, r in zip(prompts, responses)]
    if needs_chat_template:
        text_inputs = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": txt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for txt in raw_inputs
        ]
    else:
        text_inputs = raw_inputs

    # wildguard template has literal "<s>" — skip add_special_tokens to avoid BOS dup
    add_special = (judge_key != "wildguard")
    max_input_len = max_model_len - max_new_tokens
    n_truncated = 0
    vllm_token_ids: List[List[int]] = []
    for text in text_inputs:
        ids = tokenizer.encode(text, add_special_tokens=add_special)
        if len(ids) > max_input_len:
            ids = ids[:max_input_len]
            n_truncated += 1
        vllm_token_ids.append(ids)
    if n_truncated:
        log.warning("[%s] %d/%d inputs right-truncated to %d tokens",
                    judge_key, n_truncated, len(text_inputs), max_input_len)

    if use_hf:
        log.info("Loading judge model %s via transformers...", model_id)
        return [parse_fn(t) for t in _run_hf_judge(model_id, vllm_token_ids, max_new_tokens)]

    from vllm import LLM, SamplingParams
    log.info("Loading judge model %s via vLLM...", model_id)

    try:
        import os
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
        import vllm
        from packaging import version

        use_triton = os.environ.get("SAFETUNE_TRITON_ATTN", "1") == "1"
        attn_backend = "TRITON_ATTN" if use_triton else None

        llm_kwargs: dict = dict(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            dtype=dtype_name(device="cuda"),
            tensor_parallel_size=get_config().tensor_parallel_size,
        )
        # attention_backend set via VLLM_ATTENTION_BACKEND env var instead.

        from safetune.utils.errors import hf_access_errors
        with hf_access_errors(model_id, kind="judge model"):
            llm = LLM(**llm_kwargs)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        # Pass token IDs directly — bypasses vLLM's internal tokenization entirely
        vllm_inputs = [{"prompt_token_ids": ids} for ids in vllm_token_ids]
        outputs = llm.generate(vllm_inputs, sampling_params)
        raw_texts = [out.outputs[0].text for out in outputs]

    finally:
        # Always delete the model and free GPU memory, even if generation fails.
        try:
            del llm
        except NameError:
            pass
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return [parse_fn(t) for t in raw_texts]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_judge(
    judge_key: str,
    prompts: List[str],
    responses: List[str],
    gpu_memory_utilization: float = None,
) -> List[float]:
    """Score (prompt, response) pairs with the named judge.

    Args:
        judge_key: One of the keys in JUDGE_REGISTRY.
        prompts: List of input prompts (used as the "behavior" / "question"
            field in the judge template).
        responses: List of model responses, one per prompt.
        gpu_memory_utilization: Fraction of GPU memory to allocate to vLLM
            (default: runtime setting, else 0.85).

    Returns:
        List of floats, one per (prompt, response) pair.
        0.0 = refused / safe,  1.0 = complied / jailbroken / unsafe.

    Raises:
        KeyError: If judge_key is not in JUDGE_REGISTRY.
        ValueError: If prompts and responses have different lengths.
    Model judges run on vLLM when it is installed, otherwise on transformers.
    Registry entries can be overridden with ``configure(judges={...})``.
    """
    if judge_key not in JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown judge: {judge_key!r}. Available: {list(JUDGE_REGISTRY)}"
        )
    if len(prompts) != len(responses):
        raise ValueError(
            f"prompts and responses must have the same length "
            f"(got {len(prompts)} vs {len(responses)})"
        )
    if not prompts:
        return []

    if judge_key == "advbench":
        return _score_advbench(responses)

    gpu_memory_utilization = (gpu_memory_utilization
                              or get_config().gpu_memory_utilization or 0.85)
    return _run_vllm_judge(judge_key, prompts, responses, gpu_memory_utilization)


__all__ = ["JUDGE_REGISTRY", "run_judge"]
