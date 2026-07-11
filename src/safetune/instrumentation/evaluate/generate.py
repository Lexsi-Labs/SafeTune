"""vLLM and HF generation backends for safety evaluation."""
from __future__ import annotations

import gc
import logging
import os
from typing import List, Optional

from .judges import JUDGE_REGISTRY  # re-export for convenience

_TRITON_ATTN = os.environ.get("SAFETUNE_TRITON_ATTN", "1") == "1"

log = logging.getLogger(__name__)


def _generate_vllm(
    model_path: str,
    formatted_prompts: List[str],
    *,
    max_new_tokens: int,
    temperature: float,
    gpu_memory_utilization: float,
    max_model_len: int,
    tokenizer_name: Optional[str] = None,
) -> List[str]:
    """Run generation with vLLM. Model is deleted after use."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for the vllm backend. Install with: pip install vllm"
        ) from exc

    llm_kwargs: dict = dict(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )
    if tokenizer_name:
        llm_kwargs["tokenizer"] = tokenizer_name
    # attention_backend set via VLLM_ATTENTION_BACKEND env var instead.
    # Passing as kwarg conflicts with env-var-based attention_config in vLLM >= 0.14.

    log.info("Loading generation model %s via vLLM...", model_path)
    try:
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        outputs = llm.generate(formatted_prompts, sampling_params)
        responses = [out.outputs[0].text for out in outputs]
    finally:
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

    return responses


def _generate_hf(
    model_path: str,
    formatted_prompts: List[str],
    *,
    max_new_tokens: int,
    temperature: float,
    tokenizer_name: Optional[str],
    batch_size: int = 8,
) -> List[str]:
    """Run generation with HuggingFace Transformers (device_map=auto, bfloat16)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the hf backend. "
            "Install with: pip install transformers"
        ) from exc

    tok_name = tokenizer_name or model_path
    log.info("Loading generation model %s via HF (device_map=auto)...", model_path)

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    if model.config.pad_token_id is None:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        model.config.pad_token_id = pad_id
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = pad_id

    responses: List[str] = []
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)
        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
        )
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        decoded = tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)
        responses.extend(decoded)

    return responses


def generate_responses(
    model_path: str,
    prompts: List[str],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    apply_chat_template: bool = True,
    tokenizer_name: Optional[str] = None,
    backend: str = "vllm",
) -> List[str]:
    """Generate responses for a list of raw prompts.

    Args:
        model_path: Path or HF Hub ID of the model to use for generation.
        prompts: List of raw user-facing prompts.
        max_new_tokens: Maximum number of tokens to generate per prompt.
        temperature: Sampling temperature. 0.0 = greedy / deterministic.
        gpu_memory_utilization: Fraction of GPU memory to allocate (vLLM only).
        max_model_len: Maximum total sequence length (vLLM only).
        apply_chat_template: When True, wrap each prompt as a user message via the
            model's chat template before generation. When False the prompt string is
            passed to the model verbatim.
        tokenizer_name: Override the tokenizer to use. Defaults to model_path.
        backend: ``"vllm"`` (default) or ``"hf"``.

    Returns:
        List of generated response strings (just the new tokens, not the prompt).
    """
    if not prompts:
        return []

    tok_name = tokenizer_name or model_path

    # Optionally format prompts through the model's chat template.
    if apply_chat_template:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required to apply the chat template. "
                "Install with: pip install transformers"
            ) from exc

        log.info("Applying chat template from %s...", tok_name)
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        formatted: List[str] = []
        for p in prompts:
            chat = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted.append(text)
    else:
        formatted = list(prompts)

    if backend == "vllm":
        return _generate_vllm(
            model_path,
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tokenizer_name=tok_name,
        )
    elif backend == "hf":
        return _generate_hf(
            model_path,
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            tokenizer_name=tok_name,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Supported: 'vllm', 'hf'.")


__all__ = ["generate_responses", "JUDGE_REGISTRY"]
