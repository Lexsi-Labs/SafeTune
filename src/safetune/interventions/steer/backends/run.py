"""``safetune.steer.run`` — one entry point to generate from a steered model.

A SafeTune steering method gives you a *steered model* (a wrapper such as
:class:`~safetune.steer.RefusalDirectionModel`, ``CAAModel``, ``SCANSModel`` …).
:func:`run` takes that wrapper plus prompts and returns generated text, choosing
*how* to execute it via the ``backend`` argument:

================  =========================================================
``backend``       behaviour
================  =========================================================
``"hf"``          Reference path. Installs the wrapper's hooks (via its
                  context-manager protocol) and generates with
                  ``transformers``. Always available.
``"vllm-hook"``   Activation steering served inside vLLM through the
                  IBM/vLLM-Hook worker. ~6× faster. Requires ``vllm`` and a
                  base ``model_id`` (vLLM loads weights from an id/path).
``"vllm-logits"`` Decoding-steering methods served inside vLLM through a V1
                  batch-level ``LogitsProcessor``. Pass a
                  :class:`~safetune.steer.backends.DecodeSteerSpec` as
                  ``decode_spec`` and a base ``model_id``.
================  =========================================================

To bake activation steering permanently into the weights instead of running
hooks (the "weight-fold" path), use
:func:`safetune.steer.orthogonalize_weights` and then serve the resulting
checkpoint with plain ``transformers``/vLLM — no backend needed.

Note on input-conditional methods: ``SCANSModel`` / ``AdaSteerModel`` choose a
per-prompt steering sign/coefficient inside their own ``generate()``. The
``hf`` backend installs their hooks but generates uniformly; for full
faithfulness call the wrapper's own ``generate(prompt=...)`` directly. The
``vllm-hook`` backend bakes a *static* spec (documented in
``backends/vllm_hook.py``).
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

_BACKENDS = ("hf", "vllm-hook", "vllm-logits")


def _underlying_model(steered_model: Any) -> Any:
    """Return the raw ``nn.Module`` inside a SafeTune steer wrapper.

    SafeTune steer wrappers store the wrapped model on ``.model``; a plain
    ``nn.Module`` is returned as-is.
    """
    inner = getattr(steered_model, "model", None)
    if inner is not None and hasattr(inner, "forward"):
        return inner
    return steered_model


def render_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    apply_chat_template: bool,
) -> List[str]:
    """Render ``prompts`` for generation, applying the chat template if present.

    When ``apply_chat_template`` is set each prompt is formatted as a one-turn
    user message. A *base-model* tokenizer still exposes the
    ``apply_chat_template`` method but has ``chat_template=None``; calling it
    then raises ``ValueError``. So we only format as chat when a template is
    actually set, and otherwise fall back to the raw prompt (warning once).

    Shared by the ``hf``, ``vllm-hook`` and ``vllm-logits`` backends so the
    chat-template behaviour is identical across all three.
    """
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if apply_chat_template and not has_template:
        logger.warning(
            "apply_chat_template=True but the tokenizer has no chat_template "
            "(base model?); generating from the raw prompt. Pass "
            "apply_chat_template=False to silence this."
        )
    use_template = bool(apply_chat_template) and has_template
    rendered: List[str] = []
    for prompt in prompts:
        if use_template:
            rendered.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            ))
        else:
            rendered.append(prompt)
    return rendered


def _hf_generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    temperature: float,
    apply_chat_template: bool,
) -> List[str]:
    import torch

    if tokenizer is None:
        raise ValueError("backend='hf' needs a tokenizer (tokenizer=...).")
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - model with no params
        device = "cpu"

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)

    rendered = render_prompts(tokenizer, prompts, apply_chat_template)
    outputs: List[str] = []
    for text in rendered:
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=pad_id,
            )
        new_tokens = gen[0][enc["input_ids"].shape[1]:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return outputs


def run(
    steered_model: Any,
    prompts: Sequence[str],
    *,
    backend: str = "hf",
    tokenizer: Any = None,
    model_id: Optional[str] = None,
    decode_spec: Any = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    apply_chat_template: bool = True,
    **backend_kwargs: Any,
) -> List[str]:
    """Generate text from a steered model on ``prompts``.

    Args:
        steered_model: a SafeTune steer wrapper (``RefusalDirectionModel``,
            ``CAAModel`` …). For ``vllm-logits`` this argument is unused — pass
            ``None`` and supply ``decode_spec``.
        prompts: prompts to generate completions for.
        backend: one of ``"hf"`` / ``"vllm-hook"`` / ``"vllm-logits"``.
        tokenizer: required for ``backend="hf"``.
        model_id: base model HF id / path — required for the vLLM backends
            (vLLM loads weights from an id, not an ``nn.Module``).
        decode_spec: a :class:`DecodeSteerSpec` — required for ``vllm-logits``.
        max_new_tokens, temperature, apply_chat_template: generation controls.
        **backend_kwargs: forwarded to the vLLM backend constructor
            (e.g. ``gpu_memory_utilization``, ``max_model_len``).

    Returns:
        A list of generated strings, one per prompt.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")

    # ── hf: reference path ──────────────────────────────────────────────────
    if backend == "hf":
        model = _underlying_model(steered_model)
        # SafeTune steer wrappers are context managers: __enter__ installs the
        # hooks, __exit__ removes them. A plain nn.Module has neither.
        if hasattr(steered_model, "__enter__") and hasattr(steered_model, "__exit__"):
            with steered_model:
                return _hf_generate(
                    model, tokenizer, prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    apply_chat_template=apply_chat_template,
                )
        logger.warning(
            "steer.run(backend='hf'): %s is not a context-manager steer "
            "wrapper — generating without installing steering hooks.",
            type(steered_model).__name__,
        )
        return _hf_generate(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            apply_chat_template=apply_chat_template,
        )

    # ── vllm-hook: activation steering inside vLLM ──────────────────────────
    if backend == "vllm-hook":
        if model_id is None:
            raise ValueError(
                "backend='vllm-hook' needs model_id=... (vLLM loads the base "
                "model weights from an HF id / local path)."
            )
        from .vllm_hook import VLLMHookSteer, extract_steer_spec

        spec = extract_steer_spec(steered_model)
        runner = VLLMHookSteer(model_id, spec, **backend_kwargs)
        return runner.generate(
            prompts,
            apply_chat_template=apply_chat_template,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

    # ── vllm-logits: decoding steering inside vLLM ──────────────────────────
    if decode_spec is None:
        raise ValueError(
            "backend='vllm-logits' needs decode_spec=DecodeSteerSpec(...)."
        )
    if model_id is None:
        raise ValueError("backend='vllm-logits' needs model_id=... (the target).")
    from .vllm_logits import VLLMDecodeSteer

    runner = VLLMDecodeSteer(model_id, decode_spec, **backend_kwargs)
    return runner.generate(
        prompts,
        apply_chat_template=apply_chat_template,
        temperature=temperature,
        max_tokens=max_new_tokens,
    )


__all__ = ["run"]
