"""Safety-subspace and refusal-direction extraction utilities.

Pure PyTorch helpers that compute contrast-based hidden-state subspaces from
an ``nn.Module`` — used to seed inference-time steering methods and
training-time projection defences.
"""
from __future__ import annotations

from typing import Any, List

import torch


def _encode(
    model: Any,
    prompts: List[str],
    tokenizer: Any = None,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Return a ``(N, H)`` tensor of last-token hidden states at ``layer_idx``.

    If ``tokenizer`` is ``None``, ``prompts`` must already be tensor-like
    inputs compatible with ``model(**inputs)``.
    """
    model.eval()
    device = next(model.parameters()).device
    hiddens: List[torch.Tensor] = []

    for prompt in prompts:
        if tokenizer is not None and isinstance(prompt, str):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        elif isinstance(prompt, dict):
            inputs = {k: v.to(device) for k, v in prompt.items()}
        else:
            inputs = {"input_ids": prompt.to(device)}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        hs = getattr(out, "hidden_states", None)
        if hs is None:
            raise RuntimeError(
                "model(**inputs) did not return hidden_states; "
                "is the model a HuggingFace causal LM?"
            )
        # Select the requested layer (supports negative indexing).
        layer = hs[layer_idx]
        # Last-token hidden state: (1, H)
        last = layer[:, -1, :].detach().float().cpu()
        hiddens.append(last.squeeze(0))

    return torch.stack(hiddens, dim=0)  # (N, H)


def compute_safety_subspace(
    model: Any,
    safe_prompts: List[Any],
    unsafe_prompts: List[Any],
    *,
    rank: int = 32,
    layer_idx: int = -1,
    tokenizer: Any = None,
) -> torch.Tensor:
    """Compute a rank-``rank`` safety subspace via SVD on contrast deltas.

    Procedure:
        1. Collect last-token hidden states at ``layer_idx`` for both
           ``safe_prompts`` and ``unsafe_prompts``.
        2. Mean-center each group and build the contrast matrix
           ``Δ = [safe_centred; -unsafe_centred]``.
        3. Return the top-``rank`` left singular vectors (shape ``(H, rank)``).
    """
    if not safe_prompts or not unsafe_prompts:
        raise ValueError("compute_safety_subspace: both prompt sets must be non-empty.")

    safe = _encode(model, safe_prompts, tokenizer=tokenizer, layer_idx=layer_idx)
    unsafe = _encode(model, unsafe_prompts, tokenizer=tokenizer, layer_idx=layer_idx)

    safe_c = safe - safe.mean(dim=0, keepdim=True)
    unsafe_c = unsafe - unsafe.mean(dim=0, keepdim=True)

    contrast = torch.cat([safe_c, -unsafe_c], dim=0)  # (2N, H)

    # SVD of contrast.T (H, 2N) — left singular vectors span the subspace.
    U, _S, _Vh = torch.linalg.svd(contrast.T, full_matrices=False)
    k = min(rank, U.shape[1])
    return U[:, :k].contiguous()  # (H, k)


def compute_refusal_direction(
    model: Any,
    refusal_prompts: List[Any],
    comply_prompts: List[Any],
    *,
    layer_idx: int = -1,
    tokenizer: Any = None,
) -> torch.Tensor:
    """Return the unit vector pointing from compliance to refusal behaviour.

    Computes the normalised difference of mean last-token hidden states at
    ``layer_idx`` between refusal-eliciting and comply-eliciting prompts.
    """
    if not refusal_prompts or not comply_prompts:
        raise ValueError(
            "compute_refusal_direction: both prompt sets must be non-empty."
        )

    ref = _encode(model, refusal_prompts, tokenizer=tokenizer, layer_idx=layer_idx)
    com = _encode(model, comply_prompts, tokenizer=tokenizer, layer_idx=layer_idx)

    direction = ref.mean(dim=0) - com.mean(dim=0)  # (H,)
    norm = direction.norm()
    if norm > 1e-8:
        direction = direction / norm
    return direction


__all__ = ["compute_safety_subspace", "compute_refusal_direction"]
