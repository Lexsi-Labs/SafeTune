"""RepNoise-based post-hoc recovery (Rosati et al., NeurIPS 2024).

Rosati et al. showed that injecting structured noise into the *harmful
representation subspace* during training prevents safety fine-tuning attacks
("Representation Noising: A Defence Mechanism Against Harmful Finetuning").
This module adapts the core mechanism as a **post-hoc weight-space recovery**
technique for already-drifted models — no retraining is required.

Algorithm (post-hoc adaptation of the NeurIPS 2024 paper)
----------------------------------------------------------
1. Run the drifted model on a harmful calibration set; collect the intermediate
   hidden-state tensors from the target transformer layers.

2. Per layer, extract the top-``subspace_rank`` directions of the harmful
   representation subspace via truncated SVD:

       H_harm ∈ ℝ^{N × d}  (N = n_tokens across calibration set, d = hidden_dim)
       U_harm, _, _ = svd(H_harm)
       U_harm = U_harm[:, :subspace_rank]   # top-k left singular vectors

3. For each target-layer weight matrix W ∈ ℝ^{d_out × d_in}, inject Gaussian
   noise **restricted to the harmful subspace**:

       N_harm = U_harm @ torch.randn(subspace_rank, d_in)
       W_new  = W + noise_scale * N_harm / ||N_harm||_F

   The Frobenius normalisation keeps the noise magnitude independent of the
   rank. This degrades the model's ability to activate the harmful directions
   while leaving orthogonal (benign) directions intact — the same mechanism
   RepNoise uses, but applied directly to the weights rather than the training
   loss.

No retraining or labelled safe data is required; only harmful calibration
prompts and ``noise_scale`` are needed.

vLLM backend
------------
Activation hooks for subspace extraction require PyTorch model internals
(vLLM does not expose per-layer hidden states). When ``vllm_engine`` is
supplied (a ``vllm.LLM`` instance), it is used to *generate* full harmful
continuations from the calibration prompts, providing a richer (prompt +
model continuation) representation signal for the subspace computation.
When ``None``, only the input prompts are forwarded through the HF model.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


def _collect_harmful_representations(
    model: nn.Module,
    harmful_inputs: List[torch.Tensor],
    layer_indices: List[int],
    max_samples: int,
) -> Dict[int, torch.Tensor]:
    """Run harmful inputs through the model and collect hidden states.

    Returns dict: layer_idx → hidden_state matrix  (N_tok × d_model).
    Hooks on ``model.model.layers[i]`` output (the residual stream after each
    transformer block), compatible with LLaMA / Mistral architectures.
    """
    reps: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_indices}

    # Resolve the layers container (try both HF naming conventions).
    layers_container = None
    for attr in ("model", "transformer", "base_model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            layers_container = getattr(inner, "layers", None) or \
                               getattr(inner, "h", None) or \
                               getattr(inner, "blocks", None)
            if layers_container is not None:
                break

    if layers_container is None:
        # Fallback: directly iterate named modules looking for "layers.*" patterns.
        logger.warning(
            "repnoise_recover: could not resolve transformer layers container; "
            "no representations collected."
        )
        return {}

    hooks = []
    for idx in layer_indices:
        if idx >= len(layers_container):
            continue
        layer = layers_container[idx]

        def _make_hook(layer_idx: int):
            def hook(module: nn.Module, inp: Any, out: Any) -> None:
                # out is typically (hidden_state,) or a ModelOutput
                hs = out[0] if isinstance(out, (tuple, list)) else \
                     getattr(out, "last_hidden_state", out)
                if isinstance(hs, torch.Tensor):
                    # Flatten to (n_tokens, d_model)
                    reps[layer_idx].append(hs.detach().float().reshape(-1, hs.size(-1)))
            return hook

        hooks.append(layer.register_forward_hook(_make_hook(idx)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for ids in harmful_inputs[:max_samples]:
                ids = ids.to(next(model.parameters()).device)
                try:
                    model(input_ids=ids)
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("repnoise_recover: forward failed: %s", e)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    # Concatenate collected representations.
    result: Dict[int, torch.Tensor] = {}
    for idx, tensors in reps.items():
        if tensors:
            result[idx] = torch.cat(tensors, dim=0)  # (N_tok, d_model)
    return result


def _harmful_subspace(
    H: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Top-``rank`` right singular vectors of the harmful representation matrix H.

    Returns V ∈ ℝ^{d_model × rank}.
    """
    n, d = H.shape
    eff_rank = min(rank, n, d)
    try:
        # Economy SVD: We need the feature dimension (d_model), which corresponds
        # to the RIGHT singular vectors (Vh), not the left (U).
        _, _, Vh = torch.linalg.svd(H, full_matrices=False)
        
        # Vh is (min(n,d), d). Transpose to get columns as basis vectors (d, min(n,d))
        return Vh.T[:, :eff_rank]  # (d, eff_rank)
    except Exception as e:  # pragma: no cover - degenerate matrix
        logger.debug("repnoise_recover: SVD failed (%s); using random subspace.", e)
        V, _ = torch.linalg.qr(torch.randn(d, eff_rank, device=H.device))
        return V


@assert_mutates("apply_repnoise_recover")
def apply_repnoise_recover(
    model: nn.Module,
    harmful_inputs: Sequence[torch.Tensor],
    *,
    noise_scale: float = 0.01,
    subspace_rank: int = 8,
    target_layers: Optional[List[int]] = None,
    target_modules: Optional[List[str]] = None,
    max_samples: int = 32,
    seed: Optional[int] = None,
    vllm_engine: Any = None,
) -> nn.Module:
    """Inject harmful-subspace noise into the drifted model's weight matrices.

    Parameters
    ----------
    model:
        The drifted model to patch (mutated in-place).
    harmful_inputs:
        Sequence of ``input_ids`` tensors (shape ``(1, T)`` each) from
        tokenised harmful prompts, used to drive the subspace extraction.
    noise_scale:
        Frobenius-normalised noise magnitude added per target weight matrix.
        Larger values increase safety recovery at the cost of capability;
        paper-range: 0.001 – 0.05.
    subspace_rank:
        Dimensionality of the harmful representation subspace (top-k singular
        directions of the harmful hidden-state matrix).  Default: 8.
    target_layers:
        Transformer layer indices to extract representations from and to target
        with noise injection.  Default: middle third of the model (where the
        refusal/harm circuit literature locates safety-relevant neurons).
    target_modules:
        Substrings of weight matrix names to inject noise into within each
        target layer (default: ``["mlp.down_proj", "self_attn.o_proj"]``).
    max_samples:
        Maximum number of harmful calibration inputs used for subspace
        extraction.
    seed:
        Optional random seed for reproducible noise draws.
    vllm_engine:
        Optional ``vllm.LLM`` instance.  When supplied, used to generate full
        harmful continuations from the input prompts before subspace
        extraction, enriching the representation signal.

    Returns
    -------
    The mutated ``model``.
    """
    if noise_scale <= 0.0:
        raise ValueError(f"noise_scale must be > 0, got {noise_scale}")

    device = next(model.parameters()).device

    # Infer number of layers for target-layer defaulting.
    n_layers: int = 0
    for attr in ("model", "transformer", "base_model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            lc = getattr(inner, "layers", None) or getattr(inner, "h", None)
            if lc is not None:
                n_layers = len(lc)
                break

    if target_layers is None:
        if n_layers > 0:
            lo = n_layers // 3
            hi = 2 * n_layers // 3
            target_layers = list(range(lo, hi))
        else:
            # Cannot determine layer count; target layers 8–20 as a safe default.
            target_layers = list(range(8, 21))

    modules_filter = target_modules or ["mlp.down_proj", "self_attn.o_proj"]

    inputs = list(harmful_inputs) if harmful_inputs is not None else []

    # Optional vLLM continuation augmentation.
    if vllm_engine is not None and inputs:
        try:
            from vllm import SamplingParams  # type: ignore[import-not-found]
            tok = getattr(vllm_engine, "tokenizer", None) or getattr(model, "tokenizer", None)
            if tok is not None:
                texts = [tok.decode(ids[0].tolist(), skip_special_tokens=True)
                         for ids in inputs[:max_samples]]
                sp = SamplingParams(max_tokens=64, temperature=0.8)
                results = vllm_engine.generate(texts, sp)
                aug_texts = [
                    texts[i] + " " + r.outputs[0].text
                    for i, r in enumerate(results) if r.outputs
                ]
                aug_ids = [
                    tok(t, return_tensors="pt", truncation=True, max_length=256)["input_ids"]
                    for t in aug_texts
                ]
                inputs = aug_ids + inputs
                logger.info(
                    "repnoise_recover: vLLM augmented calibration to %d inputs.", len(inputs)
                )
        except Exception as e:  # pragma: no cover - optional
            logger.debug("repnoise_recover: vLLM augmentation skipped: %s", e)

    # Collect harmful representations.
    logger.info(
        "repnoise_recover: extracting harmful subspace from %d calibration inputs "
        "at layers %s.", min(len(inputs), max_samples), target_layers
    )
    layer_reps = _collect_harmful_representations(model, inputs, target_layers, max_samples)

    if not layer_reps:
        logger.warning(
            "repnoise_recover: no representations collected (model architecture "
            "may not use the expected layer naming); no noise injected."
        )
        return model

    if seed is not None:
        torch.manual_seed(seed)

    injected_layers = 0
    injected_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            if not any(m in name for m in modules_filter):
                continue

            # Determine which layer this parameter belongs to.
            layer_idx: Optional[int] = None
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None or layer_idx not in layer_reps:
                continue

            H_harm = layer_reps[layer_idx].to(device)
            d_model = H_harm.size(1)

            # The weight matrix may have d_out != d_model; use the matching dim.
            # For down_proj (d_model × d_ffn) the output dim matches d_model.
            # For o_proj (d_model × d_heads) the output dim matches d_model.
            if param.size(0) != d_model:
                # Try projecting via input dim.
                if param.size(1) == d_model:
                    # Transpose convention: noise in the input (column) subspace.
                    U = _harmful_subspace(H_harm, min(subspace_rank, d_model))
                    # N_harm shape: (d_model, rank) → noise in row space
                    N_raw = (torch.randn(param.size(0), U.size(1), device=device) @ U.T.to(device))
                else:
                    logger.debug(
                        "repnoise_recover: param %s shape %s doesn't align with "
                        "d_model=%d; skipping.", name, param.shape, d_model
                    )
                    continue
            else:
                U = _harmful_subspace(H_harm, min(subspace_rank, d_model))
                # Noise projected into the harmful subspace.
                # N_harm ∈ ℝ^{d_out × d_in}: columns live in span(U).
                noise_dir = torch.randn(U.size(1), param.size(1), device=device)
                N_raw = U.to(device) @ noise_dir  # (d_out, d_in)

            # Frobenius-normalise so noise_scale is architecture-independent.
            frob = N_raw.norm() + 1e-8
            N_normalised = (noise_scale * N_raw / frob).to(param.dtype)

            param.data.add_(N_normalised)
            injected_layers += 1
            injected_params += N_normalised.numel()

    logger.info(
        "repnoise_recover: injected noise into %d weight matrices (%d params) "
        "across %d layers (noise_scale=%.4f, rank=%d).",
        injected_layers, injected_params, len(layer_reps), noise_scale, subspace_rank,
    )
    return model


__all__ = ["apply_repnoise_recover"]
