"""RESTA: safety re-alignment via task arithmetic (training-free).

Implements RESTA from Bhardwaj et al., ACL 2024 ("Language Models are Homer
Simpson! Safety Re-Alignment of Fine-tuned Language Models through Task
Arithmetic", arXiv:2402.11746; repo https://github.com/declare-lab/resta).

RESTA adds a *safety vector* ``v = theta_aligned - theta_unaligned`` to a
compromised fine-tuned model: ``theta_safe = theta_finetuned + alpha * v``.
The paper's headline contribution is pairing this with **DARE**
(Drop-And-Rescale, Yu et al. 2024) sparsification of the safety vector before
addition, which reduces interference with the fine-tuned task. The RESTA repo
performs the safety-vector addition with ``mergekit``; the optional ``dare``
mode here reproduces mergekit's ``dare`` pre-processing step: randomly drop a
fraction ``p`` of the safety-vector entries and rescale the survivors by
``1 / (1 - p)`` so the expected magnitude is preserved.

The bare ``dare=False`` path is the RESTA-without-DARE baseline.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ._invariant import assert_mutates


def _dare_drop_and_rescale(
    safety_vector: Dict[str, torch.Tensor],
    drop_rate: float,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Apply DARE (Drop-And-Rescale) to a safety vector."""
    if not (0.0 <= drop_rate < 1.0):
        raise ValueError(f"dare drop_rate must be in [0, 1), got {drop_rate}")
    if drop_rate == 0.0:
        return {k: v.clone() for k, v in safety_vector.items()}

    keep_prob = 1.0 - drop_rate
    rescale = 1.0 / keep_prob
    out: Dict[str, torch.Tensor] = {}
    
    # Determine generator device fallback
    gen_device = generator.device if generator is not None else 'cpu'

    for key, delta in safety_vector.items():
        orig_dtype = delta.dtype
        # Calculate in fp32 to avoid precision issues
        delta_fp32 = delta.float()
        
        # Generate random values on the generator's device, then move to delta's device.
        # Using torch.rand is typically faster and safer for cross-device masking than bernoulli
        rand_vals = torch.rand(delta_fp32.shape, generator=generator, device=gen_device)
        mask = (rand_vals < keep_prob).to(delta.device)
        
        # Apply mask and rescale, then immediately downcast to save memory
        out[key] = (delta_fp32 * mask * rescale).to(orig_dtype)
        
    return out


@assert_mutates("apply_resta")
def apply_resta(
    finetuned: nn.Module,
    base: nn.Module,
    aligned: nn.Module,
    alpha: float = 1.0,
    param_filter: Optional[list] = None,
    dare: bool = False,
    dare_drop_rate: float = 0.9,
    dare_seed: Optional[int] = None,
) -> nn.Module:
    """Apply the RESTA safety vector ``(aligned - base)`` to a fine-tuned model.

    Computes the safety vector ``v = theta_aligned - theta_base`` and applies
    ``theta_safe = theta_finetuned + alpha * v``. Mutates ``finetuned``
    in-place via ``load_state_dict`` and returns it.

    Args:
        finetuned: the compromised fine-tuned model to re-align (mutated).
        base: the unaligned / base model (``theta_unaligned``).
        aligned: the safety-aligned model (``theta_aligned``).
        alpha: scaling coefficient ``b`` for the safety-vector addition.
        param_filter: optional substring include-filter over parameter names.
        dare: if True, apply DARE (Drop-And-Rescale) sparsification to the
            safety vector before adding it -- the paper's central technique for
            reducing interference with the fine-tuned task. If False (default,
            for backward compatibility) the plain RESTA-without-DARE addition
            is performed.
        dare_drop_rate: DARE drop probability ``p`` (default ``0.9``, the value
            used by mergekit's ``dare`` mode and the RESTA paper's experiments).
        dare_seed: optional seed for reproducible DARE drop masks.

    Returns:
        The mutated ``finetuned`` model.
    """
    try:
        from safetune.core.optim.resta import RESTAConfig, RESTAWrapper
    except ImportError as e:  # pragma: no cover - defensive
        raise ImportError(
            f"apply_resta needs safetune.core.optim.resta: {e}"
        ) from e

    cfg = RESTAConfig(alpha=alpha, param_filter=param_filter or [])
    wrapper = RESTAWrapper(
        aligned_state_dict=aligned.state_dict(),
        base_state_dict=base.state_dict(),
        config=cfg,
    )

    if dare:
        # Reproduce the RESTA repo's mergekit ``dare`` pre-processing: drop a
        # fraction of the safety-vector entries and rescale the survivors,
        # then add the sparsified vector with theta_ft + alpha * v_dare.
        generator: Optional[torch.Generator] = None
        if dare_seed is not None:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int(dare_seed))
        safety_vector = wrapper.get_safety_vector()
        dare_vector = _dare_drop_and_rescale(
            safety_vector, drop_rate=dare_drop_rate, generator=generator
        )
        ft_sd = finetuned.state_dict()
        new_sd: Dict[str, torch.Tensor] = {}
        for key, val in ft_sd.items():
            if key in dare_vector:
                dare_device = dare_vector[key].to(val.device)
                merged = val.float() + alpha * dare_device
                new_sd[key] = merged.to(val.dtype)
            else:
                new_sd[key] = val
    else:
        new_sd = wrapper.apply(finetuned.state_dict(), alpha=alpha)

    finetuned.load_state_dict(new_sd, strict=False)
    return finetuned


__all__ = ["apply_resta"]
