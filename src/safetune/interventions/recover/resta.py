"""RESTA: safety re-alignment via task arithmetic (training-free).

Implements RESTA from Bhardwaj et al., ACL 2024 ("Language Models are Homer
Simpson! Safety Re-Alignment of Fine-tuned Language Models through Task
Arithmetic", arXiv:2402.11746; repo https://github.com/declare-lab/resta).

RESTA adds a *safety vector* ``v = theta_aligned - theta_unaligned`` to a
compromised fine-tuned model: ``theta_safe = theta_finetuned + alpha * v``.
The paper pairs this with **DARE** (Drop-And-Rescale, Yu et al. 2024) on the
*fine-tuning* delta (section 2, Eq. 7-9; section 4: "DARE applies drop and rescale
to the delta parameters of the SFT model"): a fraction ``p`` of the entries of
``theta_finetuned - theta_aligned`` is dropped and the rest rescaled by
``1 / (1 - p)``, which removes redundant fine-tuning changes, and the safety
vector is then added whole (the paper's ``RESTA_d``)::

    theta_safe = theta_aligned + DARE(theta_finetuned - theta_aligned, p) + alpha * v

The bare ``dare=False`` path is the RESTA-without-DARE baseline.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ._invariant import assert_mutates
from ._contract import keyword_refs


def _dare_drop_and_rescale(
    delta: torch.Tensor,
    drop_rate: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply DARE (Drop-And-Rescale) to one fp32 safety-vector tensor."""
    if not (0.0 <= drop_rate < 1.0):
        raise ValueError(f"dare drop_rate must be in [0, 1), got {drop_rate}")
    if drop_rate == 0.0:
        return delta
    keep_prob = 1.0 - drop_rate
    # Draw on the generator's device (CPU by default) so a seed gives the same
    # mask on any device, then move the mask to the delta.
    gen_device = generator.device if generator is not None else "cpu"
    mask = (torch.rand(delta.shape, generator=generator, device=gen_device) < keep_prob)
    return delta * mask.to(delta.device) * (1.0 / keep_prob)


@assert_mutates("apply_resta")
@keyword_refs("base", "aligned", "alpha", "param_filter", "dare", "dare_drop_rate", "dare_seed",
              "device")
def apply_resta(
    finetuned: nn.Module,
    *,
    base: nn.Module,
    aligned: nn.Module,
    alpha: float = 1.0,
    param_filter: Optional[list] = None,
    dare: bool = False,
    dare_drop_rate: Optional[float] = None,
    dare_seed: Optional[int] = None,
    device: Optional[object] = None,
) -> nn.Module:
    """Apply the RESTA safety vector ``(aligned - base)`` to a fine-tuned model.

    Computes the safety vector ``v = theta_aligned - theta_base`` and applies
    ``theta_safe = theta_finetuned + alpha * v``. Mutates ``finetuned``
    in place and returns it.

    Streams one tensor at a time: for each parameter, ``aligned - base`` is
    computed in fp32, DARE is applied to it, ``alpha * delta`` is added to the
    fine-tuned weight in place (cast back to its dtype), and the delta is freed.
    With ``dare`` the fine-tuned weight's difference from ``aligned`` is dropped and
    rescaled first (see the module docstring); ``finetuned`` must then have been
    fine-tuned from ``aligned``.
    Extra memory is a few copies of the largest tensor, not of the model.

    Args:
        finetuned: the compromised fine-tuned model to re-align (mutated).
        base: the unaligned / base model (``theta_unaligned``).
        aligned: the safety-aligned model (``theta_aligned``).
        alpha: scaling coefficient ``b`` for the safety-vector addition.
        param_filter: optional substring include-filter over parameter names.
        dare: if True, apply DARE (Drop-And-Rescale) to the fine-tuning delta
            ``finetuned - aligned`` before the safety vector is added (the
            paper's ``RESTA_d``). If False (default, for backward
            compatibility) the plain RESTA addition is performed.
        dare_drop_rate: DARE drop probability ``p``. ``None``: 0.3, the RESTA
            paper's value ("we keep hyperparameters p = 0.3 and gamma = 0.5",
            Bhardwaj et al. 2024, Sec. 4), or the old 0.9 with
            ``safetune.configure(legacy_resta_drop_rate=True)``. At 0.9 (90% of
            entries dropped, the rest scaled by 10) Qwen2.5-0.5B's full
            base-to-instruct safety vector breaks the model (2/16 HarmBench
            refusals, garbled answers); at 0.3 it restores 13/16.
        dare_seed: optional seed for reproducible DARE drop masks.
        device: where each delta is computed. ``None`` (default): the
            fine-tuned weight's device. ``"cpu"`` keeps the extra memory off
            the GPU when ``base`` and ``aligned`` are on CPU.

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

    if dare_drop_rate is None:
        from safetune.config import get_config
        dare_drop_rate = 0.9 if get_config().legacy_resta_drop_rate else 0.3
    # ``dare``: drop a fraction of the fine-tuning delta's entries and rescale the survivors.
    generator: Optional[torch.Generator] = None
    if dare and dare_seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(dare_seed))

    aligned_sd = aligned.state_dict()
    ft_sd = finetuned.state_dict()  # views of the live weights
    # A tied weight (embed / lm_head) appears under several names. The old
    # load_state_dict path let the last name win; keep that (it matters under
    # DARE, where each name draws its own mask).
    keys = set(wrapper.keys)
    owner = {v.data_ptr(): k for k, v in ft_sd.items() if k in keys}
    with torch.no_grad():
        for key in wrapper.keys:  # aligned order: the DARE masks match a seed
            val = ft_sd.get(key)
            if val is None:
                continue
            delta = wrapper.delta(key, device if device is not None else val.device)
            if dare:
                # DARE acts on the fine-tuning delta (paper Eq. 7-9), not on the safety vector:
                # theta = aligned + DARE(ft - aligned) + alpha * (aligned - base).
                a = aligned_sd[key].to(delta.device).float()
                sft = _dare_drop_and_rescale(val.to(delta.device).float() - a,
                                             drop_rate=dare_drop_rate, generator=generator)
                new = a + sft + alpha * delta
            else:
                new = val.float() + alpha * delta.to(val.device)
            if owner[val.data_ptr()] != key:
                continue
            val.copy_(new.to(val.device).to(val.dtype))
            del delta, new
    return finetuned

__all__ = ["apply_resta"]
