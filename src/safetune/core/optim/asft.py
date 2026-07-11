"""
AsFT: Anchored Safety Fine-Tuning Within Narrow Safety Basin.
PKU-YuanGroup/AsFT

Decomposes parameter updates into safety-aligned (d_aligned) and orthogonal (d_perp)
components. Applies subspace regularization to suppress d_perp, keeping fine-tuning
within the "narrow safety basin."
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class AsFTConfig:
    """Configuration for AsFT subspace regularization."""
    # Regularization weight for the orthogonal component penalty
    reg_lambda: float = 0.1
    # Whether to fully zero out the orthogonal gradient component (hard constraint)
    hard_constraint: bool = False


class AsFTWrapper:
    """
    Anchored Safety Fine-Tuning: decomposes gradients into aligned and orthogonal
    components relative to the alignment direction.

    Usage::

        wrapper = AsFTWrapper(model, aligned_sd, base_sd, config)

        for batch in dataloader:
            loss = compute_loss(model, batch)
            loss.backward()
            with wrapper.apply_subspace_constraint():
                pass
            optimizer.step()
    """

    def __init__(
        self,
        model: Any,
        aligned_state_dict: Dict[str, Any],
        base_state_dict: Dict[str, Any],
        config: Optional[AsFTConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or AsFTConfig()
        self._alignment_dir: Dict[str, Any] = {}
        # Keep the aligned reference so the penalty can be computed on the
        # *update* Delta = theta_current - theta_aligned (see H5), not on the
        # raw parameter theta.
        self._aligned_sd: Dict[str, Any] = aligned_state_dict
        self._compute_alignment_direction(aligned_state_dict, base_state_dict)

    def _compute_alignment_direction(
        self,
        aligned_sd: Dict[str, Any],
        base_sd: Dict[str, Any],
    ) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError("AsFT requires PyTorch.")

        for name, param in self.model.named_parameters():
            if name not in aligned_sd or name not in base_sd:
                continue
            d = aligned_sd[name].float() - base_sd[name].float()
            norm = d.norm()
            if norm.item() > 0:
                self._alignment_dir[name] = d / norm
            else:
                self._alignment_dir[name] = d
        logger.info("AsFT: computed alignment direction for %d parameters.", len(self._alignment_dir))

    @contextmanager
    def apply_subspace_constraint(self) -> Iterator[None]:
        """
        Context manager: after backward(), decomposes gradients and suppresses
        the orthogonal component.
        """
        yield

        try:
            import torch
        except ImportError:
            return

        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self._alignment_dir:
                continue

            d_align = self._alignment_dir[name].to(param.grad.device).to(param.grad.dtype)
            g = param.grad.data

            # Project gradient onto alignment direction
            flat_g = g.view(-1)
            flat_d = d_align.view(-1)
            dot = torch.dot(flat_g, flat_d)

            # Aligned component
            g_aligned = dot * flat_d
            # Orthogonal component
            g_perp = flat_g - g_aligned

            if self.config.hard_constraint:
                # Zero out the orthogonal component entirely
                param.grad.data = g_aligned.view_as(g)
            else:
                # Soft: shrink the orthogonal component by reg_lambda
                param.grad.data = (g_aligned + (1.0 - self.config.reg_lambda) * g_perp).view_as(g)

    def compute_subspace_penalty(self) -> Any:
        """
        Optional: compute a scalar penalty for the orthogonal component magnitude.
        Add to loss before backward() for explicit regularization.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("AsFT requires PyTorch.")

        penalty = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name not in self._alignment_dir:
                continue
            if self._aligned_sd is None or name not in self._aligned_sd:
                continue
            d_align = self._alignment_dir[name].to(param.device).to(param.dtype)
            # Penalise the orthogonal component of the *update* relative to the
            # aligned reference, Delta = theta_current - theta_aligned -- NOT the
            # raw parameter theta.  At theta == theta_aligned, Delta == 0 so the
            # penalty and its gradient vanish (previously it was ~= lambda||theta||^2,
            # which destroys a from-aligned run at step 0).
            theta_aligned = self._aligned_sd[name].to(param.device).to(param.dtype)
            flat_delta = (param - theta_aligned).view(-1)
            flat_d = d_align.view(-1)
            dot = torch.dot(flat_delta, flat_d)
            delta_aligned = dot * flat_d
            delta_perp = flat_delta - delta_aligned
            penalty = penalty + delta_perp.norm().pow(2)
        return self.config.reg_lambda * penalty
