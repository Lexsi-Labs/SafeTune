"""
Spectral Entropy Monitor: detect steganographic / hidden-state-encoded jailbreaks.

Motivation: text-level safety filters cannot see what the model is doing
inside the residual stream. A fine-tuned model can develop "hidden"
behaviours where the surface output looks benign but the internal
representation has collapsed onto a low-rank subspace that encodes the
unsafe trajectory. Token-level detection misses this.

The Spectral Entropy Monitor watches the *information content* of the
residual stream itself. For every prompt we:

1. Run a forward pass with hooks attached to each decoder layer.
2. Capture the per-layer hidden-state tensor of shape ``(seq, hidden)``.
3. Compute the SVD and normalize squared singular values into a probability
   distribution ``p_i = sigma_i^2 / sum(sigma_j^2)``.
4. Compute the spectral entropy ``H = -sum(p_i log p_i)``.

H is bounded in ``[0, log(rank)]``. High H means the layer uses many
orthogonal directions (rich representation). Low H means the layer has
collapsed onto a few directions, the spectral fingerprint of:

* steganographic encoding (information packed into a low-rank subspace)
* refusal-direction "writing" (single direction dominates)
* mode collapse during generation

Usage::

    mon = SpectralEntropyMonitor(model, tokenizer)
    mon.calibrate(benign_prompts)
    flags = mon.scan(suspect_prompts)
    # flags is a list of (prompt_idx, layer_idx, z_score) for any prompt-layer
    # whose entropy is < (baseline mean - k * baseline std).

This is a defensive primitive. It produces no false sense of completeness:
not every jailbreak shows a spectral signature, and not every low-entropy
spike is malicious. Treat the output as a flag for human review, not a
binary verdict. Pair with text-level graders (e.g. StrongREJECT) for the
full picture.

Reference for the spectral-entropy formulation: this is the von Neumann
entropy of the empirical covariance matrix, a standard quantity in
representation analysis. The application to safety detection is novel
to this library.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from safetune._refusal_helpers import _get_decoder_layers

logger = logging.getLogger(__name__)


@dataclass
class SpectralMonitorConfig:
    """Configuration for the Spectral Entropy Monitor.

    Attributes:
        target_layers: Layers to monitor. ``None`` means all decoder layers.
            Deep layers are typically more discriminating; the paper-style
            default is "the last quarter" of layers.
        z_threshold: A prompt-layer pair is flagged when its entropy z-score
            against the calibrated baseline drops below ``-z_threshold``.
            Two standard deviations is a reasonable default.
        eigenvalue_floor: Numerical floor on singular values squared before
            normalization. Stops ``log(0)`` blowups on rank-deficient inputs.
        batch_size: Batch size for forward passes during calibration / scan.
    """

    target_layers: Optional[List[int]] = None
    z_threshold: float = 2.0
    eigenvalue_floor: float = 1e-12
    batch_size: int = 8


def _spectral_entropy(
    activation: torch.Tensor,
    floor: float,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Spectral entropy of a single ``(seq, hidden)`` activation tensor.

    Returns H in nats. The tensor is centered along the sequence axis so
    a uniform shift does not show up as low entropy.

    Args:
        activation: ``(seq, hidden)`` hidden-state tensor for one example.
        floor: numerical floor on squared singular values.
        mask: optional ``(seq,)`` attention mask. When provided, only the
            positions with mask == 1 (real, non-padding tokens) enter the
            SVD / entropy computation. This makes the metric independent of
            how many padding tokens the batch tokenizer appended, i.e.
            length- and batch-invariant. ``None`` keeps every row (legacy
            behaviour, only safe for unpadded single-example input).
    """
    if activation.dim() < 2:
        return 0.0
    a = activation.detach().float()
    if a.dim() > 2:
        a = a.reshape(-1, a.shape[-1])
    if mask is not None:
        m = mask.detach().reshape(-1).to(a.device)
        keep = m > 0.5
        # Guard the seq dimension only; if the mask length disagrees with the
        # activation rows (should not happen) fall back to no masking.
        if keep.numel() == a.shape[0]:
            a = a[keep]
    # Need at least two real tokens for a meaningful covariance spectrum.
    if a.shape[0] < 2:
        return 0.0
    a = a - a.mean(dim=0, keepdim=True)
    # Singular values of (seq, hidden) live on the residual stream's
    # principal axes; their squares are eigenvalues of A^T A.
    try:
        s = torch.linalg.svdvals(a)
    except RuntimeError:
        # Degenerate: zero rows or NaNs. Treat as zero entropy.
        return 0.0
    eig = s.pow(2).clamp_min(floor)
    p = eig / eig.sum()
    return float(-(p * p.log()).sum().item())


class SpectralEntropyMonitor:
    """Per-layer spectral entropy monitor for hidden states.

    Lifecycle:

    1. ``calibrate(benign_prompts)``: build per-layer mean and std of the
       spectral entropy distribution over a corpus of known-safe prompts.
    2. ``scan(prompts)``: run forward over each prompt, compute per-layer
       entropies, return a list of ``(prompt_idx, layer_idx, z_score)`` for
       layers whose entropy drops below ``-z_threshold`` standard deviations
       from the calibrated mean.
    3. ``entropy_trajectory(prompt)``: convenience for analysis. Returns
       ``{layer_idx: entropy}`` for a single prompt with no thresholding.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[SpectralMonitorConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SpectralMonitorConfig()
        self._captured: Dict[int, List[torch.Tensor]] = {}
        self._hooks: List[Any] = []
        self.baseline: Dict[int, Tuple[float, float]] = {}

    # ---------------------------------------------------------------- hooks

    def _make_hook(self, layer_idx: int):
        def hook(_m: nn.Module, _i: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            # Detach + CPU keep the GPU pool clear during long scans.
            self._captured.setdefault(layer_idx, []).append(h.detach().cpu())
        return hook

    def _register(self) -> None:
        self._remove()
        layers = _get_decoder_layers(self.model)
        if not layers:
            raise RuntimeError(
                "SpectralEntropyMonitor: could not locate decoder layers. "
                "Expected model.model.language_model.layers (Gemma-3), "
                "model.model.layers (Llama/Mistral/Qwen/Gemma), or "
                "model.transformer.h (GPT-style)."
            )
        target = self.config.target_layers
        if target is None:
            target = list(range(len(layers)))
        for idx in target:
            if 0 <= idx < len(layers):
                self._hooks.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def _remove(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    def _reset_capture(self) -> None:
        self._captured.clear()

    # ----------------------------------------------------------- core loop

    def _run_prompts(self, prompts: List[str]) -> List[Dict[int, float]]:
        """Return one ``{layer_idx: entropy}`` dict per prompt, in order."""
        self._reset_capture()
        self._register()
        # ``padding=True`` requires a pad token; many base-model tokenizers
        # ship without one. Fall back to the eos token (standard practice)
        # so batching does not crash. The attention mask still marks these
        # positions as padding, so they are excluded from the entropy.
        if getattr(self.tokenizer, "pad_token", None) is None:
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos is not None:
                self.tokenizer.pad_token = eos
        try:
            self.model.eval()
            per_prompt: List[Dict[int, float]] = []
            for i in range(0, len(prompts), self.config.batch_size):
                batch = prompts[i : i + self.config.batch_size]
                tokenized = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                device = next(self.model.parameters()).device
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                with torch.no_grad():
                    self.model(**tokenized)
                # Attention mask tells real tokens from padding. Without it,
                # padding rows (near-identical, low-rank) are folded into the
                # SVD and depress the entropy by a batch-composition-dependent
                # amount. Keep it on CPU alongside the captured activations.
                attn_mask = tokenized.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.detach().cpu()
                # For this batch, _captured has one tensor per hook of shape
                # (batch, seq, hidden). Convert into per-example entropies,
                # excluding padding positions via the attention mask.
                for b in range(len(batch)):
                    ent: Dict[int, float] = {}
                    row_mask = attn_mask[b] if attn_mask is not None else None
                    for layer_idx, acts in self._captured.items():
                        if not acts:
                            continue
                        # Last entry of acts is the most recent batch.
                        h = acts[-1][b]  # (seq, hidden)
                        ent[layer_idx] = _spectral_entropy(
                            h, self.config.eigenvalue_floor, mask=row_mask
                        )
                    per_prompt.append(ent)
                # Drop captured tensors for this batch to bound memory.
                self._reset_capture()
            return per_prompt
        finally:
            self._remove()

    # ------------------------------------------------------------ public API

    def calibrate(self, benign_prompts: List[str]) -> Dict[int, Tuple[float, float]]:
        """Compute per-layer (mean, std) of spectral entropy on benign prompts.

        Returns the baseline dict. Also stores it on ``self.baseline``.
        """
        if not benign_prompts:
            raise ValueError("calibrate() requires at least one benign prompt.")
        per_prompt = self._run_prompts(benign_prompts)

        # Aggregate per-layer.
        layer_values: Dict[int, List[float]] = {}
        for ent in per_prompt:
            for li, v in ent.items():
                layer_values.setdefault(li, []).append(v)

        baseline: Dict[int, Tuple[float, float]] = {}
        for li, vals in layer_values.items():
            t = torch.tensor(vals, dtype=torch.float64)
            mu = float(t.mean().item())
            sigma = float(t.std(unbiased=False).item()) if len(vals) > 1 else 1e-6
            baseline[li] = (mu, max(sigma, 1e-9))
        self.baseline = baseline
        logger.info(
            "SpectralEntropyMonitor: calibrated on %d prompts, %d layers.",
            len(benign_prompts),
            len(baseline),
        )
        return baseline

    def scan(self, prompts: List[str]) -> List[Tuple[int, int, float, float]]:
        """Scan prompts for low-entropy anomalies.

        Returns a list of ``(prompt_idx, layer_idx, entropy, z_score)`` for
        every prompt-layer pair whose z-score is below ``-z_threshold``.
        """
        if not self.baseline:
            raise RuntimeError("Call .calibrate(...) before .scan(...).")
        per_prompt = self._run_prompts(prompts)
        flags: List[Tuple[int, int, float, float]] = []
        for pi, ent in enumerate(per_prompt):
            for li, v in ent.items():
                if li not in self.baseline:
                    continue
                mu, sigma = self.baseline[li]
                z = (v - mu) / sigma
                if z < -self.config.z_threshold:
                    flags.append((pi, li, v, z))
        logger.info(
            "SpectralEntropyMonitor: scanned %d prompts, flagged %d (prompt, layer) pairs.",
            len(prompts),
            len(flags),
        )
        return flags

    def entropy_trajectory(self, prompt: str) -> Dict[int, float]:
        """Return ``{layer_idx: entropy}`` for a single prompt (no thresholding)."""
        per_prompt = self._run_prompts([prompt])
        return per_prompt[0] if per_prompt else {}


__all__ = [
    "SpectralMonitorConfig",
    "SpectralEntropyMonitor",
]
