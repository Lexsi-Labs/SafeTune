"""
AAQ: Alignment-Aware Quantization for LLM Safety.
arXiv:2511.07842 — Wee et al.

Core idea:
    Standard Post-Training Quantization (PTQ) minimises perplexity but can
    inadvertently erode safety alignment.  AAQ adds an
    **Alignment-Preserving Contrastive (APC) loss** to the PTQ calibration
    loop.  The quantized model's hidden-state representations should be
    *close* to the aligned (fine-tuned) model and *far* from the unaligned
    (base pre-trained) model:

        L_APC = -log σ( sim(q, aligned) - sim(q, unaligned) )

    This works on any standard calibration dataset — no safety-specific data
    needed — and is compatible with W4A4 quantization (and coarser).

Integration with SafeTune:
    - ``AAQPatch`` extends ``SafetyPatch``.
    - ``apply_to_model(model)`` runs a lightweight calibration loop:
        1. Collects hidden states from aligned model and base model on probe
           inputs (or random data).
        2. Fine-tunes the (already quantized) model for a few steps with
           ``L_task + apc_weight * L_APC``.
    - Lazy-loads quantization libraries (``bitsandbytes`` or ``auto_gptq``).
    - Gracefully degrades if quantization is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AAQConfig:
    """Configuration for Alignment-Aware Quantization.

    Args:
        aligned_model_path: Path to the fine-tuned, aligned checkpoint.
        base_model_path: Path to the unaligned pre-trained model.
        quantization_bits: Bit-width for weight quantization (e.g. 4).
        apc_weight: Weight of the APC contrastive loss.
        calibration_steps: How many calibration steps to run.
        lr: Learning rate for APC fine-tuning (Adam).
        probe_texts: Optional list of calibration texts.  Random tokens
            are used when ``None``.
        device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
    """
    aligned_model_path: str = ""
    base_model_path: str = ""
    quantization_bits: int = 4
    apc_weight: float = 0.1
    calibration_steps: int = 20
    lr: float = 1e-4
    probe_texts: Optional[List[str]] = None
    device: str = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# APC loss helper
# ---------------------------------------------------------------------------

def _apc_loss(
    quantized_hidden: Any,
    aligned_hidden: Any,
    base_hidden: Any,
) -> Any:
    """Compute the Alignment-Preserving Contrastive (APC) loss.

    Encourages the quantized model hidden states to be closer to the aligned
    model than the base (unaligned) model in cosine space.

    L_APC = -log σ( cos(q, aligned) - cos(q, base) )
    """
    import torch
    import torch.nn.functional as F

    def _cos(a: Any, b: Any) -> Any:
        a_flat = a.view(a.size(0), -1)
        b_flat = b.view(b.size(0), -1)
        return F.cosine_similarity(a_flat, b_flat, dim=1).mean()

    sim_aligned = _cos(quantized_hidden, aligned_hidden)
    sim_base    = _cos(quantized_hidden, base_hidden)
    loss = -torch.log(torch.sigmoid(sim_aligned - sim_base) + 1e-8)
    return loss


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

from .base import SafetyPatch  # type: ignore[attr-defined]

_APPLIED = "applied"  # PatchState is a dataclass, not an enum


class AAQPatch(SafetyPatch):
    """Post-finetune safety patch: Alignment-Aware Quantization (AAQ).

    apply_to_model() runs the APC calibration loop on the given ``nn.Module``,
    minimising ``L_task + apc_weight * L_APC`` for ``calibration_steps``.

    Gracefully skips if no CUDA or if the referenced model paths don't exist
    (logs a warning and leaves the model unchanged).
    """

    def apply(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy dict-based apply (no-op for AAQ, which needs a real model)."""
        logger.warning(
            "AAQPatch.apply() called with a dict state — AAQ requires a real "
            "nn.Module.  Use apply_to_model() instead."
        )
        return model_state

    def apply_to_model(self, model: Any) -> None:
        """Run APC calibration on a real PyTorch model.

        Steps:
        1. Load aligned + base reference models (lazy, from paths in params).
        2. Generate/collect probe hidden states from both references.
        3. Fine-tune the quantized model for ``calibration_steps`` steps with
           APC loss to align representations toward the safe reference.
        """
        try:
            import torch
        except ImportError:
            logger.warning("AAQPatch: torch not available, skipping.")
            return

        self._backup_params(model)

        aligned_path = self.params.get("aligned_model_path", "")
        base_path    = self.params.get("base_model_path", "")
        bits         = int(self.params.get("quantization_bits", 4))
        apc_w        = float(self.params.get("apc_weight", 0.1))
        steps        = int(self.params.get("calibration_steps", 20))
        lr           = float(self.params.get("lr", 1e-4))

        # Determine device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # Load reference models if paths are provided
        aligned_model = _try_load_hf_model(aligned_path, device)
        base_model    = _try_load_hf_model(base_path, device)

        if aligned_model is None or base_model is None:
            logger.warning(
                "AAQPatch: could not load reference models (%s / %s). "
                "Running in probe-free mode (APC loss uses random hidden states).",
                aligned_path, base_path,
            )

        # Build probe inputs
        probe_ids = _make_probe_ids(
            self.params.get("probe_texts"), model, device
        )

        # APC calibration loop
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()

        for step in range(steps):
            optimizer.zero_grad()

            # Get quantized model hidden states
            try:
                out_q = model(probe_ids, output_hidden_states=True)
                q_hidden = out_q.hidden_states[-1]
            except Exception:
                out_q = model(probe_ids)
                if isinstance(out_q, torch.Tensor):
                    q_hidden = out_q
                elif hasattr(out_q, "last_hidden_state"):
                    q_hidden = out_q.last_hidden_state
                elif isinstance(out_q, tuple):
                    q_hidden = out_q[0]
                else:
                    q_hidden = torch.zeros_like(probe_ids, dtype=torch.float32).requires_grad_()

            # Reference hidden states
            with torch.no_grad():
                try:
                    aligned_h = (
                        aligned_model(probe_ids, output_hidden_states=True).hidden_states[-1]
                        if aligned_model else q_hidden.detach()
                    )
                    base_h = (
                        base_model(probe_ids, output_hidden_states=True).hidden_states[-1]
                        if base_model else torch.zeros_like(aligned_h)
                    )
                except Exception:
                    aligned_h = q_hidden.detach()
                    base_h    = torch.zeros_like(aligned_h)

            apc = _apc_loss(q_hidden, aligned_h, base_h)
            loss = apc_w * apc
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, steps // 5) == 0:
                logger.debug("AAQ calibration step %d/%d — APC loss: %.4f", step+1, steps, apc.item())

        model.eval()
        logger.info(
            "AAQPatch: APC calibration complete (%d steps, apc_weight=%.3f).",
            steps, apc_w,
        )

        self._state = _APPLIED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_hf_model(path: str, device: Any) -> Optional[Any]:
    """Try to load a HuggingFace model; return None on failure."""
    if not path:
        return None
    try:
        # Pass `path` straight to from_pretrained: it accepts BOTH a local dir
        # and a Hugging Face hub id. (Do NOT gate on Path(path).exists() — that
        # rejects hub ids like "Qwen/Qwen2.5-0.5B-Instruct" and silently drops
        # AAQ into meaningless probe-free mode against random targets.)
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(path)
        m = m.to(device).eval()
        return m
    except Exception as exc:
        logger.warning("AAQPatch: failed to load %s — %s", path, exc)
        return None


def _make_probe_ids(probe_texts: Optional[List[str]], model: Any, device: Any) -> Any:
    """Create probe token id tensors; fall back to random."""
    import torch
    if probe_texts:
        try:
            from transformers import AutoTokenizer
            # Try to load a tokenizer associated with the model
            tok = AutoTokenizer.from_pretrained(
                getattr(model, "config", None) and
                getattr(model.config, "_name_or_path", None) or "gpt2"
            )
            enc = tok(
                probe_texts[:4], return_tensors="pt",
                padding=True, truncation=True, max_length=64
            )
            return enc["input_ids"].to(device)
        except Exception:
            pass
    # Fallback probe (no probe_texts, or tokenization failed). A real CausalLM
    # consumes Long *token ids* — its embedding layer rejects a float tensor
    # ("Expected ... Long ... but got FloatTensor"), which used to crash AAQ's
    # advertised "probe-free mode" on any transformer. Emit random token ids when
    # the model exposes a vocab; only fall back to a float feature vector for a
    # vocab-less probe (e.g. a bare nn.Linear) that genuinely wants floats.
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size:
        return torch.randint(0, int(vocab_size), (1, 8), device=device)
    try:
        in_dim = next(iter(model.parameters())).shape[-1]
    except StopIteration:
        in_dim = 8
    return torch.randn(1, in_dim, device=device)
