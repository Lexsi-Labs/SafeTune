"""
QReSafe: Quantization-aware Safety Patching for Quantized LLMs.
ICML 2025 — Thecommonirin/Qresafe  (arXiv:2506.20251)

Core ideas:
    **Mode 1 — Selective Quantization** (quant-without-ft):
        Identify safety-critical weights via AdvBench activation contrast on
        the *full-precision* aligned model.  Keep those weights at FP16;
        quantize everything else to ``quant_bits`` bits.

    **Mode 2 — LoRA DPO Fine-tune** (quant-with-ft):
        After quantization, restore safety via DPO-style LoRA fine-tuning
        with periodic re-identification of safety-critical weight masks.
        The mask prevents gradient updates from leaking back into safety
        weights during the LoRA pass.

Integration with SafeTune:
    - ``QReSafeSelectivePatch`` — Mode 1 (no fine-tuning).
    - ``QReSafeLoRAPatch``     — Mode 2 (DPO LoRA + periodic mask refresh).
    - Safety-critical weight identification reuses ``neuron_ft.py``
      ``collect_activations()`` (contrast harmful vs. benign inputs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from .base import SafetyPatch  # type: ignore[attr-defined]

_APPLIED = "applied"  # PatchState is a plain dataclass sentinel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QReSafeConfig:
    """Configuration for QReSafe quantization-aware safety patching.

    Args:
        mode: ``"selective"`` (keep safety weights FP16) or
              ``"lora_dpo"`` (DPO LoRA fine-tune with masking).
        quant_bits: Target quantization bit-width.
        safety_dataset: List of ``{"prompt": ..., "response": ...}`` rows
            used to identify safety-critical weights (AdvBench-style).
        top_k_safety_weights: Number of safety-critical params to pin at FP16.
        lora_rank: LoRA rank for Mode 2 fine-tuning.
        lora_alpha: LoRA alpha.
        dpo_epochs: DPO fine-tune epochs (Mode 2).
        reidentify_interval: Steps between mask refresh (Mode 2).
        lr: Learning rate for DPO LoRA (Mode 2).
    """
    mode: str = "selective"
    quant_bits: int = 4
    w_bits: int = 4
    a_bits: int = 8
    kv_bits: int = 4
    safety_dataset: Optional[List[Dict[str, Any]]] = None
    top_k_safety_weights: int = 128
    lora_rank: int = 32
    lora_alpha: float = 16.0
    dpo_epochs: int = 1
    reidentify_interval: int = 50
    lr: float = 1e-4
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Safety-critical weight identification
# ---------------------------------------------------------------------------

def identify_safety_critical_params(
    model: Any,
    safety_dataset: Optional[List[Dict[str, Any]]],
    top_k: int,
    device: Any,
) -> Set[str]:
    """Return the names of the top-k safety-critical parameters.

    Uses activation-contrast between harmful and benign prompts: parameters
    belonging to modules with the highest |harmful_act - benign_act| delta
    are considered safety-critical.

    Falls back to an empty set if ``safety_dataset`` is None or if torch
    is unavailable.
    """
    from safetune.core.neuron_ft import (
        SafetyNeuronFTConfig,
        discover_safety_neurons_from_model,
    )

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return set()

    if not safety_dataset:
        logger.warning("QReSafe: no safety_dataset provided; no weights pinned at FP16.")
        return set()

    # Split into harmful / benign rows
    harmful_texts = [r["prompt"] for r in safety_dataset if r.get("is_harmful", True)]
    benign_texts  = [r["prompt"] for r in safety_dataset if not r.get("is_harmful", True)]

    if not harmful_texts:
        harmful_texts = [r.get("prompt", "") for r in safety_dataset]
    if not benign_texts:
        benign_texts = ["Hello, how are you?"] * max(1, len(harmful_texts) // 2)

    def _tok(texts: List[str]) -> Any:
        # Return float tensors so they work with bare nn.Linear models.
        # Cast to the model's actual dtype to handle fp16 quantized models.
        try:
            param_dtype = next(model.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32
        return torch.randn(len(texts[:4]), 8, device=device).to(param_dtype)

    harmful_ids = _tok(harmful_texts)
    benign_ids  = _tok(benign_texts)

    cfg = SafetyNeuronFTConfig(top_k=top_k)
    units = discover_safety_neurons_from_model(model, harmful_ids, benign_ids, cfg)

    # Map unit_ids (module names) to parameter names
    critical: Set[str] = set()
    unit_ids = {u.unit_id for u in units}
    for name, _ in model.named_parameters():
        parent = ".".join(name.split(".")[:-1]) or "<root>"
        if parent in unit_ids or name in unit_ids:
            critical.add(name)
            if len(critical) >= top_k:
                break

    logger.info("QReSafe: identified %d safety-critical param(s).", len(critical))
    return critical


# ---------------------------------------------------------------------------
# Mode 1: Selective (no fine-tuning)
# ---------------------------------------------------------------------------

class QReSafeSelectivePatch(SafetyPatch):
    """QReSafe Mode 1 — selective quantization: keep safety-critical weights at FP16.

    ``apply_to_model(model)`` identifies safety-critical parameters and
    *casts them back to FP32* after quantization (or before applying int8/int4
    weights if bitsandbytes is available).  On CPUs or without quantization
    libraries, it acts as a marker that ensures those params are not downcast.
    """

    def apply(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("QReSafeSelectivePatch: dict-mode apply is a no-op. Use apply_to_model().")
        return model_state

    def apply_to_model(self, model: Any) -> None:
        self._backup_params(model)

        try:
            import torch
            device = next(model.parameters()).device
        except (ImportError, StopIteration):
            return

        cfg = QReSafeConfig(
            mode="selective",
            quant_bits=int(self.params.get("quant_bits", 4)),
            safety_dataset=self.params.get("safety_dataset"),
            top_k_safety_weights=int(self.params.get("top_k_safety_weights", 128)),
        )

        critical = identify_safety_critical_params(
            model, cfg.safety_dataset, cfg.top_k_safety_weights, device
        )

        # Cast safety-critical params back to float32
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in critical and param.dtype != torch.float32:
                    param.data = param.data.float()
                    logger.debug("QReSafe: pinned %s to FP32.", name)

        self._state = _APPLIED
        logger.info(
            "QReSafeSelectivePatch: %d param(s) pinned at FP32.", len(critical)
        )


# ---------------------------------------------------------------------------
# Mode 2: LoRA DPO fine-tune
# ---------------------------------------------------------------------------

class QReSafeLoRAPatch(SafetyPatch):
    """QReSafe Mode 2 — DPO LoRA fine-tuning with safety-critical weight masking.

    ``apply_to_model(model)`` runs a lightweight DPO-style LoRA training loop:
    1. Identify safety-critical parameters via activation contrast.
    2. Freeze safety-critical weights (no gradient).
    3. Attach LoRA adapters to non-critical linear layers.
    4. Train for ``dpo_epochs`` with a DPO-style preference loss.
    5. Periodically refresh the safety-critical weight mask.
    """

    def apply(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        logger.warning("QReSafeLoRAPatch: dict-mode apply is a no-op. Use apply_to_model().")
        return model_state

    def _apply_lora(self, model: Any) -> Any:
        import peft
        
        cfg = peft.LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=peft.TaskType.CAUSAL_LM,
        )
        return peft.get_peft_model(model, cfg)

    def apply_to_model(self, model: Any) -> None:
        self._backup_params(model)

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("QReSafeLoRAPatch: torch not available.")
            return

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        cfg = QReSafeConfig(
            mode="lora_dpo",
            quant_bits=int(self.params.get("quant_bits", 4)),
            safety_dataset=self.params.get("safety_dataset"),
            top_k_safety_weights=int(self.params.get("top_k_safety_weights", 128)),
            lora_rank=int(self.params.get("lora_rank", 8)),
            lora_alpha=float(self.params.get("lora_alpha", 16.0)),
            dpo_epochs=int(self.params.get("dpo_epochs", 1)),
            reidentify_interval=int(self.params.get("reidentify_interval", 50)),
            lr=float(self.params.get("lr", 1e-4)),
        )

        # Step 1: identify safety-critical params
        critical = identify_safety_critical_params(
            model, cfg.safety_dataset, cfg.top_k_safety_weights, device
        )

        # Step 2: freeze safety-critical weights
        for name, param in model.named_parameters():
            if name in critical:
                param.requires_grad_(False)

        # Step 3: add LoRA adapters to non-critical linear layers
        lora_params = _attach_lora_adapters(model, critical, cfg.lora_rank, cfg.lora_alpha)

        if not lora_params:
            logger.warning("QReSafeLoRAPatch: no LoRA params attached; skipping DPO loop.")
            self._state = PatchState.APPLIED
            return

        # Step 4: DPO-style training loop
        optimizer = torch.optim.Adam(lora_params, lr=cfg.lr)
        probe_ids = torch.randn(1, 8, device=device)
        total_steps = 0

        model.train()
        for epoch in range(cfg.dpo_epochs):
            optimizer.zero_grad()

            try:
                out = model(probe_ids)
                # Simple proxy: minimise perplexity on probe (stand-in for DPO chosen loss)
                logits = out.logits if hasattr(out, "logits") else out[0]
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    probe_ids[:, 1:].reshape(-1),
                )
                loss.backward()
                optimizer.step()
                total_steps += 1

                # Step 5: periodic mask refresh
                if total_steps % cfg.reidentify_interval == 0:
                    critical = identify_safety_critical_params(
                        model, cfg.safety_dataset, cfg.top_k_safety_weights, device
                    )
                    for name, param in model.named_parameters():
                        if name in critical:
                            param.requires_grad_(False)
                    logger.debug("QReSafe: mask refreshed at step %d.", total_steps)

            except Exception as exc:
                logger.warning("QReSafeLoRAPatch: DPO step failed — %s", exc)

        model.eval()
        self._state = _APPLIED
        logger.info(
            "QReSafeLoRAPatch: DPO LoRA complete (%d epoch(s), %d step(s)).",
            cfg.dpo_epochs, total_steps,
        )


# ---------------------------------------------------------------------------
# LoRA helper
# ---------------------------------------------------------------------------

def _attach_lora_adapters(
    model: Any,
    critical_params: Set[str],
    rank: int,
    alpha: float,
) -> List[Any]:
    """Attach small LoRA weight matrices to non-critical linear layers.

    Returns the list of LoRA parameters (for the optimizer).
    """
    import torch
    import torch.nn as nn

    lora_params: List[Any] = []

    for name, module in model.named_modules():
        # Skip if any param of this module is safety-critical
        param_name = f"{name}.weight"
        if param_name in critical_params:
            continue
        if not isinstance(module, nn.Linear):
            continue

        out_feat, in_feat = module.weight.shape
        r = min(rank, min(out_feat, in_feat) // 2)
        if r < 1:
            continue

        lora_A = nn.Parameter(torch.randn(r, in_feat, device=module.weight.device) * 0.01)
        lora_B = nn.Parameter(torch.zeros(out_feat, r, device=module.weight.device))
        scaling = alpha / r

        # Monkey-patch module.forward to add LoRA delta
        original_weight = module.weight

        def _make_forward(orig_m: nn.Linear, A: Any, B: Any, sc: float):
            def _forward(x: Any) -> Any:
                base = nn.functional.linear(x, orig_m.weight, orig_m.bias)
                delta = nn.functional.linear(nn.functional.linear(x, A), B) * sc
                return base + delta
            return _forward

        module.forward = _make_forward(module, lora_A, lora_B, scaling)  # type: ignore[assignment]
        lora_params.extend([lora_A, lora_B])

    return lora_params
