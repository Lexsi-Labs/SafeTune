"""
SafetyLoRA: LoRA-based safety alignment with optional circuit-guided targeting.

When circuit_guide is provided (from CircuitKIT), use it to set lora_target_modules
or layer subset; otherwise use default or "all-linear" style (LoRA without circuits).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .circuit_kit import get_circuit_info, CircuitInfo


@dataclass
class SafetyLoRAConfig:
    """Config for SafetyLoRA: LoRA params and optional circuit-guided modules."""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    circuit_guided: bool = False
    circuit_guide_path: Optional[str] = None
    circuit_info: Optional[CircuitInfo] = None

    def resolve_target_modules(
        self,
        default_all_linear: bool = False,
    ) -> List[str]:
        """
        Resolve final lora_target_modules: use circuit suggestions if available,
        else use configured lora_target_modules or "all-linear" if default_all_linear.
        """
        if self.circuit_guided and self.circuit_info and self.circuit_info.layer_suggestions:
            sug = self.circuit_info.layer_suggestions
            if sug.target_modules:
                return list(sug.target_modules)
        if default_all_linear and not self.lora_target_modules:
            return ["all-linear"]
        return list(self.lora_target_modules) if self.lora_target_modules else []

    def to_lora_kwargs(self, default_all_linear: bool = False) -> Dict[str, Any]:
        """Return dict suitable for model/trainer LoRA config (e.g. lora_r, lora_target_modules)."""
        target_modules = self.resolve_target_modules(default_all_linear=default_all_linear)
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": target_modules,
        }


def build_safety_lora_config(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    circuit_guided: bool = False,
    circuit_guide_path: Optional[str] = None,
    circuit_info: Optional[CircuitInfo] = None,
) -> SafetyLoRAConfig:
    """
    Build SafetyLoRA config. If circuit_guided and circuit_guide_path is set,
    loads circuit info from file and uses it for targeting when available.
    """
    info = circuit_info
    if circuit_guided and circuit_guide_path and info is None:
        info = get_circuit_info(source=circuit_guide_path)

    return SafetyLoRAConfig(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules or [],
        circuit_guided=circuit_guided,
        circuit_guide_path=circuit_guide_path,
        circuit_info=info,
    )


def merge_safety_lora_into_model_config(
    model_config: Dict[str, Any],
    safety_lora: SafetyLoRAConfig,
    default_all_linear: bool = False,
) -> Dict[str, Any]:
    """
    Merge SafetyLoRA resolved LoRA params into a model config dict.
    Used when a top-level safety_lora block is present in recipe config.
    """
    out = dict(model_config)
    lora_kw = safety_lora.to_lora_kwargs(default_all_linear=default_all_linear)
    for k, v in lora_kw.items():
        out[k] = v
    return out
