"""
Unified Safety Configuration.

Single YAML-driven config that orchestrates all safety modules.
Provides a declarative interface for enabling / configuring
training-time, inference-time, and evaluation safety features.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TrainingSafetyConfig:
    """Training-time safety module configuration."""

    safedelta: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "strength": 0.5})
    asft: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "reg_lambda": 0.1})
    resta: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "alpha": 1.0})
    lox: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "rank": 64, "factor": 1.5})
    safety_layers: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "method": "cosine"})
    seal: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "retain_ratio": 0.8})
    door: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "use_weighted": True})
    repbend: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "strength": 0.3})
    tar: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "inner_steps": 4})


@dataclass
class InferenceSafetyConfig:
    """Inference-time safety module configuration."""

    adasteer: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "target_layers": []})
    scans: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    sta: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    safeswitch: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "threshold": 0.7})
    circuit_breaker: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "threshold": 0.5})
    cosa: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "categories": []})
    conversation_guard: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "window_size": 5})


@dataclass
class GuardrailConfig:
    """Production guardrail configuration."""

    input_sanitizer: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "detect_injection": True, "scrub_pii": True})
    output_verifier: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "check_toxicity": True})
    audit_logger: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "log_file": None})


@dataclass
class EvalSafetyConfig:
    """Evaluation safety configuration."""

    benchmarks: List[str] = field(default_factory=lambda: ["harmbench", "xstest"])
    min_safety_score: float = 0.80
    alert: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "max_prompts": 100})
    owasp: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})


@dataclass
class UnifiedSafetyConfig:
    """Unified safety configuration encompassing all modules."""

    training: TrainingSafetyConfig = field(default_factory=TrainingSafetyConfig)
    inference: InferenceSafetyConfig = field(default_factory=InferenceSafetyConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    evaluation: EvalSafetyConfig = field(default_factory=EvalSafetyConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedSafetyConfig":
        return cls(
            training=TrainingSafetyConfig(**data.get("training", {})),
            inference=InferenceSafetyConfig(**data.get("inference", {})),
            guardrails=GuardrailConfig(**data.get("guardrails", {})),
            evaluation=EvalSafetyConfig(**data.get("evaluation", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "UnifiedSafetyConfig":
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config loading.")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json_file(cls, path: str) -> "UnifiedSafetyConfig":
        """Load config from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config saving.")
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info("Saved safety config to %s", path)

    def save_json(self, path: str) -> None:
        """Save config to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved safety config to %s", path)

    def get_enabled_modules(self) -> Dict[str, List[str]]:
        """List all enabled modules across all categories."""
        enabled: Dict[str, List[str]] = {"training": [], "inference": [], "guardrails": [], "evaluation": []}
        for name, cfg in asdict(self.training).items():
            if isinstance(cfg, dict) and cfg.get("enabled"):
                enabled["training"].append(name)
        for name, cfg in asdict(self.inference).items():
            if isinstance(cfg, dict) and cfg.get("enabled"):
                enabled["inference"].append(name)
        for name, cfg in asdict(self.guardrails).items():
            if isinstance(cfg, dict) and cfg.get("enabled"):
                enabled["guardrails"].append(name)
        for name, cfg in asdict(self.evaluation).items():
            if isinstance(cfg, dict) and cfg.get("enabled"):
                enabled["evaluation"].append(name)
        return enabled

    def summary(self) -> str:
        """Human-readable summary of the configuration."""
        enabled = self.get_enabled_modules()
        lines = ["Safety Configuration Summary:"]
        for category, modules in enabled.items():
            if modules:
                lines.append(f"  {category}: {', '.join(modules)}")
            else:
                lines.append(f"  {category}: (none enabled)")
        return "\n".join(lines)
