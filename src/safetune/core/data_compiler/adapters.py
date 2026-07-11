"""Model adapter descriptors for compiler inputs."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelAdapterDescriptor:
    provider: str  # hf | vllm | api
    model_id: str
    chat_template: str = "auto"
    metadata: Dict[str, str] = field(default_factory=dict)


def build_adapter_descriptor(payload: Dict[str, str]) -> ModelAdapterDescriptor:
    return ModelAdapterDescriptor(
        provider=payload.get("provider", "hf"),
        model_id=payload.get("model_id", ""),
        chat_template=payload.get("chat_template", "auto"),
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
    )
