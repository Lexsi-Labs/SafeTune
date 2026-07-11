"""
SafeTune ``core`` — shared building blocks, internal infrastructure.

Holds the cross-pillar machinery reused across interventions and instrumentation:
parameter-efficient ``SafetyLoRA``, safety-neuron fine-tuning, the weight-space
``patches`` family, ``CircuitInfo`` / CircuitKIT targeting,
``neuron_safety`` utilities, and ``UnifiedSafetyConfig``.

Also contains internal infrastructure packages moved under ``core/`` for a
cleaner top-level namespace: ``eval``, ``runtime``, ``extras``, ``safety``,
``training``, and ``scripts``.
"""

from .safety_lora import (
    SafetyLoRAConfig,
    build_safety_lora_config,
    merge_safety_lora_into_model_config,
)
from .neuron_ft import (
    NeuronUnitScore,
    SafetyNeuronIntervention,
    SafetyNeuronFTConfig,
    build_safety_neuronft_config,
    discover_safety_neurons,
    validate_neuron_causality,
    build_intervention,
)
from .artifacts import (
    SafetyArtifactBundle,
    SafetyArtifactManager,
)
from .data_compiler import (
    ModelAdapterDescriptor,
    CompiledSafetyData,
    CompilerConfig,
    SafetyPack,
    build_adapter_descriptor,
    build_compiler_config,
    compile_utility_to_safety,
    resolve_pack,
    resolve_packs,
)
from .patches import (
    SafetyPatch,
    PatchState,
    AntidotePatch,
    MSCPProjectionPatch,
    NLSRPatch,
    SafeLoRAPatch,
    create_patch,
)
from .circuit_kit import (
    CircuitInfo,
    LayerModuleSuggestions,
    SafetyRelevantUnits,
    get_circuit_info,
    load_circuit_info_from_file,
)
from .neuron_safety import (
    NeuronSafetyResult,
    identify_safety_units,
    get_lora_targeting_from_circuit,
    validate_impact,
    apply_circuit_to_safety_lora,
)
from .safety_config import (
    UnifiedSafetyConfig,
    TrainingSafetyConfig,
    InferenceSafetyConfig,
    GuardrailConfig,
    EvalSafetyConfig,
)
from . import eval as _eval  # noqa: F811  (low-level evaluation infrastructure)
from . import extras          # steering vectors, safety subspace, unified config
from . import runtime         # inference runtime, guardrails, CoSAlign
from . import safety          # multi-turn safety
from . import training        # training orchestration (internal)
from . import scripts         # CLI utilities

# interpret was moved to safetune.interpret (real package); core.interpret is
# a backward-compat shim kept for existing imports.

__all__ = [
    "SafetyLoRAConfig",
    "SafetyNeuronFTConfig",
    "NeuronUnitScore",
    "SafetyNeuronIntervention",
    "build_safety_lora_config",
    "build_safety_neuronft_config",
    "discover_safety_neurons",
    "validate_neuron_causality",
    "build_intervention",
    "merge_safety_lora_into_model_config",
    "SafetyArtifactBundle",
    "SafetyArtifactManager",
    "ModelAdapterDescriptor",
    "CompiledSafetyData",
    "CompilerConfig",
    "SafetyPack",
    "build_adapter_descriptor",
    "build_compiler_config",
    "compile_utility_to_safety",
    "resolve_pack",
    "resolve_packs",
    "SafetyPatch",
    "PatchState",
    "AntidotePatch",
    "MSCPProjectionPatch",
    "NLSRPatch",
    "SafeLoRAPatch",
    "create_patch",
    "CircuitInfo",
    "LayerModuleSuggestions",
    "SafetyRelevantUnits",
    "get_circuit_info",
    "load_circuit_info_from_file",
    # neuron safety
    "NeuronSafetyResult",
    "identify_safety_units",
    "get_lora_targeting_from_circuit",
    "validate_impact",
    "apply_circuit_to_safety_lora",
    # safety config
    "UnifiedSafetyConfig",
    "TrainingSafetyConfig",
    "InferenceSafetyConfig",
    "GuardrailConfig",
    "EvalSafetyConfig",
    # sub-packages
    "eval",
    "extras",
    "runtime",
    "safety",
    "training",
    "scripts",
]
