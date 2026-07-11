"""Algo registry — single source of truth for every named method.

The CLI and any programmatic dispatch both read from these dicts, so adding a
new method only requires editing this file (plus the implementation module).

Third-party code can extend the registries at runtime::

    from safetune.runner._registry import register_harden
    register_harden("mymethod", "MyTrainer")   # MyTrainer must be importable
                                                # from safetune.runner.harden
"""
from __future__ import annotations

# Maps CLI alias → Trainer class name inside safetune.runner.<pillar>

HARDEN_REGISTRY: dict[str, str] = {
    # gradient surgery
    "safegrad":       "SafeGradTrainer",
    "plainsft":       "PlainSFTTrainer",
    # data shaping
    "lisa":           "LisaTrainer",
    "sppft":          "SPPFTTrainer",
    "lookahead":      "LookAheadTrainer",
    "stardss":        "STARDSSTrainer",
    "derta":          "DeRTaTrainer",
    # regularization
    "asft":           "AsFTTrainer",
    "sap":            "SAPTrainer",
    "surgery":        "SurgeryTrainer",
    "booster":        "BoosterTrainer",
    # representation perturbation
    "vaccine":        "VaccineTrainer",
    "tvaccine":       "TVaccineTrainer",
    # tamper-resistant
    "repnoise":       "RepNoiseTrainer",
    "ctrap":          "CTRAPTrainer",
    "seam":           "SEAMTrainer",
    "door":           "DOORTrainer",
    # other
    "tar":            "TARTrainer",
    "salora":         "SaLoRATrainer",
    "seal":           "SEALTrainer",
    "constrained":    "ConstrainedSFTTrainer",
    "loxharden":      "LoXHardenTrainer",
}

RECOVER_REGISTRY: dict[str, str] = {
    # whole-model
    "task-arithmetic":  "TaskArithmeticTrainer",
    "prepost":          "PrePostMergeTrainer",
    "somf":             "SOMFTrainer",
    "wiseft":           "WiseFTTrainer",
    # layer
    "resta":            "ReStaTrainer",
    "safemerge":        "SafeMergeTrainer",
    "safedelta":        "SafeDeltaTrainer",
    "safelora":         "SafeLoRATrainer",
    "qresafe":          "QReSafeTrainer",
    "aaq":              "AAQTrainer",
    "repnoise-recover": "RepNoiseRecoverTrainer",
    # low-rank
    "lox":              "LoXTrainer",
    "lssf":             "LSSFTrainer",
    "safety-vector":    "SafetyVectorRestoreTrainer",
    # neuron
    "nlsr":             "NLSRTrainer",
    "antidote":         "AntidoteTrainer",
    "antidote-v2":      "AntidoteV2Trainer",
    "mscp":             "MSCPTrainer",
    # saliency
    "grad-selective":   "GradSelectiveRecoverTrainer",
    "oneshot-patch":    "OneShotSafetyPatchTrainer",
    # other
    "pke":              "PKETrainer",
    "safereact":        "SafeReActTrainer",
    "scrub":            "SCRUBTrainer",
    # circuit — operates on safety circuits rather than full weight matrices;
    # requires base_model + aligned_model to extract the safety delta.
    "ctheta":           "CThetaTrainer",
}

UNLEARN_REGISTRY: dict[str, str] = {
    "rmu":      "RMUTrainer",
    "npo":      "NPOTrainer",
    "ga":       "GradientAscentTrainer",
    "graddiff": "GradDiffTrainer",
    "flat":     "FLATTrainer",
    "simdpo":   "SimDPOTrainer",
}


def register_harden(alias: str, trainer_class_name: str) -> None:
    """Register a custom harden trainer under the given CLI alias."""
    HARDEN_REGISTRY[alias.lower()] = trainer_class_name


def register_recover(alias: str, trainer_class_name: str) -> None:
    """Register a custom recover trainer under the given CLI alias."""
    RECOVER_REGISTRY[alias.lower()] = trainer_class_name


def register_unlearn(alias: str, trainer_class_name: str) -> None:
    """Register a custom unlearn trainer under the given CLI alias."""
    UNLEARN_REGISTRY[alias.lower()] = trainer_class_name
