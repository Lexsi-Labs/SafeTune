"""Post-finetune safety patching interfaces."""

from .base import PatchState, PatchVerificationResult, SafetyPatch
from .antidote import AntidotePatch
from .mscp_projection import MSCPProjectionPatch
from .nlsr_patch import NLSRPatch
from .safe_lora_patch import SafeLoRAPatch

PATCH_REGISTRY = {
    "antidote": AntidotePatch,
    "mscp_projection": MSCPProjectionPatch,
    "nlsr": NLSRPatch,
    "safe_lora": SafeLoRAPatch,
}


def create_patch(patch_id: str, **params):
    patch_cls = PATCH_REGISTRY.get(patch_id)
    if patch_cls is None:
        raise KeyError(f"Unknown patch_id: {patch_id}")
    return patch_cls(**params)


__all__ = [
    "PatchState",
    "PatchVerificationResult",
    "SafetyPatch",
    "AntidotePatch",
    "MSCPProjectionPatch",
    "NLSRPatch",
    "SafeLoRAPatch",
    "PATCH_REGISTRY",
    "create_patch",
]
