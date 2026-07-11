"""Training-free post-hoc safety patches (apply_* functional API)."""
from .resta import apply_resta
from .lox import apply_lox
from .safe_delta import apply_safe_delta
from .safe_lora import apply_safe_lora
from .antidote import apply_antidote
from .mscp import apply_mscp
from .nlsr import apply_nlsr
from .pke import apply_pke
from .safereact import apply_safereact
from .qresafe import apply_qresafe
from .aaq import apply_aaq
from .merge import task_arithmetic, somf_merge, learn_somf_mask, apply_prepost_merge
from .lssf import apply_lssf
from .deeprefusal import apply_deeprefusal
from .scrub import SCRUBConfig, scrub_unlearn, tracin_influence
from .ctheta import apply_ctheta, apply_ctheta_from_state_dicts, sweep_ctheta_strength
from .grad_selective_recover import apply_grad_selective_recover
from .wise_ft import apply_wise_ft
from .safety_vector_restore import apply_safety_vector_restore
from .oneshot_safety_patch import apply_oneshot_safety_patch
from .antidote_v2 import apply_antidote_v2
from .repnoise_recover import apply_repnoise_recover

try:
    from .safemerge import apply_safemerge
except Exception:  # pragma: no cover
    apply_safemerge = None  # type: ignore[assignment]

# ── Uniform model-input contract ────────────────────────────────────────────
# Every RECOVER method edits one *target* model (the finished / drifted
# checkpoint). The canonical keyword for it is ``target=``; ``model=`` /
# ``finetuned=`` keep working as back-compat aliases. See ``_contract.py``.
from ._contract import accept_target_alias as _accept_target_alias

for _name in (
    "apply_resta", "apply_lox", "apply_safe_delta", "apply_safe_lora",
    "apply_antidote", "apply_mscp", "apply_nlsr", "apply_pke",
    "apply_safereact", "apply_qresafe", "apply_aaq", "task_arithmetic",
    "somf_merge", "apply_safemerge", "apply_lssf", "apply_deeprefusal",
    "apply_ctheta", "apply_prepost_merge",
    "apply_grad_selective_recover", "apply_wise_ft",
    "apply_safety_vector_restore", "apply_oneshot_safety_patch",
    "apply_antidote_v2", "apply_repnoise_recover",
):
    _fn = globals().get(_name)
    if callable(_fn):
        globals()[_name] = _accept_target_alias(_fn)
del _name, _fn

__all__ = [
    "apply_resta",
    "apply_lox",
    "apply_safe_delta",
    "apply_safe_lora",
    "apply_antidote",
    "apply_mscp",
    "apply_nlsr",
    "apply_pke",
    "apply_safereact",
    "apply_qresafe",
    "apply_aaq",
    "task_arithmetic",
    "somf_merge",
    "learn_somf_mask",
    "apply_safemerge",
    "apply_lssf",
    "apply_deeprefusal",
    "SCRUBConfig",
    "scrub_unlearn",
    "tracin_influence",
    "apply_ctheta",
    "apply_ctheta_from_state_dicts",
    "sweep_ctheta_strength",
    "apply_prepost_merge",
    "apply_grad_selective_recover",
    "apply_wise_ft",
    "apply_safety_vector_restore",
    "apply_oneshot_safety_patch",
    "apply_antidote_v2",
    "apply_repnoise_recover",
]
