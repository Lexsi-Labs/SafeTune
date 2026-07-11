"""QReSafe: quantization-aware safety patching for quantized LLMs.

Paper
-----
"Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for
Quantized Large Language Models", Kejia Chen, Jiawen Zhang, Jiacong Hu, Yu Wang,
Jian Lou, Zunlei Feng, Mingli Song. ICML 2025 (PMLR 267). arXiv:2506.20251.
Repo: https://github.com/Thecommonirin/Qresafe

Faithful algorithm (paper Section 4.2 + Algorithm 1)
----------------------------------------------------
Q-resafe re-aligns a *quantized* LLM ``pi_Q0`` with its *pre-quantization*
counterpart ``pi_W`` by twisting only the safety-critical weights.

* **Mode 2 — lora_dpo (the paper's core algorithm).**

  1. *Safety-patching dataset.* For each calibration prompt ``x``, sample
     ``y_w ~ pi_W`` (pre-quantization model -> *winner* / preferred) and
     ``y_l ~ pi_Q0`` (quantized model -> *loser* / dispreferred). The triplet
     ``(x, y_w, y_l)`` is the preference sample; no manual annotation needed.

  2. *Periodic safety-critical weight identification.* Safety-critical weights
     are scored with the **SNIP score** (Lee et al. 2019), NOT activation
     contrast:  for loss ``L(x) = -log p(y|x)`` (conditional NLL),

         I(W_ij, x) = | W_ij * dL(x)/dW_ij |
         SafeScore(Q) = E_{x in D_calib} I(Q_ij, x)

     Weights in the top-``tau`` percentile of ``SafeScore`` are safety-critical;
     ``M_Q`` has 1's there. The subset is *re-identified every K iterations*
     from the current weights ``Q_t``.

  3. *Masked-LoRA DPO update.* The conceptual objective is the DPO loss

         L = -E log sigma( beta*log(pi_Q(y_w|x)/pi_Q0(y_w|x))
                          - beta*log(pi_Q(y_l|x)/pi_Q0(y_l|x)) )

     subject to ``Q = Q0 + Quant(M_Q (.) AB)``. ``pi_Q0`` is the (frozen)
     reference. LoRA matrices A, B are updated with a *masked* SGD step:

         A_{t+1} = M_A (.) (A_t - eta*grad_A L) + (1 - M_A) (.) A_t

     so only safety-critical positions move; other LoRA entries stay intact.

* **Mode 1 — selective (quant-without-ft).** No DPO. Identify safety-critical
  weights with the SNIP score on the *full-precision* pre-quantization model
  and keep them at FP16/FP32 while the rest are quantized to ``quant_bits``.

Implementation notes
--------------------
The paper's defining mechanisms (SNIP-score identification, the
``y_w ~ pi_W`` / ``y_l ~ pi_Q0`` preference construction, the real DPO loss
with ``pi_Q0`` as reference, and the masked-LoRA update) are implemented here
directly. They require two extra inputs the legacy ``core`` patch never had:
the pre-quantization reference model and real calibration prompts. These are
threaded through as *optional* keyword arguments (``pre_quant_model``,
``calib_inputs``) so the public signature stays backward compatible: when they
are absent the call degrades to the legacy ``core.patches.qresafe_patch``
skeleton with a logged warning, exactly as before.

In ``lora_dpo`` mode the backbone is frozen during the DPO updates (only the
LoRA factors train), so at each periodic re-identification the backbone is
*temporarily unfrozen* for the SNIP forward/backward only: the score is
computed over the current backbone weights ``Q_t`` (the paper's top-``tau``
safety-critical *weight* identification), the resulting per-weight mask is
projected onto the LoRA factors (a row/column of A/B is active iff the
corresponding input/output feature contains at least one safety-critical
weight), and the backbone is re-frozen with its scratch gradients discarded
before the next masked-LoRA step.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SNIP-score safety-critical weight identification (paper Eq. 3)
# ---------------------------------------------------------------------------

def _conditional_nll(model: Any, x: Any) -> Any:
    """Conditional negative log-likelihood ``L(x) = -log p(y|x)``.

    With teacher-forcing on a token sequence the per-sequence NLL is the usual
    next-token cross-entropy; the paper takes exactly this loss when scoring
    weights.  ``x`` is a tensor of input ids (or a dict of model kwargs).
    """
    import torch.nn.functional as F

    if isinstance(x, dict):
        out = model(**x)
        ids = x.get("input_ids")
    else:
        out = model(x)
        ids = x
    logits = out.logits if hasattr(out, "logits") else out[0]
    # Next-token prediction: predict ids[:, 1:] from logits[:, :-1].
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = ids[:, 1:].reshape(-1).long()
    return F.cross_entropy(shift_logits, shift_labels)


def _snip_safe_scores(
    model: Any,
    calib_inputs: List[Any],
) -> Dict[str, Any]:
    """Per-weight SNIP score ``SafeScore = E_x | W (.) dL(x)/dW |``.

    Returns a dict mapping parameter name -> elementwise score tensor. The
    score is accumulated (averaged) over every calibration prompt, matching the
    paper's ``E_{x in D_calib}`` expectation.
    """

    scores: Dict[str, Any] = {}
    n = 0
    for x in calib_inputs:
        model.zero_grad(set_to_none=True)
        loss = _conditional_nll(model, x)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            # SNIP: |W * grad| -- weight-magnitude-times-gradient salience.
            contrib = (p.detach() * p.grad.detach()).abs()
            if name in scores:
                scores[name] = scores[name] + contrib
            else:
                scores[name] = contrib.clone()
        n += 1
    model.zero_grad(set_to_none=True)
    if n > 1:
        for name in scores:
            scores[name] = scores[name] / n
    return scores


def _critical_mask_from_scores(
    scores: Dict[str, Any],
    tau: float,
) -> Dict[str, Any]:
    """Top-``tau`` percentile mask ``M`` (1 = safety-critical) per parameter.

    ``tau`` is the fraction of weights kept as safety-critical (paper default
    0.6 -> top 60%).  ``tau=1`` updates every weight (no identification);
    ``tau=0`` updates nothing.
    """
    import torch

    masks: Dict[str, Any] = {}
    tau = float(min(1.0, max(0.0, tau)))
    for name, s in scores.items():
        flat = s.reshape(-1)
        if tau >= 1.0:
            masks[name] = torch.ones_like(s)
            continue
        if tau <= 0.0:
            masks[name] = torch.zeros_like(s)
            continue
        # keep the top-tau fraction by score -> threshold at the (1-tau) quantile
        k = max(1, int(flat.numel() * tau))
        thresh = torch.topk(flat, k, largest=True).values.min()
        masks[name] = (s >= thresh).to(s.dtype)
    return masks


# ---------------------------------------------------------------------------
# DPO objective (paper Eq. 1)
# ---------------------------------------------------------------------------

def _sequence_logprob(model: Any, x: Any) -> Any:
    """Sum of teacher-forced token log-probs ``log pi(y|x)`` for sequence ``x``."""
    import torch.nn.functional as F

    if isinstance(x, dict):
        out = model(**x)
        ids = x.get("input_ids")
    else:
        out = model(x)
        ids = x
    logits = out.logits if hasattr(out, "logits") else out[0]
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:].long().unsqueeze(-1)
    tok_logp = logp.gather(-1, tgt).squeeze(-1)
    return tok_logp.sum(dim=-1)


def _dpo_loss(
    policy: Any,
    reference: Any,
    chosen: Any,
    rejected: Any,
    beta: float,
) -> Any:
    """QReSafe DPO loss (paper Eq. 1).

    ``L = -log sigma( beta*(logp_policy(y_w) - logp_ref(y_w))
                     - beta*(logp_policy(y_l) - logp_ref(y_l)) )``
    where the reference is the frozen quantized model ``pi_Q0``.
    """
    import torch
    import torch.nn.functional as F

    pol_w = _sequence_logprob(policy, chosen)
    pol_l = _sequence_logprob(policy, rejected)
    with torch.no_grad():
        ref_w = _sequence_logprob(reference, chosen)
        ref_l = _sequence_logprob(reference, rejected)
    chosen_rewards = beta * (pol_w - ref_w)
    rejected_rewards = beta * (pol_l - ref_l)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


# ---------------------------------------------------------------------------
# Faithful Mode 2: masked-LoRA DPO patching loop (Algorithm 1)
# ---------------------------------------------------------------------------

def _greedy_continue(model: Any, prompt_ids: Any, gen_len: int) -> Any:
    """Greedy-decode ``gen_len`` tokens and append them to ``prompt_ids``.

    Implements the ``y ~ pi(.|x)`` sampling of Algorithm 1 lines 2-3 with
    deterministic argmax decoding (no external generate() dependency, so it
    works for bare ``nn.Module`` stubs as well as HF ``*ForCausalLM`` models).
    """
    import torch

    seq = prompt_ids
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, gen_len)):
            out = model(seq) if not isinstance(seq, dict) else model(**seq)
            logits = out.logits if hasattr(out, "logits") else out[0]
            nxt = logits[:, -1:].argmax(dim=-1)
            seq = torch.cat([seq, nxt], dim=1)
    return seq


def _build_patch_dataset(
    quant_model: Any,
    pre_quant_model: Any,
    calib_inputs: List[Any],
    gen_len: int = 4,
) -> List[Dict[str, Any]]:
    """Construct the safety-patching dataset (Algorithm 1, lines 1-5).

    For each calibration prompt ``x`` the *winner* ``y_w ~ pi_W`` is decoded
    from the pre-quantization model and the *loser* ``y_l ~ pi_Q0`` from the
    quantized model. The chosen/rejected sequences are ``[x ; y_w]`` and
    ``[x ; y_l]`` -- distinct continuations, so the DPO preference signal is
    non-degenerate. Dict-style model inputs are skipped for generation and the
    prompt is reused (their continuations need tokenizer state).
    """
    triplets: List[Dict[str, Any]] = []
    for x in calib_inputs:
        if isinstance(x, dict):
            # cannot greedily continue a kwargs dict without a tokenizer
            triplets.append({"x": x, "chosen": x, "rejected": x})
            continue
        chosen = _greedy_continue(pre_quant_model, x, gen_len)
        rejected = _greedy_continue(quant_model, x, gen_len)
        triplets.append({"x": x, "chosen": chosen, "rejected": rejected})
    logger.info("QReSafe: built safety-patching dataset of %d triplet(s).", len(triplets))
    return triplets


def _faithful_lora_dpo(
    model: nn.Module,
    pre_quant_model: nn.Module,
    calib_inputs: List[Any],
    *,
    quant_bits: int,
    lora_rank: int,
    lora_alpha: float,
    dpo_epochs: int,
    reidentify_interval: int,
    lr: float,
    tau: float,
    beta: float,
) -> nn.Module:
    """Mode 2 -- masked-LoRA DPO safety patching (paper Algorithm 1)."""
    import copy

    import torch
    import torch.nn as _nn

    # pi_Q0: frozen quantized reference model for the DPO loss.
    reference = copy.deepcopy(model)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    patch_data = _build_patch_dataset(model, pre_quant_model, calib_inputs)

    # Attach LoRA matrices A (din x r) and B (r x dout) to every nn.Linear,
    # with the conventional zero-init on B so the initial delta is zero.
    lora_state: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, _nn.Linear):
            continue
        out_feat, in_feat = module.weight.shape
        r = min(lora_rank, max(1, min(out_feat, in_feat)))
        A = _nn.Parameter(torch.randn(r, in_feat, device=module.weight.device) * 0.01)
        B = _nn.Parameter(torch.zeros(out_feat, r, device=module.weight.device))
        scaling = lora_alpha / r
        orig_forward = module.forward

        def _make_forward(m: _nn.Linear, A_: Any, B_: Any, sc: float):
            def _fwd(inp: Any) -> Any:
                base = _nn.functional.linear(inp, m.weight, m.bias)
                delta = _nn.functional.linear(_nn.functional.linear(inp, A_), B_) * sc
                return base + delta
            return _fwd

        module.forward = _make_forward(module, A, B, scaling)  # type: ignore[assignment]
        lora_state.append({"name": name, "A": A, "B": B})

    if not lora_state:
        logger.warning("QReSafe: no nn.Linear modules found; nothing to patch.")
        return model

    # Freeze the quantized backbone; only LoRA matrices receive gradients.
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params = [d["A"] for d in lora_state] + [d["B"] for d in lora_state]
    for p in lora_params:
        p.requires_grad_(True)

    optimizer = torch.optim.SGD(lora_params, lr=lr)
    masks: Dict[str, Any] = {}
    step = 0
    model.train()

    for _epoch in range(max(1, dpo_epochs)):
        for triplet in patch_data:
            # --- periodic safety-critical weight identification (line 7-10) ---
            if step % max(1, reidentify_interval) == 0:
                # The backbone is frozen for the DPO updates, so its params
                # would receive no gradients and `_snip_safe_scores` would
                # return {} (-> all-ones masks, `tau` silently ignored).
                # Temporarily unfreeze it for the identification
                # forward/backward ONLY, so the SNIP score is computed over
                # the current backbone weights Q_t exactly as the paper
                # specifies, then re-freeze and drop the scratch gradients.
                backbone_params = list(model.parameters())
                for p in backbone_params:
                    p.requires_grad_(True)
                try:
                    scores = _snip_safe_scores(model, [triplet["x"]])
                finally:
                    for p in backbone_params:
                        p.requires_grad_(False)
                        p.grad = None
                    # The SNIP backward also deposits grads on the (closure)
                    # LoRA params; clear them so they don't leak into the
                    # next optimizer step.
                    for p in lora_params:
                        p.grad = None
                weight_masks = _critical_mask_from_scores(scores, tau)
                # MapMask: project the per-weight mask onto the LoRA factors.
                # A row of M_A is active iff that input feature is safety-
                # critical for any output; a row of M_B iff that output is.
                masks = {}
                for d in lora_state:
                    wname = f"{d['name']}.weight"
                    wm = weight_masks.get(wname)
                    if wm is None:
                        masks[d["name"]] = (torch.ones_like(d["A"]),
                                            torch.ones_like(d["B"]))
                        continue
                    # wm: (out_feat, in_feat). Reduce to per-feature masks.
                    in_active = (wm.sum(dim=0) > 0).to(d["A"].dtype)   # (in_feat,)
                    out_active = (wm.sum(dim=1) > 0).to(d["B"].dtype)  # (out_feat,)
                    M_A = in_active.unsqueeze(0).expand_as(d["A"]).contiguous()
                    M_B = out_active.unsqueeze(1).expand_as(d["B"]).contiguous()
                    masks[d["name"]] = (M_A, M_B)
                logger.debug("QReSafe: safety-critical mask refreshed at step %d.", step)

            # --- masked-LoRA DPO step (lines 11-13) -----------------------
            optimizer.zero_grad(set_to_none=True)
            try:
                loss = _dpo_loss(model, reference, triplet["chosen"],
                                 triplet["rejected"], beta)
                loss.backward()
                # Restrict the update to safety-critical positions:
                # A_{t+1} = M_A (.) (A - eta*grad) + (1 - M_A) (.) A.
                # Zeroing the gradient at non-critical entries before the SGD
                # step is exactly equivalent for plain SGD.
                with torch.no_grad():
                    for d in lora_state:
                        M_A, M_B = masks.get(
                            d["name"],
                            (torch.ones_like(d["A"]), torch.ones_like(d["B"])),
                        )
                        if d["A"].grad is not None:
                            d["A"].grad.mul_(M_A)
                        if d["B"].grad is not None:
                            d["B"].grad.mul_(M_B)
                optimizer.step()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("QReSafe: DPO step %d failed -- %s", step, exc)
            step += 1

    # Merge the trained LoRA delta into the weights: Q = Q0 + Quant(M (.) AB).
    # Wrapped in try/finally so the monkey-patched forwards are always removed
    # even if the weight update raises (shape mismatch, OOM, etc.) — otherwise
    # the model is left with patched forwards and is unusable for normal inference.
    name_to_mod = dict(model.named_modules())
    try:
        with torch.no_grad():
            for d in lora_state:
                module = name_to_mod[d["name"]]
                scaling = lora_alpha / d["A"].shape[0]
                delta = (d["B"] @ d["A"]) * scaling
                # Apply the final safety-critical mask to the merged delta.
                M_A, M_B = masks.get(
                    d["name"], (torch.ones_like(d["A"]), torch.ones_like(d["B"]))
                )
                # Reconstruct a (out,in) mask from the factor masks.
                w_mask = (M_B[:, :1] * M_A[:1, :])
                w_mask = (w_mask != 0).to(delta.dtype) if w_mask.numel() else 1.0
                module.weight.data = module.weight.data + (delta * w_mask).to(
                    module.weight.dtype
                )
    finally:
        # Always restore the original (non-monkey-patched) forwards regardless
        # of whether the weight update above succeeded or raised.
        with torch.no_grad():
            for d in lora_state:
                module = name_to_mod.get(d["name"])
                if module is not None and isinstance(module, _nn.Linear):
                    try:
                        del module.forward  # type: ignore[attr-defined]
                    except AttributeError:
                        pass

    model.eval()
    logger.info(
        "QReSafe: masked-LoRA DPO complete (%d step(s), tau=%.2f, beta=%.2f).",
        step, tau, beta,
    )
    return model


# ---------------------------------------------------------------------------
# Faithful Mode 1: selective quantization (quant-without-ft)
# ---------------------------------------------------------------------------

def _faithful_selective(
    model: nn.Module,
    calib_inputs: List[Any],
    *,
    quant_bits: int,
    tau: float,
) -> nn.Module:
    """Mode 1 -- keep SNIP-identified safety-critical weights at full precision,
    quantize the rest to ``quant_bits`` (paper, quant-without-ft).

    The fake-quantization round-trip below performs symmetric per-tensor
    integer quantization of the non-critical weights so the safety-critical
    weights are demonstrably the only ones left at full precision -- this is
    the actual ``Q0`` of the paper, not a no-op marker.
    """
    import torch

    scores = _snip_safe_scores(model, calib_inputs)
    masks = _critical_mask_from_scores(scores, tau)

    n_levels = max(2, 2 ** int(quant_bits))
    qmax = (n_levels // 2) - 1

    pinned = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            m = masks.get(name)
            if m is None:
                continue
            crit = m.to(torch.bool)
            non_crit = ~crit
            if non_crit.any():
                vals = p.data[non_crit]
                scale = vals.abs().max() / qmax if vals.numel() else None
                if scale is not None and scale > 0:
                    q = torch.clamp(torch.round(vals / scale), -qmax, qmax)
                    p.data[non_crit] = (q * scale).to(p.dtype)
            # safety-critical entries are left untouched (full precision)
            pinned += int(crit.sum().item())

    logger.info(
        "QReSafe: selective quantization done -- %d safety-critical weight(s) "
        "kept at full precision, rest quantized to %d-bit.", pinned, quant_bits
    )
    return model


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

@assert_mutates("apply_qresafe")
def apply_qresafe(
    model: nn.Module,
    mode: str = "selective",
    quant_bits: int = 4,
    safety_dataset: Optional[List[Dict[str, Any]]] = None,
    top_k_safety_weights: int = 128,
    lora_rank: int = 32,
    lora_alpha: float = 16.0,
    dpo_epochs: int = 1,
    reidentify_interval: int = 50,
    lr: float = 1e-4,
    *,
    pre_quant_model: Optional[nn.Module] = None,
    calib_inputs: Optional[List[Any]] = None,
    tau: float = 0.6,
    dpo_beta: float = 0.1,
    **extra: Any,
) -> nn.Module:
    """Apply QReSafe quantization-aware safety patching to ``model``.

    ``mode="selective"`` (quant-without-ft) identifies safety-critical weights
    with the SNIP score and pins them at full precision while quantizing the
    rest. ``mode="lora_dpo"`` runs the paper's masked-LoRA DPO loop: it
    contrasts the pre-quantization model (winner) against the quantized model
    (loser) and updates only safety-critical LoRA positions, periodically
    re-identifying the safety-critical mask.

    Faithful (paper Algorithm 1) inputs -- optional, backward compatible:

    * ``pre_quant_model``: the pre-quantization LLM ``pi_W`` (required for the
      faithful ``lora_dpo`` path; it provides the preferred responses and is
      the safety teacher).
    * ``calib_inputs``: a list of real calibration prompt tensors / model-kwarg
      dicts. Used both for the SNIP-score weight identification and as the
      DPO calibration set. When omitted the SNIP/DPO path cannot run.
    * ``tau``: fraction of weights kept as safety-critical (paper default 0.6).
    * ``dpo_beta``: DPO temperature ``beta`` (paper Eq. 1).

    When the faithful inputs are unavailable the call falls back to the legacy
    ``core.patches.qresafe_patch`` skeleton (random-probe identification, CE
    stand-in) with a logged warning, preserving prior behaviour.
    """
    if mode not in ("selective", "lora_dpo"):
        raise ValueError(f"apply_qresafe: unknown mode {mode!r}")

    # Resolve calibration inputs: explicit `calib_inputs` wins; otherwise try
    # to lift tokenised tensors out of `safety_dataset` rows.
    resolved_calib: Optional[List[Any]] = calib_inputs
    if resolved_calib is None and safety_dataset:
        lifted = [
            r["input_ids"] for r in safety_dataset
            if isinstance(r, dict) and "input_ids" in r
        ]
        resolved_calib = lifted or None

    # ---- Faithful path -----------------------------------------------------
    if resolved_calib:
        try:
            if mode == "lora_dpo":
                if pre_quant_model is None:
                    raise ValueError(
                        "faithful lora_dpo needs `pre_quant_model` (pi_W)"
                    )
                return _faithful_lora_dpo(
                    model, pre_quant_model, resolved_calib,
                    quant_bits=quant_bits,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    dpo_epochs=dpo_epochs,
                    reidentify_interval=reidentify_interval,
                    lr=lr,
                    tau=tau,
                    beta=dpo_beta,
                )
            return _faithful_selective(
                model, resolved_calib, quant_bits=quant_bits, tau=tau
            )
        except Exception as exc:
            logger.warning(
                "QReSafe: faithful path failed (%s); falling back to legacy "
                "core patch skeleton.", exc
            )

    # ---- Legacy fallback (no real calibration data available) --------------
    logger.warning(
        "QReSafe: running legacy core-patch skeleton -- pass `calib_inputs` "
        "(and `pre_quant_model` for lora_dpo) for the faithful Algorithm 1 path."
    )
    try:
        from safetune.core.patches.qresafe_patch import (
            QReSafeLoRAPatch,
            QReSafeSelectivePatch,
        )
    except ImportError as e:  # pragma: no cover - defensive
        raise ImportError(
            f"apply_qresafe needs safetune.core.patches.qresafe_patch: {e}"
        ) from e

    params: Dict[str, Any] = {
        "quant_bits": quant_bits,
        "safety_dataset": safety_dataset,
        "top_k_safety_weights": top_k_safety_weights,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "dpo_epochs": dpo_epochs,
        "reidentify_interval": reidentify_interval,
        "lr": lr,
    }
    params.update(extra)

    if mode == "lora_dpo":
        patch = QReSafeLoRAPatch(**params)
    else:
        patch = QReSafeSelectivePatch(**params)

    patch.apply_to_model(model)
    return model


__all__ = ["apply_qresafe"]
