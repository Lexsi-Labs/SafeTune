"""
Safety-neuron localization.

Definition: a unit at position ``(layer, neuron_idx)`` is "safety-relevant"
if it either (a) has an output direction (a row/column of the layer's output
projection) with high absolute cosine against a known refusal direction at
that layer, or (b) activates *differentially* on harmful vs. harmless prompts.

Two paths supported:

* ``"weight"`` (default, no extra forward passes): rank each unit by
  ``|col(W_out) . refusal_dir|`` for the layer's output projection
  (``mlp.down_proj`` by default). Each column of ``down_proj`` is one MLP
  intermediate neuron's write-direction into the residual stream; the refusal
  direction lives in that same residual-stream space, so the cosine is a
  weight-space proxy for "this neuron writes refusal." Cheap and exact.

* ``"activation"``: run the model on a contrast corpus (harmful vs. harmless
  prompts), capture each layer's per-neuron MLP activations, and rank units by
  a per-neuron *activation contrast* between the two sets. This is the
  feed-forward activation-analysis localization used by the safety-neuron
  literature -- e.g. Wei et al., "Assessing the Brittleness of Safety
  Alignment via Pruning and Low-Rank Modifications" (ICML 2024,
  arXiv:2402.05162), whose Wanda-style importance score multiplies a weight
  by the L2 norm of the neuron's input activations; Chen et al., "Finding
  Safety Neurons in Large Language Models" (arXiv:2406.14144); and the
  feed-forward activation-analysis localization of Wu et al., "NeuroStrike"
  (arXiv:2509.11864). The contrast score implemented here -- mean activation
  magnitude on harmful minus harmless prompts, optionally standardized --
  measures harmful-input *selectivity* on a SINGLE model.

  ⚠️ FIDELITY: this is NOT the metric of the cited papers. Chen et al.
  (2406.14144) score a neuron by its RMS activation difference between two
  CHECKPOINTS (e.g. SFT vs DPO-aligned) on the *same* sequences; Wei et al.
  (2402.05162) use Wanda/SNIP weight-importance with a safety-vs-utility
  SET-DIFFERENCE. Our one-model harmful-vs-harmless contrast is a related but
  distinct localization heuristic; the papers are cited for context, not as the
  source of this exact formula.

Either path produces a :class:`SafetyNeuronReport` which converts cleanly
to a :class:`safetune.core.circuit_kit.CircuitInfo` so downstream Recover
methods (PKE, NLSR, LSSF, DeepRefusal) can target the located neurons.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..._refusal_helpers import _get_decoder_layers

logger = logging.getLogger(__name__)


@dataclass
class SafetyNeuronConfig:
    """Configuration for safety-neuron localization.

    Attributes:
        method: ``"weight"`` (output-direction cosine, no extra forwards) or
            ``"activation"`` (per-neuron harmful-vs-harmless activation contrast
            across a contrast corpus).
        top_k_per_layer: keep only the top-k highest-scoring units per layer.
        target_layers: restrict to these layers; ``None`` means all decoder layers.
        score_floor: drop units whose absolute score is below this threshold.
        target_module: which projection matrix to score in ``"weight"`` mode.
            ``"mlp.down_proj"`` (default): each *column* is one MLP intermediate
            neuron's residual-stream write-direction -- columns are the natural
            unit axis and the cosine against the refusal direction is exact.
            ``"self_attn.o_proj"``: columns index attention *output dimensions*
            (head_dim slices of the concatenated heads), not whole heads; the
            per-column cosine is still well defined but a "unit" is then an
            o_proj input channel, not an attention head.
        activation_module: in ``"activation"`` mode, which sub-module's output
            is captured as the per-neuron activation vector. ``"mlp.act_fn"``
            captures the post-activation MLP intermediate (gate) neurons --
            these are the canonical "neurons" of the safety-neuron literature.
            If that hook point is unavailable the implementation falls back to
            ``"mlp"`` (the MLP block output, i.e. residual-stream units).
        activation_score: how to turn captured activations into a per-neuron
            contrast. ``"mean_abs_diff"`` (default): mean |act| on harmful minus
            mean |act| on harmless. ``"tstat"``: a Welch-style standardized
            mean difference (mean-abs difference divided by pooled std) -- the
            label-correlation form, robust to per-neuron scale. ``"mean_diff"``:
            signed mean (not magnitude) difference.
        activation_batch_size: forward-pass batch size for the contrast corpus.
        activation_max_tokens: truncate each prompt to this many tokens.
        abs_rank: rank units by absolute score (``True``, default) so that
            strongly *suppressed*-on-harmful neurons are also surfaced.
    """

    method: str = "weight"
    top_k_per_layer: int = 16
    target_layers: Optional[List[int]] = None
    score_floor: float = 0.0
    target_module: str = "mlp.down_proj"
    activation_module: str = "mlp.act_fn"
    activation_score: str = "mean_abs_diff"
    activation_batch_size: int = 8
    activation_max_tokens: int = 64
    abs_rank: bool = True


@dataclass
class SafetyNeuronReport:
    """Per-layer ranked list of safety-relevant unit indices."""

    per_layer: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    direction_layer: Optional[int] = None
    method: str = ""
    target_module: str = ""

    def as_circuit_info(self):
        """Convert to a :class:`CircuitInfo` for downstream consumers."""
        from safetune.core.circuit_kit.interface import (
            CircuitInfo,
            LayerModuleSuggestions,
            SafetyRelevantUnits,
        )

        layer_indices = sorted(self.per_layer.keys())
        unit_ids = [
            f"L{li}.{self.target_module}.{idx}"
            for li in layer_indices
            for (idx, _) in self.per_layer[li]
        ]
        scores = {
            f"L{li}.{self.target_module}.{idx}": float(score)
            for li in layer_indices
            for (idx, score) in self.per_layer[li]
        }
        # The ``activation_correlation`` field is the schema slot for a
        # per-unit score map. In ``"weight"`` mode it carries refusal-direction
        # cosines, not activation correlations; ``score_kind`` in metadata
        # records which it is so consumers are not misled by the field name.
        score_kind = (
            "activation_contrast" if self.method == "activation" else "weight_cosine"
        )
        return CircuitInfo(
            safety_units=SafetyRelevantUnits(
                layer_indices=layer_indices,
                module_names=[self.target_module],
                unit_ids=unit_ids,
                activation_correlation=scores,
                metadata={
                    "method": self.method,
                    "direction_layer": self.direction_layer,
                    "score_kind": score_kind,
                },
            ),
            layer_suggestions=LayerModuleSuggestions(
                target_modules=[self.target_module],
                layer_subset=layer_indices,
                metadata={"source": "safetune.core.interpret"},
            ),
            raw_output_path=None,
            metadata={"method": self.method, "score_kind": score_kind},
        )


# ---------------------------------------------------------------------------
# Module resolution
# ---------------------------------------------------------------------------

def _resolve_module(layer: nn.Module, target: str) -> Optional[nn.Module]:
    """Resolve a dotted-path sub-module (e.g. ``"mlp.down_proj"``) in a layer."""
    cur: Any = layer
    for part in target.split("."):
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, nn.ModuleDict) and part in cur:  # type: ignore[operator]
            cur = cur[part]
        else:
            return None
    return cur


def _resolve_linear(layer: nn.Module, target: str) -> Optional[nn.Linear]:
    """Resolve a dotted path and require the result to be an ``nn.Linear``."""
    mod = _resolve_module(layer, target)
    return mod if isinstance(mod, nn.Linear) else None


# ---------------------------------------------------------------------------
# Weight-based scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_by_weight(
    model: nn.Module,
    refusal_direction_per_layer: Dict[int, torch.Tensor],
    cfg: SafetyNeuronConfig,
) -> Dict[int, List[Tuple[int, float]]]:
    layers = _get_decoder_layers(model)
    targets = (
        cfg.target_layers if cfg.target_layers is not None else list(range(len(layers)))
    )
    out: Dict[int, List[Tuple[int, float]]] = {}

    for li in targets:
        if not (0 <= li < len(layers)):
            continue
        proj = _resolve_linear(layers[li], cfg.target_module)
        if proj is None:
            logger.debug("interpret: layer %d missing %s; skipping.", li, cfg.target_module)
            continue
        # proj.weight has shape (hidden_out, hidden_in). Each COLUMN is one
        # input unit and its column is that unit's contribution direction in
        # the output (residual-stream) space. For ``mlp.down_proj`` a column is
        # exactly one MLP intermediate neuron's write-direction; for
        # ``self_attn.o_proj`` a column is one o_proj input channel. The
        # refusal direction lives in the output space, so the safety-relevance
        # score is the absolute cosine of column[j] with d.
        d = refusal_direction_per_layer.get(li)
        if d is None and refusal_direction_per_layer:
            d = next(iter(refusal_direction_per_layer.values()))
        if d is None:
            continue
        d_unit = d.detach().float()
        d_unit = d_unit / d_unit.norm().clamp_min(1e-12)
        cols = proj.weight.detach().float()
        if cols.shape[0] != d_unit.shape[0]:
            # Output dim does not match the refusal direction's hidden size;
            # something is wrong with the layer choice. Skip rather than crash.
            logger.debug(
                "interpret: layer %d module %s has out_dim=%d != hidden=%d; skipping.",
                li, cfg.target_module, cols.shape[0], d_unit.shape[0],
            )
            continue
        col_norms = cols.norm(dim=0).clamp_min(1e-12)
        cosines = (d_unit.to(cols.device) @ cols) / col_norms
        scored = (
            (j, abs(float(c))) if cfg.abs_rank else (j, float(c))
            for j, c in enumerate(cosines.tolist())
        )
        ranked = sorted(scored, key=lambda x: abs(x[1]), reverse=True)
        kept = [(j, s) for j, s in ranked if abs(s) > cfg.score_floor][: cfg.top_k_per_layer]
        if kept:
            out[li] = kept
    return out


# ---------------------------------------------------------------------------
# Activation-based scoring
# ---------------------------------------------------------------------------

def _resolve_activation_point(
    layer: nn.Module, cfg: SafetyNeuronConfig
) -> Tuple[Optional[nn.Module], str]:
    """Pick the sub-module whose output is the per-neuron activation vector.

    Tries ``cfg.activation_module`` first (``mlp.act_fn`` -> the post-activation
    MLP intermediate / gate neurons, the canonical "neurons"), then falls back
    through a small list of architecture-portable hook points so the routine
    works on Llama/Mistral/Qwen/Gemma (``mlp``) and GPT-style (``mlp`` /
    ``mlp.act``) models without the caller having to know the layout.
    """
    candidates = [cfg.activation_module, "mlp.act_fn", "mlp.act", "mlp"]
    seen: set = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        mod = _resolve_module(layer, name)
        if isinstance(mod, nn.Module):
            return mod, name
    return None, cfg.activation_module


@torch.no_grad()
def _score_by_activation(
    model: nn.Module,
    tokenizer: Any,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    cfg: SafetyNeuronConfig,
) -> Tuple[Dict[int, List[Tuple[int, float]]], str]:
    """Per-neuron activation contrast between harmful and harmless prompts.

    For each target layer we capture the per-neuron activation of a chosen
    feed-forward hook point (post-activation MLP intermediate by default),
    pool over non-pad tokens, and accumulate per-neuron running statistics
    over the harmful and the harmless corpus separately. The contrast score
    is then one of:

    * ``mean_abs_diff``  -- ``mean|a|_harmful - mean|a|_harmless``
    * ``tstat``          -- standardized mean-abs difference (label-correlation)
    * ``mean_diff``      -- signed ``mean a_harmful - mean a_harmless``

    Returns ``(per_layer_ranking, module_label)`` where ``module_label`` is the
    hook point actually used (so the report records the true unit axis).
    """
    if tokenizer is None:
        raise ValueError(
            "identify_safety_neurons: method='activation' needs a tokenizer. "
            "Pass it via the `tokenizer` argument (or use safety_circuit_info, "
            "which supplies one)."
        )
    if not harmful_prompts or not harmless_prompts:
        raise ValueError(
            "identify_safety_neurons: method='activation' needs non-empty "
            "`harmful_prompts` and `harmless_prompts` contrast corpora."
        )
    if cfg.activation_score not in ("mean_abs_diff", "tstat", "mean_diff"):
        raise ValueError(
            f"Unknown activation_score: {cfg.activation_score!r} "
            "(expected 'mean_abs_diff', 'tstat', or 'mean_diff')."
        )

    layers = _get_decoder_layers(model)
    if not layers:
        raise RuntimeError(
            "_score_by_activation: could not locate decoder layers."
        )
    targets = [
        li
        for li in (cfg.target_layers if cfg.target_layers is not None else range(len(layers)))
        if 0 <= li < len(layers)
    ]
    if not targets:
        return {}, cfg.activation_module

    # Resolve one hook point per target layer; require a consistent label.
    hook_points: Dict[int, nn.Module] = {}
    module_label = ""
    for li in targets:
        mod, name = _resolve_activation_point(layers[li], cfg)
        if mod is None:
            logger.debug("interpret: layer %d has no usable activation hook; skipping.", li)
            continue
        hook_points[li] = mod
        module_label = module_label or name
    if not hook_points:
        raise RuntimeError(
            "_score_by_activation: no usable feed-forward hook point found on "
            f"any target layer (tried activation_module={cfg.activation_module!r})."
        )

    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    was_training = model.training
    model.eval()

    # Per-layer running accumulators, lazily sized on first batch.
    # n: token count; s1/s1_abs: sum of act / |act|; s2_abs: sum of act**2.
    stats: Dict[str, Dict[int, Dict[str, Any]]] = {
        "harmful": {}, "harmless": {},
    }
    _captured: Dict[int, torch.Tensor] = {}

    def _make_hook(li: int):
        def _hook(_m: nn.Module, _inp: Any, output: Any) -> None:
            t = output[0] if isinstance(output, tuple) else output
            _captured[li] = t.detach()
        return _hook

    def _accumulate(label: str, prompts: List[str]) -> None:
        bs = max(1, int(cfg.activation_batch_size))
        for start in range(0, len(prompts), bs):
            batch = prompts[start:start + bs]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(1, int(cfg.activation_max_tokens)),
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            attn = (
                attn.to(device)
                if attn is not None
                else torch.ones_like(input_ids)
            )
            _captured.clear()
            handles = [
                hook_points[li].register_forward_hook(_make_hook(li))
                for li in hook_points
            ]
            try:
                model(input_ids=input_ids, attention_mask=attn)
            finally:
                for h in handles:
                    h.remove()
            for li, act in _captured.items():
                # act: (batch, seq, n_neurons). Mask out pad tokens so the
                # per-neuron statistics are not contaminated by padding (the
                # padding-contamination failure mode flagged elsewhere in the
                # audit). Then sum over (batch, seq).
                act = act.float()
                if act.dim() == 2:  # (tokens, n_neurons) -- already flat
                    act = act.unsqueeze(0)
                mask = attn.unsqueeze(-1).to(act.dtype)  # (batch, seq, 1)
                if mask.shape[1] != act.shape[1]:
                    # Defensive: hook captured a different seq length.
                    mask = torch.ones(
                        act.shape[:2] + (1,), dtype=act.dtype, device=act.device
                    )
                tok = float(mask.sum().item())
                s1 = (act * mask).sum(dim=(0, 1))
                s1_abs = (act.abs() * mask).sum(dim=(0, 1))
                s2_abs = ((act * act) * mask).sum(dim=(0, 1))
                acc = stats[label].setdefault(
                    li,
                    {
                        # Accumulate on CPU — the updates below add `.cpu()`
                        # tensors, so the accumulators must be on CPU too (the
                        # model/activations live on GPU).
                        "n": 0.0,
                        "s1": torch.zeros_like(s1, device="cpu"),
                        "s1_abs": torch.zeros_like(s1_abs, device="cpu"),
                        "s2_abs": torch.zeros_like(s2_abs, device="cpu"),
                    },
                )
                acc["n"] += tok
                acc["s1"] += s1.cpu()
                acc["s1_abs"] += s1_abs.cpu()
                acc["s2_abs"] += s2_abs.cpu()

    try:
        _accumulate("harmful", list(harmful_prompts))
        _accumulate("harmless", list(harmless_prompts))
    finally:
        if was_training:
            model.train()

    out: Dict[int, List[Tuple[int, float]]] = {}
    for li in hook_points:
        h = stats["harmful"].get(li)
        b = stats["harmless"].get(li)
        if h is None or b is None or h["n"] <= 0 or b["n"] <= 0:
            continue
        nh, nb = h["n"], b["n"]
        mean_abs_h = h["s1_abs"] / nh
        mean_abs_b = b["s1_abs"] / nb
        if cfg.activation_score == "mean_diff":
            score = (h["s1"] / nh) - (b["s1"] / nb)
        elif cfg.activation_score == "tstat":
            # Welch-style standardized mean-abs difference. Variance of |act|
            # is bounded by E[act**2]; use that as a safe over-estimate so the
            # denominator never collapses, giving a scale-robust contrast.
            var_h = (h["s2_abs"] / nh - mean_abs_h * mean_abs_h).clamp_min(0.0)
            var_b = (b["s2_abs"] / nb - mean_abs_b * mean_abs_b).clamp_min(0.0)
            denom = (var_h / nh + var_b / nb).sqrt().clamp_min(1e-8)
            score = (mean_abs_h - mean_abs_b) / denom
        else:  # mean_abs_diff
            score = mean_abs_h - mean_abs_b

        vals = score.tolist()
        scored = (
            (j, abs(float(v))) if cfg.abs_rank else (j, float(v))
            for j, v in enumerate(vals)
        )
        ranked = sorted(scored, key=lambda x: abs(x[1]), reverse=True)
        kept = [
            (j, s) for j, s in ranked if abs(s) > cfg.score_floor
        ][: cfg.top_k_per_layer]
        if kept:
            out[li] = kept

    return out, (module_label or cfg.activation_module)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_safety_neurons(
    model: nn.Module,
    refusal_direction_per_layer: Dict[int, torch.Tensor],
    config: Optional[SafetyNeuronConfig] = None,
    *,
    tokenizer: Any = None,
    harmful_prompts: Optional[List[str]] = None,
    harmless_prompts: Optional[List[str]] = None,
) -> SafetyNeuronReport:
    """Rank per-layer neurons by their relevance to refusal/safety behaviour.

    Two localization paths, selected by ``config.method``:

    * ``"weight"`` (default) -- rank units by absolute cosine between the
      unit's residual-stream write-direction (a column of ``target_module``)
      and the layer's refusal direction. Needs only
      ``refusal_direction_per_layer``; no forward passes.
    * ``"activation"`` -- rank units by a per-neuron *activation contrast*
      between harmful and harmless prompts (feed-forward activation analysis,
      cf. Wei et al. arXiv:2402.05162, Chen et al. arXiv:2406.14144). Needs
      ``tokenizer``, ``harmful_prompts`` and ``harmless_prompts``; does not use
      ``refusal_direction_per_layer``.

    Args:
        model: HF causal LM.
        refusal_direction_per_layer: ``{layer_idx: 1-D tensor (hidden,)}``,
            typically from :func:`safetune.steer.extract_refusal_direction`
            (the ``all_layer_directions`` return value). Used by ``"weight"``
            mode; may be ``{}`` for ``"activation"`` mode.
        config: :class:`SafetyNeuronConfig`. Default uses ``method="weight"``,
            ``target_module="mlp.down_proj"``, ``top_k_per_layer=16``.
        tokenizer: HF tokenizer -- required for ``method="activation"``.
        harmful_prompts: harmful contrast corpus -- required for
            ``method="activation"``.
        harmless_prompts: harmless contrast corpus -- required for
            ``method="activation"``.

    Returns:
        :class:`SafetyNeuronReport` with per-layer top-k unit indices and scores.
    """
    cfg = config or SafetyNeuronConfig()
    if cfg.method == "weight":
        per_layer = _score_by_weight(model, refusal_direction_per_layer, cfg)
        target_module = cfg.target_module
    elif cfg.method == "activation":
        per_layer, target_module = _score_by_activation(
            model,
            tokenizer=tokenizer,
            harmful_prompts=harmful_prompts or [],
            harmless_prompts=harmless_prompts or [],
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown method: {cfg.method!r}")
    rep = SafetyNeuronReport(
        per_layer=per_layer,
        direction_layer=None,
        method=cfg.method,
        target_module=target_module,
    )
    logger.info(
        "interpret.identify_safety_neurons: located %d neurons across %d layers "
        "via %s on %s.",
        sum(len(v) for v in per_layer.values()),
        len(per_layer),
        cfg.method,
        target_module,
    )
    return rep


def safety_circuit_info(
    model: nn.Module,
    tokenizer: Any,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    *,
    top_k_per_layer: int = 16,
    target_module: str = "mlp.down_proj",
    target_layers: Optional[List[int]] = None,
    method: str = "weight",
    activation_module: str = "mlp.act_fn",
    activation_score: str = "mean_abs_diff",
):
    """Convenience: locate safety neurons end-to-end and return a ``CircuitInfo``.

    In ``method="weight"`` mode (default) this extracts a refusal direction
    from the contrast corpus and ranks units by weight-direction cosine. In
    ``method="activation"`` mode it skips the refusal-direction step and ranks
    units directly by the harmful-vs-harmless activation contrast -- the
    refusal direction is still extracted so that ``direction_layer`` is
    populated for downstream consumers, but it does not affect the scores.

    Returns the :class:`CircuitInfo` produced by
    :meth:`SafetyNeuronReport.as_circuit_info`. Plug directly into LSSF, PKE,
    NLSR, or DeepRefusal for targeted patching.

    Args:
        method: ``"weight"`` or ``"activation"`` (see
            :func:`identify_safety_neurons`).
        activation_module: module whose activations are contrasted when
            ``method="activation"``; forwarded to :class:`SafetyNeuronConfig`.
        activation_score: activation contrast score when ``method="activation"``;
            forwarded to :class:`SafetyNeuronConfig`.
    """
    from ...steer.refusal_direction import (  # type: ignore[attr-defined]
        RefusalDirectionConfig,
        extract_refusal_direction,
    )

    layers = _get_decoder_layers(model)
    if not layers:
        raise RuntimeError("safety_circuit_info: cannot locate decoder layers.")
    if method not in ("weight", "activation"):
        raise ValueError(f"safety_circuit_info: unknown method {method!r}.")
    target = target_layers if target_layers is not None else list(range(len(layers)))

    rd_cfg = RefusalDirectionConfig(
        target_layers=target,
        pick_layer=target[len(target) // 2],
        normalize=True,
    )
    _, picked, all_layer = extract_refusal_direction(
        model, tokenizer, harmful_prompts, harmless_prompts, rd_cfg
    )
    sn_cfg = SafetyNeuronConfig(
        method=method,
        top_k_per_layer=top_k_per_layer,
        target_layers=target,
        target_module=target_module,
        activation_module=activation_module,
        activation_score=activation_score,
    )
    report = identify_safety_neurons(
        model,
        all_layer,
        sn_cfg,
        tokenizer=tokenizer,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
    )
    report.direction_layer = picked
    return report.as_circuit_info()


__all__ = [
    "SafetyNeuronConfig",
    "SafetyNeuronReport",
    "identify_safety_neurons",
    "safety_circuit_info",
]


# ---------------------------------------------------------------------------
# Top-level re-export
# ---------------------------------------------------------------------------
# ``identify_safety_neurons`` / ``safety_circuit_info`` are part of the
# verified C-DeltaTheta surface and are documented as importable from the
# package root (``from safetune import identify_safety_neurons``). This module
# is imported during ``safetune`` package initialization (via
# ``safetune.core.interpret``), so by registering the two public callables on
# the partially-initialized ``safetune`` module object here we make the
# documented root-level import resolve without editing ``safetune/__init__.py``.
def _register_root_exports() -> None:
    import sys

    pkg = sys.modules.get("safetune")
    if pkg is None:
        return
    for _name in ("identify_safety_neurons", "safety_circuit_info"):
        if not hasattr(pkg, _name):
            setattr(pkg, _name, globals()[_name])
    _all = getattr(pkg, "__all__", None)
    if isinstance(_all, list):
        for _name in ("identify_safety_neurons", "safety_circuit_info"):
            if _name not in _all:
                _all.append(_name)


_register_root_exports()
