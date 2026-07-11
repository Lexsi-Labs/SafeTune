"""
EAP / EAP-IG safety-circuit discovery — a faithful from-scratch implementation.

Public entry point: :func:`eap_safety_circuit`. Given a model name, a contrast
set of harmful vs. harmless prompts, and a few hyperparameters, this function
discovers the *safety circuit* — the subset of model components most responsible
for the refusal-vs-compliance logit difference — and returns it as a SafeTune
:class:`CircuitInfo` that downstream Recover methods (PKE, NLSR, LSSF,
DeepRefusal) can target.

Algorithm
---------
This module implements **Edge Attribution Patching (EAP)** and its
integrated-gradients refinement **EAP-IG** directly, with no external
dependency:

* EAP — Syed, Rager & Conmy, "Attribution Patching Outperforms Automated
  Circuit Discovery," arXiv:2310.10348.
* EAP-IG — Hanna, Pezzelle & Belinkov, "Have Faith in Faithfulness: Going
  Beyond Circuit Overlap When Finding Model Mechanisms," arXiv:2403.17806.
  Reference repo: https://github.com/hannamw/EAP-IG

The core idea is a first-order (Taylor) approximation to activation patching.
For a component whose output activation is ``z``, the change in a scalar metric
``L`` caused by replacing ``z`` with its corrupted value ``z_corrupt`` is
approximated by

    attribution(z)  ≈  (z_corrupt − z_clean) · ∂L/∂z

where ``∂L/∂z`` is the gradient of the metric w.r.t. the clean activation and
``·`` is a dot product summed over the hidden dimension (and over positions /
batch). EAP gets every component's score from a *single* clean forward+backward
pass plus one corrupted forward pass. EAP-IG replaces the single clean gradient
with the gradient averaged over ``m`` points on the straight line between the
corrupted and clean activations (an integrated-gradients estimate), which the
EAP-IG paper shows yields more *faithful* circuits.

Granularity
-----------
The reference EAP-IG repo (``hannamw/EAP-IG``, ``src/eap/graph.py``) gives every
attention head its own node, named ``a{layer}.h{head}``, and slices the
per-head attention result with an index tuple ``(slice(None), slice(None),
head)``. We reproduce that **per-attention-head granularity** without a
TransformerLens dependency by exploiting the structure of an HF attention
block: the tensor fed into the output projection ``o_proj`` has shape
``(batch, seq, num_heads * head_dim)`` and is the concatenation of the
per-head attention results, so head ``h`` owns the contiguous channel slice
``[h*head_dim : (h+1)*head_dim]`` *before* ``o_proj`` mixes them. A
forward-pre-hook on ``o_proj`` therefore exposes (and can patch) each head's
contribution independently.

* ``granularity="head"`` (default) — scored nodes are, for every decoder block
  ``i``, each attention head ``blocks.i.attn.head<H>`` (its pre-``o_proj``
  channel slice) and the block's MLP-output write ``blocks.i.mlp``. This
  matches the reference repo's per-head node set (MLP stays per-block, as it
  has no head structure).
* ``granularity="block"`` — the legacy coarse node set: each block's whole
  attention-output write ``blocks.i.attn`` and MLP-output write
  ``blocks.i.mlp``. Preserved for backward compatibility.

The attribution math above (``(corrupted − clean) · gradient``, EAP-IG
averaging the gradient over ``ig_steps`` interpolation points) is applied
unchanged to whichever node set is selected. ``top_k_edges`` then thresholds
the scored nodes into the circuit.

If `circuitkit` / `ckit_theta` ever becomes a real package this module does not
need it — the attribution is computed here.
"""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EAPSafetyCircuitConfig:
    """Configuration for the EAP / EAP-IG safety-circuit runner.

    Attributes:
        method: ``"eap"`` for raw Edge Attribution Patching (single clean
            gradient), ``"eap-ig"`` for the integrated-gradients variant
            (gradient averaged over ``ig_steps`` interpolation points).
        target: metric to attribute against. ``"logit_diff"`` (compliance-token
            logit minus refusal-token logit, at the final position) is the
            canonical choice and the only one implemented.
        refusal_token: token string for the refusal class. Default ``"I"``
            (as in "I cannot ..."). Only its first sub-token is used.
        compliance_token: token string for the compliance class. Default
            ``"Sure"``. Only its first sub-token is used.
        top_k_edges: number of top-scoring components to keep in the resulting
            CircuitInfo.
        granularity: node resolution of the component graph. ``"head"``
            (default) decomposes every attention block into per-head nodes
            (``blocks.<L>.attn.head<H>``, the head's pre-``o_proj`` channel
            slice) plus a per-block MLP node, matching the ``hannamw/EAP-IG``
            reference graph. ``"block"`` preserves the legacy coarse node set
            (one ``blocks.<L>.attn`` node per block). MLP nodes are per-block
            in both modes.
        intervention: corrupted-baseline construction. ``"patching"`` (default)
            uses the harmless prompt's activations as the corrupted baseline;
            ``"zero"`` ablates the component to zero; ``"mean"`` uses the
            per-component mean activation over the harmless batch.
        ig_steps: number of integrated-gradients interpolation steps for
            ``method="eap-ig"`` (ignored for ``"eap"``). The EAP-IG paper uses
            values around 5–10.
        batch_size: number of contrast pairs processed per forward/backward
            pass.
        max_seq_len: tokenizer truncation length for prompts.
        device: ``"cpu"``, ``"cuda"``, or ``None`` to auto-select.
        dtype: torch dtype name for model weights (``"float32"`` on CPU).
        output_dir: where to dump raw outputs (scores JSON, contrast pairs).
            If ``None`` a tmpdir is used.
        extra: free-form extra keys, recorded in metadata for provenance.
    """

    method: str = "eap-ig"
    target: str = "logit_diff"
    refusal_token: str = "I"
    compliance_token: str = "Sure"
    top_k_edges: int = 100
    granularity: str = "head"
    intervention: str = "patching"
    ig_steps: int = 5
    batch_size: int = 8
    max_seq_len: int = 64
    device: Optional[str] = None
    dtype: str = "float32"
    output_dir: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def _select_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model_and_tokenizer(model_name: str, cfg: EAPSafetyCircuitConfig):
    """Load a causal LM + tokenizer. Imports torch/transformers lazily so this
    module imports cleanly with no ML stack present."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _select_device(cfg.device)
    dtype = getattr(torch, cfg.dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _first_token_id(tokenizer: Any, text: str) -> int:
    """Return the id of the first sub-token of ``text`` (no special tokens)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        # Fall back to a leading-space variant ("ĠSure" style tokenizers).
        ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"eap_safety_circuit: token {text!r} encodes to nothing.")
    return int(ids[0])


# --------------------------------------------------------------------------
# Component graph
# --------------------------------------------------------------------------
@dataclass
class _Node:
    """One scorable node in the EAP component graph.

    Attributes:
        name: graph name, e.g. ``blocks.3.attn.head7``, ``blocks.3.attn`` or
            ``blocks.3.mlp``.
        module: the ``nn.Module`` to attach the hook to.
        hook: ``"forward"`` captures/patches the module's *output*;
            ``"pre"`` captures/patches the module's *input* (used for per-head
            nodes, where the hooked module is ``o_proj`` and the captured
            tensor is the pre-projection per-head attention result).
        channels: ``None`` for whole-tensor nodes, or ``(start, end)`` for a
            contiguous slice of the last (hidden) dim — the per-head channel
            band ``[h*head_dim : (h+1)*head_dim]``.
    """

    name: str
    module: Any
    hook: str = "forward"
    channels: Optional[Tuple[int, int]] = None


def _resolve_block_modules(layer: Any) -> Dict[str, Any]:
    """Return ``{"attn": <module>, "mlp": <module>}`` for one decoder block.

    Covers the common HF causal-LM zoo:

    * Llama / Mistral / Qwen / Gemma: ``self_attn`` + ``mlp``
    * GPT-2 / Falcon / NeoX-style: ``attn`` + (``mlp`` | ``ffn``)
    """
    comps: Dict[str, Any] = {}
    for attr in ("self_attn", "attn", "attention"):
        if hasattr(layer, attr):
            comps["attn"] = getattr(layer, attr)
            break
    for attr in ("mlp", "ffn", "feed_forward"):
        if hasattr(layer, attr):
            comps["mlp"] = getattr(layer, attr)
            break
    return comps


def _resolve_output_proj(attn_module: Any) -> Optional[Any]:
    """Find the attention output projection (``o_proj`` / ``c_proj`` / …).

    The tensor fed into this projection is the concatenation of the per-head
    attention results; hooking its *input* gives per-head channel slices.
    """
    for attr in ("o_proj", "out_proj", "c_proj", "dense", "wo"):
        if hasattr(attn_module, attr):
            return getattr(attn_module, attr)
    return None


def _resolve_head_geometry(model: Any, attn_module: Any) -> Optional[Tuple[int, int]]:
    """Return ``(num_attention_heads, head_dim)`` for per-head decomposition.

    Prefers the module's own attributes (authoritative for the actual
    projection width), then falls back to the model config. Returns ``None``
    when the geometry cannot be determined (caller then falls back to a
    per-block attention node).
    """
    cfg = getattr(model, "config", None)
    n_heads = getattr(attn_module, "num_heads", None)
    if n_heads is None and cfg is not None:
        n_heads = getattr(cfg, "num_attention_heads", None) or getattr(
            cfg, "n_head", None
        )
    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None and cfg is not None and n_heads:
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        if hidden:
            head_dim = hidden // int(n_heads)
    if not n_heads or not head_dim:
        return None
    return int(n_heads), int(head_dim)


def _enumerate_nodes(model: Any, granularity: str) -> List[_Node]:
    """Build the scorable node list, in graph order.

    With ``granularity="head"`` every attention block contributes one node per
    attention head (``blocks.<i>.attn.head<H>``), captured as the per-head
    channel slice of the projection input; ``granularity="block"`` contributes
    a single ``blocks.<i>.attn`` node per block. MLP nodes are per-block in
    both modes (``blocks.<i>.mlp``).
    """
    from ..._refusal_helpers import _get_decoder_layers

    layers = _get_decoder_layers(model)
    if not layers:
        raise ValueError(
            "eap_safety_circuit: could not locate decoder blocks on the model; "
            "unsupported architecture."
        )

    nodes: List[_Node] = []
    for i, layer in enumerate(layers):
        block = _resolve_block_modules(layer)
        attn = block.get("attn")
        mlp = block.get("mlp")

        if attn is not None:
            o_proj = _resolve_output_proj(attn) if granularity == "head" else None
            geom = (
                _resolve_head_geometry(model, attn)
                if (granularity == "head" and o_proj is not None)
                else None
            )
            if geom is not None:
                n_heads, head_dim = geom
                # Per-head nodes: each head owns a contiguous channel band of
                # the tensor entering o_proj — the concatenated per-head
                # attention results, exactly the reference repo's per-head
                # decomposition (slice (:, :, head) -> contiguous channels).
                for h in range(n_heads):
                    nodes.append(
                        _Node(
                            name=f"blocks.{i}.attn.head{h}",
                            module=o_proj,
                            hook="pre",
                            channels=(h * head_dim, (h + 1) * head_dim),
                        )
                    )
            else:
                # block granularity, or per-head geometry unavailable: fall
                # back to one whole-attention-output node for the block.
                nodes.append(_Node(name=f"blocks.{i}.attn", module=attn, hook="forward"))

        if mlp is not None:
            nodes.append(_Node(name=f"blocks.{i}.mlp", module=mlp, hook="forward"))

    return nodes


# --------------------------------------------------------------------------
# Activation capture
# --------------------------------------------------------------------------
def _module_output_tensor(output: Any):
    """Modules may return a tensor or a tuple ``(hidden, ...)``; pick the
    hidden-state tensor."""
    if isinstance(output, tuple):
        return output[0]
    return output


class _ActivationCapture:
    """Hooks that record (and optionally substitute) per-node activations.

    Supports both node hook kinds:

    * ``forward`` nodes (whole-attention-output / MLP-output) — a forward hook
      on the node's module records/patches the module's *output* tensor.
    * ``pre`` nodes (per-attention-head) — a forward-pre-hook on the shared
      ``o_proj`` module records/patches that module's *input* tensor; each head
      node owns a contiguous channel slice of it. Several head nodes therefore
      share one underlying hook; in patch mode the hook reassembles the full
      projection input from the per-head patch tensors (heads not supplied keep
      their original slice).

    Two modes:

    * ``record`` — store each node's (sliced) activation tensor.
    * ``patch``  — substitute precomputed tensors into the forward graph (used
      for the corrupted baseline and the IG interpolation passes).
    """

    def __init__(self, nodes: List[_Node]):
        self.nodes = nodes
        self.handles: List[Any] = []
        self.recorded: Dict[str, Any] = {}
        self._patch_values: Dict[str, Any] = {}
        self._mode = "record"

        # Group nodes by the (module, hook-kind) pair that physically carries
        # them — per-head nodes on the same o_proj share a single hook.
        self._groups: Dict[Tuple[int, str], List[_Node]] = {}
        for node in nodes:
            self._groups.setdefault((id(node.module), node.hook), []).append(node)

    @staticmethod
    def _slice(tensor: Any, channels: Optional[Tuple[int, int]]):
        if channels is None:
            return tensor
        return tensor[..., channels[0] : channels[1]]

    def _make_forward_hook(self, group: List[_Node]) -> Callable:
        def hook(_module, _inp, output):
            tensor = _module_output_tensor(output)
            if self._mode == "record":
                for node in group:
                    self.recorded[node.name] = self._slice(
                        tensor, node.channels
                    ).detach()
                return output
            # patch mode — substitute supplied node tensors into the output.
            new = tensor
            for node in group:
                if node.name not in self._patch_values:
                    continue
                repl = self._patch_values[node.name]
                if node.channels is None:
                    new = repl
                else:
                    new = new.clone()
                    new[..., node.channels[0] : node.channels[1]] = repl
            if new is tensor:
                return output
            if isinstance(output, tuple):
                return (new,) + tuple(output[1:])
            return new

        return hook

    def _make_pre_hook(self, group: List[_Node]) -> Callable:
        def hook(_module, args):
            if not args:
                return args
            tensor = args[0]
            if self._mode == "record":
                for node in group:
                    self.recorded[node.name] = self._slice(
                        tensor, node.channels
                    ).detach()
                return args
            # patch mode — rebuild the projection input from per-head patches.
            new = tensor
            for node in group:
                if node.name not in self._patch_values:
                    continue
                repl = self._patch_values[node.name]
                if node.channels is None:
                    new = repl
                else:
                    if new is tensor:
                        new = tensor.clone()
                    new[..., node.channels[0] : node.channels[1]] = repl
            if new is tensor:
                return args
            return (new,) + tuple(args[1:])

        return hook

    def __enter__(self) -> "_ActivationCapture":
        for (_, kind), group in self._groups.items():
            module = group[0].module
            if kind == "pre":
                self.handles.append(
                    module.register_forward_pre_hook(self._make_pre_hook(group))
                )
            else:
                self.handles.append(
                    module.register_forward_hook(self._make_forward_hook(group))
                )
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def set_record(self) -> None:
        self._mode = "record"
        self.recorded = {}

    def set_patch(self, values: Dict[str, Any]) -> None:
        self._mode = "patch"
        self._patch_values = values


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------
def _logit_diff_metric(
    logits: Any,
    last_idx: Any,
    compliance_id: int,
    refusal_id: int,
):
    """Compliance-minus-refusal logit at the final real token, summed over batch.

    A higher value means the model is *more* compliant. The attribution then
    answers: which components, if patched toward the harmless (compliant)
    baseline, most increase compliance — i.e. which components implement
    refusal.

    ``last_idx`` is a 1-D tensor of the final non-pad position per example.
    """
    import torch

    batch = logits.shape[0]
    rows = torch.arange(batch, device=logits.device)
    final = logits[rows, last_idx, :]  # (batch, vocab)
    return (final[:, compliance_id] - final[:, refusal_id]).sum()


# --------------------------------------------------------------------------
# Core EAP / EAP-IG attribution
# --------------------------------------------------------------------------
def _tokenize_batch(tokenizer: Any, prompts: List[str], cfg: EAPSafetyCircuitConfig, device: str):
    import torch

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_seq_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    mask = enc["attention_mask"]
    last_idx = mask.sum(dim=1) - 1  # final real token per example
    last_idx = torch.clamp(last_idx, min=0)
    return enc, last_idx


def _attribute_batch(
    model: Any,
    capture: _ActivationCapture,
    nodes: List[_Node],
    clean_enc: Dict[str, Any],
    clean_last: Any,
    corrupt_enc: Dict[str, Any],
    corrupt_last: Any,
    metric_fn: Callable,
    cfg: EAPSafetyCircuitConfig,
) -> Dict[str, float]:
    """Compute EAP / EAP-IG attribution scores for one contrast batch.

    Returns ``{node_name: score}`` summed over positions and batch. ``node_name``
    is per-head (``blocks.<L>.attn.head<H>``) under head granularity.
    """
    import torch

    names = [n.name for n in nodes]

    # Align the corrupted batch to the clean batch's sequence length so patched
    # activations line up position-by-position. The two contrast batches are
    # tokenized independently (padding="longest" per batch), so harmful and
    # harmless prompts of unequal length otherwise produce mismatched
    # activation tensors and crash the "patching" intervention. Clean is the
    # attribution reference and is never modified; the corrupted batch (only a
    # baseline) is padded/truncated on the right to match.
    cl = clean_enc["input_ids"].shape[1]
    if corrupt_enc["input_ids"].shape[1] != cl:
        def _resize(enc, target):
            ids, am = enc["input_ids"], enc["attention_mask"]
            cur = ids.shape[1]
            if cur > target:
                ids, am = ids[:, :target], am[:, :target]
            elif cur < target:
                pad = target - cur
                ids = torch.cat([ids, ids.new_zeros((ids.shape[0], pad))], dim=1)
                am = torch.cat([am, am.new_zeros((am.shape[0], pad))], dim=1)
            return {**enc, "input_ids": ids, "attention_mask": am}
        corrupt_enc = _resize(corrupt_enc, cl)
        corrupt_last = torch.clamp(corrupt_enc["attention_mask"].sum(dim=1) - 1, min=0)

    # 1. Corrupted forward pass — record corrupted activations (no grad).
    capture.set_record()
    with torch.no_grad():
        model(**corrupt_enc)
    corrupt_acts = dict(capture.recorded)

    # 2. Clean forward pass — record clean activations (no grad).
    capture.set_record()
    with torch.no_grad():
        model(**clean_enc)
    clean_acts = dict(capture.recorded)

    # Activation difference (corrupted − clean) per component. Under
    # intervention="zero"/"mean" the corrupted baseline is overridden below.
    act_diff: Dict[str, Any] = {}
    for name in names:
        c = clean_acts[name]
        if cfg.intervention == "zero":
            base = torch.zeros_like(c)
        elif cfg.intervention == "mean":
            base = corrupt_acts[name].mean(dim=(0, 1), keepdim=True).expand_as(c)
        else:  # "patching"
            base = corrupt_acts[name]
            # Corrupted batch may differ in seq length; align on min length.
            if base.shape[1] != c.shape[1]:
                m = min(base.shape[1], c.shape[1])
                base = base[:, :m, :]
                c = c[:, :m, :]
        act_diff[name] = (base - c)

    # 3. Gradient(s) of the metric w.r.t. clean component activations.
    #    EAP   -> one gradient at the clean point (ig_steps effectively 1).
    #    EAP-IG-> average the gradient over `steps` points on the line from the
    #             corrupted activation to the clean activation.
    steps = max(1, cfg.ig_steps) if cfg.method == "eap-ig" else 1

    grad_accum: Dict[str, Any] = {n: torch.zeros_like(clean_acts[n]) for n in names}

    for k in range(0, steps):
        # Plain EAP (Eq.1): a single gradient at the CLEAN activation (alpha=1),
        # NOT the corrupted endpoint. EAP-IG integrates the gradient along the
        # corrupted->clean line on the k/steps grid (the inputs-variant
        # convention, which omits the alpha=1 endpoint).
        alpha = (k / steps) if cfg.method == "eap-ig" else 1.0  # 0 -> corrupted, 1 -> clean
        # Interpolated activations to substitute into the clean forward graph.
        interp: Dict[str, Any] = {}
        for name in names:
            clean = clean_acts[name]
            diff = act_diff[name]  # (base - clean)
            # interpolated = clean + (1 - alpha) * (base - clean)
            #   alpha=1 -> clean ; alpha->0 -> base (corrupted)
            z = clean + (1.0 - alpha) * diff
            z = z.clone().detach().requires_grad_(True)
            interp[name] = z

        capture.set_patch(interp)
        logits = model(**clean_enc).logits
        loss = metric_fn(logits, clean_last)
        model.zero_grad(set_to_none=True)
        loss.backward()

        for name in names:
            g = interp[name].grad
            if g is not None:
                grad_accum[name] = grad_accum[name] + g.detach()

    capture.set_record()  # leave hooks in a benign state

    # 4. Edge score = (corrupted − clean) · (mean gradient), summed over the
    #    hidden dimension, positions, and batch.  Divide the IG gradient sum by
    #    `steps` to get the average gradient (integrated-gradients estimate).
    scores: Dict[str, float] = {}
    for name in names:
        diff = act_diff[name]
        g = grad_accum[name] / steps
        if g.shape[1] != diff.shape[1]:
            m = min(g.shape[1], diff.shape[1])
            g = g[:, :m, :]
            diff = diff[:, :m, :]
        scores[name] = float((diff * g).sum().item())
    return scores


def _run_eap(
    model: Any,
    tokenizer: Any,
    device: str,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    cfg: EAPSafetyCircuitConfig,
) -> Dict[str, float]:
    """Run EAP / EAP-IG over all contrast pairs and return summed scores.

    Clean = harmful prompt (the model refuses → refusal circuit is active).
    Corrupted = harmless prompt (the paired baseline that does not trigger
    refusal). Attribution then ranks components by how much patching them toward
    the harmless baseline shifts the compliance-minus-refusal logit diff.
    """
    nodes = _enumerate_nodes(model, cfg.granularity)
    compliance_id = _first_token_id(tokenizer, cfg.compliance_token)
    refusal_id = _first_token_id(tokenizer, cfg.refusal_token)

    total: Dict[str, float] = {node.name: 0.0 for node in nodes}
    n_pairs = len(harmful_prompts)

    with _ActivationCapture(nodes) as capture:
        for start in range(0, n_pairs, cfg.batch_size):
            end = min(start + cfg.batch_size, n_pairs)
            clean_enc, clean_last = _tokenize_batch(
                tokenizer, harmful_prompts[start:end], cfg, device
            )
            corrupt_enc, corrupt_last = _tokenize_batch(
                tokenizer, harmless_prompts[start:end], cfg, device
            )

            def metric_fn(logits, last_idx):
                return _logit_diff_metric(logits, last_idx, compliance_id, refusal_id)

            batch_scores = _attribute_batch(
                model,
                capture,
                nodes,
                clean_enc,
                clean_last,
                corrupt_enc,
                corrupt_last,
                metric_fn,
                cfg,
            )
            for name, val in batch_scores.items():
                total[name] += val

    # Average over contrast pairs for a batch-size-independent magnitude.
    n_batches = max(1, (n_pairs + cfg.batch_size - 1) // cfg.batch_size)
    return {name: val / n_batches for name, val in total.items()}


# --------------------------------------------------------------------------
# Result -> CircuitInfo
# --------------------------------------------------------------------------
def _scores_to_circuit_info(
    scores: Dict[str, float],
    cfg: EAPSafetyCircuitConfig,
    output_dir: Path,
):
    """Threshold scored components into the top-k circuit and build CircuitInfo."""
    from safetune.core.circuit_kit.interface import (
        CircuitInfo,
        LayerModuleSuggestions,
        SafetyRelevantUnits,
    )

    # Rank by absolute attribution magnitude — a component matters whether it
    # pushes the metric up or down.
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = ranked[: cfg.top_k_edges]

    unit_ids = [name for name, _ in top]
    score_map = {name: val for name, val in top}

    layer_indices = sorted(
        {
            int(name.split(".")[1])
            for name in unit_ids
            if name.startswith("blocks.")
            and len(name.split(".")) >= 2
            and name.split(".")[1].isdigit()
        }
    )
    # Classify each node name into its component kind. Node names are either
    # ``blocks.<L>.attn.head<H>`` (per-head), ``blocks.<L>.attn`` (per-block
    # attention) or ``blocks.<L>.mlp``.
    def _kind_of(name: str) -> Optional[str]:
        parts = name.split(".")
        if "attn" in parts:
            return "attn"
        if "mlp" in parts:
            return "mlp"
        return None

    kinds = {k for k in (_kind_of(n) for n in unit_ids) if k is not None}
    # Heads named in the circuit, e.g. {"blocks.0.attn.head3", ...}.
    head_units = sorted(n for n in unit_ids if ".head" in n)
    # Map the "attn"/"mlp" component kinds onto concrete projection modules so
    # downstream LoRA targeting has something to attach to. Per-head attention
    # nodes still LoRA-target the block's attention projections (LoRA cannot
    # address an individual head); the head ranking is carried in unit_ids /
    # the scores JSON for finer-grained downstream use.
    target_modules: List[str] = []
    if "attn" in kinds:
        target_modules += ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "mlp" in kinds:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    target_modules = sorted(set(target_modules))

    # Persist raw scores for inspection / reproducibility.
    scores_path = output_dir / "eap_scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "method": cfg.method,
                "granularity": cfg.granularity,
                "intervention": cfg.intervention,
                "ig_steps": cfg.ig_steps,
                "all_scores": scores,
                "circuit": score_map,
            },
            indent=2,
        )
    )

    return CircuitInfo(
        safety_units=SafetyRelevantUnits(
            layer_indices=layer_indices,
            module_names=sorted(kinds),
            unit_ids=unit_ids,
            activation_correlation=score_map,
            metadata={
                "method": cfg.method,
                "target": cfg.target,
                "intervention": cfg.intervention,
                "granularity": cfg.granularity,
                "head_units": head_units,
                "score": "EAP attribution (corrupted-clean act diff . gradient)",
            },
        ),
        layer_suggestions=LayerModuleSuggestions(
            target_modules=target_modules,
            layer_subset=layer_indices,
            priority={
                name: abs(val) for name, val in score_map.items()
            },
            metadata={"source": "safetune.core.interpret.eap (native EAP/EAP-IG)"},
        ),
        raw_output_path=str(scores_path),
        metadata={
            "method": cfg.method,
            "intervention": cfg.intervention,
            "ig_steps": cfg.ig_steps if cfg.method == "eap-ig" else None,
            "granularity": (
                "per-attention-head (blocks.<L>.attn.head<H>) + per-block MLP"
                if cfg.granularity == "head"
                else "decoder-block component (attn / mlp output)"
            ),
            "extra": dict(cfg.extra or {}),
        },
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def eap_safety_circuit(
    model_name: str,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    config: Optional[EAPSafetyCircuitConfig] = None,
):
    """Discover the safety circuit with native EAP / EAP-IG attribution.

    Caches clean (harmful-prompt) and corrupted (harmless-prompt) activations of
    every graph node, computes each node's attribution score as
    ``(corrupted − clean activation) · gradient-of-metric`` (EAP-IG averages the
    gradient over ``ig_steps`` interpolation points), ranks the nodes, and
    thresholds the top ``top_k_edges`` into a circuit.

    The node set is per-attention-head by default (``config.granularity="head"``):
    each attention block contributes one node per head, named
    ``blocks.<L>.attn.head<H>`` and captured as that head's contiguous channel
    slice of the tensor entering the output projection ``o_proj``; MLP nodes
    stay per-block. Set ``config.granularity="block"`` for the legacy coarse
    node set (one ``blocks.<L>.attn`` node per block).

    Args:
        model_name: HF model id or local path of the causal LM to analyze.
        harmful_prompts: prompts that trigger refusal (the "clean" inputs whose
            mechanism we attribute).
        harmless_prompts: paired benign prompts (the "corrupted" baseline).
            Must be the same length as ``harmful_prompts``.
        config: optional :class:`EAPSafetyCircuitConfig`; defaults are used if
            ``None``.

    Returns:
        :class:`safetune.core.circuit_kit.CircuitInfo` holding the top-k scored
        components, their parent layers, per-component attribution scores, and
        suggested LoRA target modules — ready to feed LSSF / PKE / NLSR /
        DeepRefusal.

    Raises:
        ValueError: if the contrast prompt lists are of mismatched length or
            empty, or the model architecture is unsupported.
        ImportError: if ``torch`` / ``transformers`` are not installed (only at
            call time — importing this module never requires them).
    """
    if len(harmful_prompts) != len(harmless_prompts):
        raise ValueError(
            f"eap_safety_circuit: contrast lists must be the same length; "
            f"got {len(harmful_prompts)} harmful + {len(harmless_prompts)} harmless."
        )
    if not harmful_prompts:
        raise ValueError("eap_safety_circuit: contrast prompt lists are empty.")

    cfg = config or EAPSafetyCircuitConfig()
    if cfg.method not in ("eap", "eap-ig"):
        raise ValueError(
            f"eap_safety_circuit: unknown method {cfg.method!r}; "
            f"expected 'eap' or 'eap-ig'."
        )
    if cfg.intervention not in ("patching", "zero", "mean"):
        raise ValueError(
            f"eap_safety_circuit: unknown intervention {cfg.intervention!r}; "
            f"expected 'patching', 'zero', or 'mean'."
        )
    if cfg.granularity not in ("head", "block"):
        raise ValueError(
            f"eap_safety_circuit: unknown granularity {cfg.granularity!r}; "
            f"expected 'head' or 'block'."
        )

    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir
        else Path(tempfile.mkdtemp(prefix="safetune_eap_"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the contrast set for reproducibility.
    (output_dir / "contrast_pairs.json").write_text(
        json.dumps(
            [
                {"harmful": h, "harmless": s}
                for h, s in zip(harmful_prompts, harmless_prompts)
            ],
            indent=2,
        )
    )

    try:
        model, tokenizer, device = _load_model_and_tokenizer(model_name, cfg)
    except ImportError as e:  # torch / transformers missing
        raise ImportError(
            "eap_safety_circuit requires `torch` and `transformers` to be "
            "installed to run attribution. Install the ML extras and retry."
        ) from e

    logger.info(
        "eap_safety_circuit: running %s (intervention=%s, ig_steps=%s) on %s "
        "with %d contrast pairs on %s.",
        cfg.method,
        cfg.intervention,
        cfg.ig_steps if cfg.method == "eap-ig" else "-",
        model_name,
        len(harmful_prompts),
        device,
    )

    scores = _run_eap(
        model, tokenizer, device, harmful_prompts, harmless_prompts, cfg
    )
    return _scores_to_circuit_info(scores, cfg, output_dir)


__all__ = ["EAPSafetyCircuitConfig", "eap_safety_circuit"]
