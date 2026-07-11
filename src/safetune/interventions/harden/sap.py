"""
SAP Trainer adapter.

Faithful :class:`transformers.Trainer` wrapper for **Safety-Aware Probing**
(SAP), the optimization framework of

    Wu, Zhang, Wei, Zhang, Luan & Sun, "Secure LLM Fine-Tuning via
    Safety-Aware Probing" (arXiv:2505.16737).  Reference implementation:
    https://github.com/ChengcanWu/SAP  (``SAPcode/train.py``).

SAP is a **bilevel** defense.  Its defining mechanism — see Algorithm 1 of the
paper — is a *hidden-state probe* ``V`` (learnable additive tensors ``v_j`` on
the outputs of selected layers, ``x_j = l_j(x_{j-1}) + v_j``) optimized in an
*inner* step to maximize the safe-useful gap loss

    L_su(W, V) = L_useful(W + ΔW_harmful, V) - L_useful(W, V),

where ``ΔW_harmful = ε · ∇_W L_safe`` is the harmful-critical weight
perturbation.  The *outer* step updates ``W`` to minimize ``L_useful(W, V_safe)``
under the perturbed hidden states.

This module implements that bilevel loop on top of ``transformers.Trainer``:

* ``_SAPProbe`` registers forward hooks that add a learnable bias ``v_j`` to the
  output of selected (by default middle) decoder layers — the probe perturbs
  *activations*, never weights (Eq. 1-2 of the paper).
* ``training_step`` performs, per useful batch, the full Algorithm 1 step:
  compute ``ΔW_harmful`` from a contrastive safety batch, run the inner step to
  obtain ``V_safe = β · ∇_V L_su``, then return the useful loss at ``(W, V_safe)``
  so the Trainer's optimizer takes the outer ``W`` step.

The legacy precomputed-``safety_gradients`` weight-shift path is preserved for
backward compatibility: if no contrastive safety source is supplied the trainer
degrades gracefully (plain training, or the old static weight shift).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

try:
    import torch
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e

try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e

try:
    from safetune.core.optim.sap import SafetyAwareProbingWrapper
    _SAP_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    SafetyAwareProbingWrapper = None  # type: ignore[assignment]
    _SAP_IMPORT_ERROR = _e


if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class SAPConfig(TrainingArguments):  # type: ignore[misc]
        """``TrainingArguments`` with SAP hyperparameters.

        Defaults follow the paper's reported configuration (Section V-A2 and
        Algorithm 1 of arXiv:2505.16737):

        * ``grad_rate`` — relative scale of the harmful perturbation ``ΔW_harmful``;
          the actual ``ε`` is ``grad_rate · ‖W‖`` (matches ``SAPcode/train.py``,
          which scales the normalized contrastive gradient by ``grad_rate * lora_norm``).
        * ``v_update_step`` — inner probe step size ``β`` (paper default ``5e-2``).
        * ``probe_layers`` — indices of layers carrying the probe ``v_j``.  ``None``
          selects the contiguous middle block ``[T//4, 3T//4)`` (paper default
          ``v_[11:20]`` for a 32-layer model).
        """

        grad_rate: float = 0.1
        v_update_step: float = 5e-2
        probe_layers: Optional[Sequence[int]] = None
else:  # pragma: no cover
    class SAPConfig(object):  # type: ignore[assignment]
        pass


class _SAPProbe:
    """Hidden-state probe ``V`` (Eq. 1-2 of the SAP paper).

    Registers a forward hook on each selected decoder layer that adds a
    learnable additive tensor ``v_j`` to that layer's output, i.e.
    ``x_j = l_j(x_{j-1}) + v_j``.  The probe perturbs *activations*, not
    weights.  Each ``v_j`` is allocated lazily on the first forward pass so its
    shape exactly matches the layer's hidden state, then kept at ``0`` between
    SAP steps (the paper re-initializes ``V = 0`` every step).
    """

    def __init__(self, model: "torch.nn.Module", layer_indices: Sequence[int]):
        self.model = model
        self.layer_indices = list(layer_indices)
        self._layers = self._locate_decoder_layers(model)
        self._v: Dict[int, "torch.Tensor"] = {}
        self._handles: List[Any] = []
        self.enabled = False

    # -- layer discovery ---------------------------------------------------
    @staticmethod
    def _locate_decoder_layers(model: "torch.nn.Module") -> Optional["torch.nn.ModuleList"]:
        """Best-effort locate the decoder ``ModuleList`` of a HF causal LM."""
        candidates = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                # decoder stacks are typically named ``...layers``/``...h``
                leaf = name.rsplit(".", 1)[-1]
                if leaf in ("layers", "h", "blocks", "decoder"):
                    candidates.append((len(module), module))
        if not candidates:
            for _name, module in model.named_modules():
                if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                    candidates.append((len(module), module))
        if not candidates:
            return None
        # the transformer block stack is the longest ModuleList
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    @property
    def available(self) -> bool:
        return self._layers is not None and len(self.layer_indices) > 0

    def parameters(self) -> List["torch.Tensor"]:
        return list(self._v.values())

    # -- hook plumbing -----------------------------------------------------
    def _make_hook(self, idx: int) -> Callable[..., Any]:
        def hook(_module, _inputs, output):  # noqa: ANN001
            if not self.enabled:
                return output
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            if not isinstance(hidden, torch.Tensor):
                return output
            v = self._v.get(idx)
            if v is None or tuple(v.shape) != tuple(hidden.shape):
                # lazily (re)allocate the probe to match the hidden state
                v = torch.zeros_like(hidden, requires_grad=True)
                self._v[idx] = v
            hidden = hidden + v
            if is_tuple:
                return (hidden,) + tuple(output[1:])
            return hidden

        return hook

    def attach(self) -> None:
        if not self.available or self._handles:
            return
        n = len(self._layers)  # type: ignore[arg-type]
        for idx in self.layer_indices:
            if 0 <= idx < n:
                self._handles.append(
                    self._layers[idx].register_forward_hook(self._make_hook(idx))  # type: ignore[index]
                )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self) -> None:
        """Re-initialize ``V = 0`` (paper line 3: ``Initialize V = 0``)."""
        self._v = {}

    @contextmanager
    def active(self):
        """Enable the probe for the enclosed forward/backward passes."""
        prev = self.enabled
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = prev

    def zero_grad(self) -> None:
        for v in self._v.values():
            if v.grad is not None:
                v.grad = None

    def detach_grads(self) -> Dict[int, "torch.Tensor"]:
        return {
            idx: v.grad.detach().clone()
            for idx, v in self._v.items()
            if v.grad is not None
        }


class SAPTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """Trainer implementing the SAP bilevel optimization (arXiv:2505.16737).

    Args:
        safety_dataloader: an iterable / dataloader yielding contrastive safety
            batches with keys ``input_ids``, ``attention_mask``,
            ``chosen_labels`` (safe response) and ``rejected_labels`` (harmful
            response).  When supplied, every step recomputes the harmful
            direction ``ΔW_harmful = ε·∇_W L_safe`` and runs the bilevel inner
            probe optimization — the faithful SAP algorithm.
        safety_gradients: *(legacy / fallback)* a precomputed safety gradient
            dict ``{name: tensor}``.  Used only when ``safety_dataloader`` is not
            given; reproduces the old static weight-shift behaviour for
            backward compatibility.
        contrastive_temperature: temperature of the contrastive safety loss
            ``L_safe`` (paper / ``SAPcode``: ``1.0``).

    All SAP-specific arguments are optional keyword arguments with defaults, so
    the public constructor signature stays a superset of the original.
    """

    def __init__(
        self,
        *args: Any,
        safety_gradients: Optional[Dict[str, Any]] = None,
        safety_dataloader: Optional[Iterable[Dict[str, Any]]] = None,
        contrastive_temperature: float = 1.0,
        probe_layers: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for SAPTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "torch is required for SAPTrainer"
            ) from _TORCH_IMPORT_ERROR
        if _SAP_IMPORT_ERROR is not None:
            raise ImportError(
                "safetune.core.optim.sap is unavailable"
            ) from _SAP_IMPORT_ERROR

        super().__init__(*args, **kwargs)

        self.grad_rate: float = float(getattr(self.args, "grad_rate", 0.1))
        self.v_update_step: float = float(getattr(self.args, "v_update_step", 5e-2))
        self.contrastive_temperature: float = float(contrastive_temperature)

        # ``SafetyAwareProbingWrapper`` is retained for the legacy weight-shift
        # path and for its faithful contrastive ``L_safe`` helper.
        self._sap = SafetyAwareProbingWrapper(model=self.model, grad_rate=self.grad_rate)

        # legacy fallback inputs
        self._safety_gradients = safety_gradients or {}

        # faithful-SAP inputs
        self._safety_dataloader = safety_dataloader
        self._safety_iter: Optional[Iterable[Dict[str, Any]]] = None

        # hidden-state probe V (paper Eq. 1-2)
        cfg_layers = getattr(self.args, "probe_layers", None)
        layers = probe_layers if probe_layers is not None else cfg_layers
        self._probe = _SAPProbe(
            self.model, self._resolve_probe_layers(self.model, layers)
        )
        if self._safety_dataloader is not None:
            self._probe.attach()

    # -- probe-layer selection -------------------------------------------
    @staticmethod
    def _resolve_probe_layers(
        model: "torch.nn.Module", layers: Optional[Sequence[int]]
    ) -> List[int]:
        """Resolve probe layer indices.

        The paper's default probe set is the *contiguous middle* layers
        (``v_[11:20]`` for 32-layer Llama-2).  When ``layers`` is not given we
        select the middle half ``[T//4, 3T//4)``.
        """
        if layers is not None:
            return [int(i) for i in layers]
        n = 0
        for _name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > n:
                leaf = _name.rsplit(".", 1)[-1]
                if leaf in ("layers", "h", "blocks", "decoder") or len(module) > n:
                    n = max(n, len(module))
        if n <= 0:
            return []
        return list(range(n // 4, (3 * n) // 4))

    # -- contrastive safety batch ----------------------------------------
    def _next_safety_batch(self) -> Optional[Dict[str, "torch.Tensor"]]:
        if self._safety_dataloader is None:
            return None
        if self._safety_iter is None:
            self._safety_iter = iter(self._safety_dataloader)
        try:
            return next(self._safety_iter)  # type: ignore[arg-type]
        except StopIteration:
            self._safety_iter = iter(self._safety_dataloader)
            try:
                return next(self._safety_iter)  # type: ignore[arg-type]
            except StopIteration:
                return None

    def _compute_harmful_direction(
        self, model: "torch.nn.Module", batch: Dict[str, "torch.Tensor"]
    ) -> Dict[str, "torch.Tensor"]:
        """ΔW_harmful = ε·∇_W L_safe  (Algorithm 1, line 2).

        ``∇_W L_safe`` is the contrastive safety gradient; ``ε`` is realized as
        ``grad_rate · ‖W‖`` applied to the *normalized* gradient — exactly the
        ``merge_lora_parameters(model, normalize(g), grad_rate * lora_norm)``
        scaling of the authors' ``SAPcode/train.py``.
        """
        device = next(model.parameters()).device
        move = lambda t: t.to(device) if isinstance(t, torch.Tensor) else t
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            grads = SafetyAwareProbingWrapper.compute_contrastive_safety_gradient(
                model,
                input_ids=move(batch["input_ids"]),
                attention_mask=move(batch["attention_mask"]),
                chosen_labels=move(batch["chosen_labels"]),
                rejected_labels=move(batch["rejected_labels"]),
                temperature=self.contrastive_temperature,
            )
        model.zero_grad(set_to_none=True)
        return grads

    @contextmanager
    def _shift_weights(self, direction: Dict[str, "torch.Tensor"], scale: float):
        """Temporarily shift ``W`` by ``scale·direction`` (a normalized dict).

        ``scale = +ε`` shifts into ``W + ΔW_harmful``; the context restores
        ``W`` exactly on exit.
        """
        applied: List[Any] = []
        with torch.no_grad():
            named = dict(self.model.named_parameters())
            for name, g in direction.items():
                p = named.get(name)
                if p is None:
                    continue
                p.data.add_(g.to(p.device, p.dtype), alpha=scale)
                applied.append((p, g))
        try:
            yield
        finally:
            with torch.no_grad():
                for p, g in applied:
                    p.data.add_(g.to(p.device, p.dtype), alpha=-scale)

    @staticmethod
    def _useful_loss(model: "torch.nn.Module", inputs: Dict[str, Any]) -> "torch.Tensor":
        out = model(**inputs)
        loss = out.loss if hasattr(out, "loss") else out["loss"]
        return loss

    # -- SAP bilevel training step ---------------------------------------
    def _sap_training_step(
        self, model: "torch.nn.Module", inputs: Dict[str, Any]
    ) -> "torch.Tensor":
        """One faithful SAP step (Algorithm 1 of arXiv:2505.16737).

        line 2 — ``ΔW_harmful = ε·∇_W L_safe``
        line 3 — ``Initialize V = 0``
        line 4 — ``∇_V L_su = ∇_V L_useful(W+ΔW,V) - ∇_V L_useful(W,V)``
        line 5 — ``V_safe = β·∇_V L_su``
        line 6 — ``∇_W L_useful = ∇_W L_useful(W, V_safe)``
        line 7 — ``W ← W - α·∇_W L_useful``  (taken by the Trainer optimizer)
        """
        safety_batch = self._next_safety_batch()
        if safety_batch is None:
            return self._fallback_training_step(model, inputs)

        # Gradient-accumulation safety: the inner probe passes below use
        # ``model.zero_grad()`` to isolate PROBE gradients that are consumed
        # immediately.  Those calls must not erase user gradients accumulated
        # by earlier micro-batches, so stash ``param.grad`` here and restore
        # it before the final (outer) Trainer step, which then accumulates on
        # top as usual.
        stashed_grads = [
            (p, p.grad) for p in model.parameters() if p.grad is not None
        ]
        for p, _ in stashed_grads:
            p.grad = None

        norm_grads: Dict[str, "torch.Tensor"] = {}
        try:
            # --- line 2: harmful-critical direction ΔW_harmful ------------
            norm_grads = self._compute_harmful_direction(model, safety_batch)
            if norm_grads:
                # ε = grad_rate · ‖W‖  (over the perturbed params; matches SAPcode)
                with torch.no_grad():
                    named = dict(model.named_parameters())
                    flat = [named[n].detach().flatten() for n in norm_grads if n in named]
                    param_norm = torch.cat(flat).norm(p=2).item() if flat else 0.0
                epsilon = self.grad_rate * param_norm

                # --- line 3: initialize the hidden-state probe V = 0 ------
                self._probe.reset()
                self._probe.zero_grad()
                model.zero_grad(set_to_none=True)

                # --- line 4a: -∇_V L_useful(W + ΔW_harmful, V) ------------
                # W frozen so the probe is the only thing receiving gradient here.
                w_grad_flags = {
                    n: p.requires_grad for n, p in model.named_parameters()
                }
                for p in model.parameters():
                    p.requires_grad_(False)
                with self._shift_weights(norm_grads, scale=epsilon):
                    with self._probe.active(), torch.enable_grad():
                        loss_perturbed = self._useful_loss(model, inputs)
                        (-loss_perturbed).backward()
                # restore W trainability
                for n, p in model.named_parameters():
                    p.requires_grad_(w_grad_flags.get(n, p.requires_grad))

                # --- line 4b: +∇_V L_useful(W, V) -------------------------
                # Evaluated at the (unshifted) weights W with the probe
                # active.  The probe gradient now accumulates to  -∇_V L_su.
                with self._probe.active(), torch.enable_grad():
                    loss_clean = self._useful_loss(model, inputs)
                    loss_clean.backward()

                # --- line 5: V_safe = β·∇_V L_su --------------------------
                # accumulated probe grad = -∇_V L_su  ⇒  ascent on L_su.
                with torch.no_grad():
                    for idx, v in self._probe._v.items():
                        if v.grad is not None:
                            # V_safe = 0 + β · ∇_V L_su  =  -β · (accumulated grad)
                            v.data.add_(v.grad, alpha=-self.v_update_step)
                self._probe.zero_grad()
                # clear the W-gradient residue of the line-4b backward pass
                # (probe-internal; user grads are stashed and restored below)
                model.zero_grad(set_to_none=True)
        finally:
            # Restore user gradients accumulated by earlier micro-batches.
            for p, g in stashed_grads:
                p.grad = g

        if not norm_grads:
            return self._fallback_training_step(model, inputs)

        # --- line 6: ∇_W L_useful(W, V_safe), line 7 handled by optimizer -
        # Standard Trainer step with the probe (now == V_safe) active so the
        # outer W gradient sees the safety-aware perturbed hidden states.
        with self._probe.active():
            loss = self._fallback_training_step(model, inputs)
        return loss

    def _fallback_training_step(
        self, model: "torch.nn.Module", inputs: Dict[str, Any]
    ) -> "torch.Tensor":
        try:
            return Trainer.training_step(self, model, inputs)  # type: ignore[arg-type]
        except TypeError:
            return Trainer.training_step(self, model, inputs, None)  # type: ignore[arg-type]

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        # Faithful SAP path: contrastive safety batches available.
        if self._safety_dataloader is not None and self._probe.available:
            return self._sap_training_step(model, inputs)

        # Legacy fallback: precomputed static safety-gradient weight shift.
        if self._safety_gradients:
            with self._sap.probe_safe_parameters(self._safety_gradients):
                return self._fallback_training_step(model, inputs)

        # No safety signal: plain training.
        return self._fallback_training_step(model, inputs)
