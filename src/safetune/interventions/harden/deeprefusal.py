"""
DeepRefusal Trainer — faithful LoRA fine-tuning implementation.

Faithful implementation of the DeepRefusal algorithm from:

    "Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via
    Probabilistically Ablating Refusal Direction"
    Xie, Zhu, Shi, Xu, Pang, Meng, Wang & Lin. Findings of EMNLP 2025.
    ACL Anthology: https://aclanthology.org/2025.findings-emnlp.956/
    arXiv: https://arxiv.org/abs/2509.15202
    Code:  https://github.com/YuanBoXie/DeepRefusal

Algorithm summary
-----------------
DeepRefusal trains a LoRA adapter so the model learns to refuse even from
"ablated" activations — i.e., from states where the refusal direction has been
projected out, mimicking a direction-erasure jailbreak.

1. **Refusal-direction ablation hooks** (Section 3.2 of the paper).
   At each forward pass, a random subset of decoder layers is chosen.  For each
   selected layer, the unit refusal direction ``d̂ = d / ‖d‖`` is projected out
   of the hidden states at response-token positions with probability
   ``ablation_prob``::

       h_l  ←  h_l - (h_l · d̂) * d̂

   This stochastically simulates abliteration attacks during training, forcing
   the model to learn refusal pathways that do not rely on a single direction.

2. **Two-part training objective** (Eq. 1 / Section 3.3)::

       L = alpha * L_harmful + (1 - alpha) * L_benign

   where:

   * ``L_harmful`` — cross-entropy on ``(harmful_prompt, refusal_response)``
     pairs: standard next-token CE on the refusal token sequence, conditioned on
     the harmful prompt.  Trains the model to explicitly refuse harmful queries
     even when activations are ablated.

   * ``L_benign`` — cross-entropy on benign instruction-following pairs: standard
     SFT on helpfulness data, so the LoRA adapter does not collapse onto pure
     refusal and loses general capability.

3. **LoRA fine-tuning** (Section 3.1).  Only the LoRA adapter parameters are
   updated; the base model weights are frozen.  After training the adapter is
   optionally merged into the base model (``merge_after_training=True``).

Relation to ``safetune.recover.deeprefusal``
--------------------------------------------
:func:`safetune.recover.deeprefusal.apply_deeprefusal` is a *training-free*
weight edit (a SafeTune-original heuristic) that pre-dates this module.  The
present :class:`DeepRefusalTrainer` is the *faithful* implementation of the
published paper and requires gradient updates, a training dataset, and (by
default) peft.  The two share only the high-level goal of hardening refusal
against direction-based attacks; the mechanisms are entirely different.

Data format
-----------
The trainer accepts two separate datasets:

- ``harmful_dataset``  — each example must have ``input_ids``, ``attention_mask``
  and ``labels``.  Labels should be -100 for harmful-prompt tokens (no loss on
  the prompt) and real token ids for the *refusal response* tokens.
- ``benign_dataset``   — same format; benign instruction-following examples.

Internally, ``compute_loss`` draws one batch from each dataset on every step
and computes the weighted loss.  If only one dataset is supplied the other loss
term is skipped.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — torch
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e

# ---------------------------------------------------------------------------
# Optional imports — transformers
# ---------------------------------------------------------------------------
try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e

# ---------------------------------------------------------------------------
# Optional imports — peft (LoRA)
# ---------------------------------------------------------------------------
try:
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-untyped]
    _PEFT_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    _PEFT_IMPORT_ERROR = _e

# ---------------------------------------------------------------------------
# DeepRefusalConfig
# ---------------------------------------------------------------------------
if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class DeepRefusalConfig(TrainingArguments):  # type: ignore[misc]
        """``TrainingArguments`` subclass exposing DeepRefusal hyperparameters.

        All fields have paper-faithful defaults so passing a plain
        ``TrainingArguments`` still works; ``DeepRefusalConfig`` is optional
        but recommended.

        Fields
        ------
        ablation_prob : float
            Probability of applying the refusal-direction projection at a
            selected decoder layer (paper: ``p_ablate = 0.5``, Section 3.2).
        alpha : float
            Weight on the harmful-data loss term (paper: ``alpha = 0.2``,
            Section 3.3).  The benign term receives weight ``1 - alpha``.
        lora_r : int
            LoRA adapter rank (paper default 16).  Set to 0 to disable LoRA
            and train all parameters (not recommended; deviates from paper).
        lora_alpha : int
            LoRA scaling factor.  Paper convention: ``2 * lora_r``.
        lora_dropout : float
            Dropout probability applied inside each LoRA adapter layer.
        target_modules : list of str or None
            Names of linear modules to attach LoRA to.  ``None`` lets peft
            infer defaults for the model architecture.
        merge_after_training : bool
            When ``True`` (default), the LoRA adapter is merged into the base
            model's weights via ``merge_and_unload()`` after training.
        """

        ablation_prob: float = 0.5
        alpha: float = 0.2
        lora_r: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.05
        target_modules: Optional[List[str]] = field(default=None)
        merge_after_training: bool = True

else:  # pragma: no cover
    class DeepRefusalConfig(object):  # type: ignore[assignment]
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_decoder_layers(model: Any) -> List[Any]:
    """Return the list of decoder-layer modules from an HF causal LM.

    Probes the standard attribute paths used by LLaMA / Mistral / Qwen /
    GPT-2 / Pythia families before falling back to an ``nn.ModuleList`` scan.
    """
    for attr_path in (
        "model.language_model.layers",  # Gemma-3
        "model.layers",                 # LLaMA / Mistral / Qwen
        "transformer.h",                # GPT-2 / Bloom
        "model.decoder.layers",         # OPT
        "model.h",                      # GPT-J
        "gpt_neox.layers",              # Pythia / NeoX
        "base_model.model.model.layers",  # peft-wrapped LLaMA
    ):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    # Last resort: walk all named modules, pick the largest ModuleList.
    if nn is not None:
        best: Optional[Any] = None
        best_len = 0
        for _name, child in model.named_modules():
            if isinstance(child, nn.ModuleList) and len(child) > best_len:
                best = child
                best_len = len(child)
        if best is not None:
            return list(best)
    return []


def _ce_loss_from_logits(logits: Any, labels: Any) -> Any:
    """Standard next-token cross-entropy (HF convention).

    Shifts so that position ``t`` predicts token ``t + 1``.  Positions with
    label ``-100`` are excluded from the loss.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _make_dataloader(dataset: Any, batch_size: int) -> Iterator:
    """Wrap *dataset* in an infinitely cycling DataLoader."""
    try:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        return cycle(loader)
    except Exception:
        # If DataLoader construction fails (e.g. already-batched iterable),
        # fall back to cycling the raw iterable.
        return cycle(dataset)


def _next_batch(iterator: Iterator) -> Any:
    """Pull the next batch from a cycling iterator; returns None on failure."""
    try:
        return next(iterator)
    except StopIteration:
        return None


def _prepare_batch(batch: Any, device: Any) -> Any:
    """Move every tensor in *batch* to *device*."""
    if not isinstance(batch, dict):
        return batch
    return {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# Ablation hook
# ---------------------------------------------------------------------------

class _AblationHook:
    """Post-forward hook that projects the refusal direction out of hidden states.

    Registered on a single decoder layer.  The hook is a no-op unless
    :attr:`active` is ``True`` **and** a Bernoulli draw with probability
    :attr:`ablation_prob` succeeds.

    The projection implements Eq. (2) of the paper::

        h  ←  h - (h · d̂) d̂       where d̂ = d / ‖d‖

    applied element-wise to every token position in the batch.
    """

    def __init__(self, d_norm: Any, ablation_prob: float) -> None:
        self.d_norm = d_norm          # unit refusal direction, shape (H,)
        self.ablation_prob = ablation_prob
        self.active: bool = False

    def __call__(self, module: Any, inputs: Any, outputs: Any) -> Any:
        if not self.active:
            return outputs

        # Stochastic ablation gate.
        if random.random() >= self.ablation_prob:
            return outputs

        # Unpack hidden states; HF decoder layers return tuples.
        if isinstance(outputs, tuple):
            hidden = outputs[0]
            rest = outputs[1:]
        else:
            hidden = outputs
            rest = None

        if hidden is None or not isinstance(hidden, torch.Tensor):
            return outputs

        # h ← h - (h · d̂) d̂  (broadcast over batch and sequence dimensions)
        d = self.d_norm.to(hidden.device).to(hidden.dtype)  # (H,)
        # (B, T, H) @ (H,) → (B, T) — scalar projection coefficient per position
        proj_coeff = (hidden @ d).unsqueeze(-1)             # (B, T, 1)
        ablated = hidden - proj_coeff * d.unsqueeze(0).unsqueeze(0)  # (B, T, H)

        if rest is None:
            return ablated
        return (ablated,) + rest


# ---------------------------------------------------------------------------
# DeepRefusalTrainer
# ---------------------------------------------------------------------------

class DeepRefusalTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """HF ``Trainer`` implementing the DeepRefusal LoRA fine-tuning algorithm.

    DeepRefusal (Xie et al., arXiv:2509.15202) trains a LoRA adapter so the
    model learns to refuse *even when the refusal direction has been projected
    out of its activations*, hardening it against abliteration-style jailbreaks.

    Parameters
    ----------
    refusal_direction : torch.Tensor
        1-D tensor of shape ``(hidden_size,)`` — the refusal direction
        extracted from the model (e.g. via
        :func:`safetune.steer.extract_refusal_direction`).  The direction is
        normalised internally; un-normalised vectors are accepted.
    harmful_dataset :
        Dataset of ``(harmful_prompt, refusal_response)`` training pairs.
        Each example must contain ``input_ids``, ``attention_mask`` and
        ``labels`` (``-100`` on prompt positions, real token ids on refusal
        response positions).
    benign_dataset :
        Dataset of benign instruction-following pairs in the same format.
        The ``L_benign`` term is skipped when this is ``None``.
    args : DeepRefusalConfig or TrainingArguments, optional
        Training configuration.  Pass :class:`DeepRefusalConfig` to control
        DeepRefusal-specific hyperparameters; plain ``TrainingArguments``
        works too (paper defaults are applied).

    Notes
    -----
    * If ``peft`` is not installed and ``lora_r > 0``, the trainer logs a
      warning and falls back to full-parameter fine-tuning without LoRA.
    * ``merge_after_training=True`` calls ``model.merge_and_unload()`` after
      training; the resulting ``self.model`` is the merged base model.
    * The ``train_dataset`` forwarded to ``Trainer.__init__`` is used only for
      step counting and the DataLoader; the actual loss batches are drawn from
      ``harmful_dataset`` / ``benign_dataset`` iterators inside
      ``compute_loss``.  When neither is provided the Trainer's normal
      ``inputs`` (from ``train_dataset``) are used for both terms.
    """

    def __init__(
        self,
        *args: Any,
        refusal_direction: Any,
        harmful_dataset: Any = None,
        benign_dataset: Any = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for DeepRefusalTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "torch is required for DeepRefusalTrainer"
            ) from _TORCH_IMPORT_ERROR

        # ------------------------------------------------------------------
        # Resolve DeepRefusal hyperparameters from the TrainingArguments.
        # ------------------------------------------------------------------
        args_obj = kwargs.get("args", None)
        if args_obj is None and len(args) >= 2:
            args_obj = args[1]

        self._ablation_prob: float = float(getattr(args_obj, "ablation_prob", 0.5))
        self._alpha: float = float(getattr(args_obj, "alpha", 0.2))
        self._lora_r: int = int(getattr(args_obj, "lora_r", 16))
        self._lora_alpha: int = int(getattr(args_obj, "lora_alpha", 32))
        self._lora_dropout: float = float(getattr(args_obj, "lora_dropout", 0.05))
        self._target_modules: Optional[List[str]] = getattr(args_obj, "target_modules", None)
        self._merge_after_training: bool = bool(getattr(args_obj, "merge_after_training", True))

        # ------------------------------------------------------------------
        # Normalise the refusal direction to a unit vector.
        # ------------------------------------------------------------------
        if not isinstance(refusal_direction, torch.Tensor):
            refusal_direction = torch.tensor(refusal_direction, dtype=torch.float32)
        if refusal_direction.dim() != 1:
            raise ValueError(
                "DeepRefusalTrainer: `refusal_direction` must be 1-D "
                f"(hidden_size,), got shape {tuple(refusal_direction.shape)}."
            )
        norm = refusal_direction.float().norm()
        if norm < 1e-12:
            raise ValueError("DeepRefusalTrainer: `refusal_direction` is a zero vector.")
        # Keep on CPU; hooks will move to the device on first use.
        self._d_norm: torch.Tensor = (refusal_direction.float() / norm).detach().cpu()

        # ------------------------------------------------------------------
        # Store datasets; populate train_dataset for the Trainer.
        # ------------------------------------------------------------------
        self._harmful_dataset = harmful_dataset
        self._benign_dataset = benign_dataset

        if "train_dataset" not in kwargs or kwargs["train_dataset"] is None:
            if harmful_dataset is not None:
                kwargs["train_dataset"] = harmful_dataset
            elif benign_dataset is not None:
                kwargs["train_dataset"] = benign_dataset

        # ------------------------------------------------------------------
        # Apply LoRA wrapping before super().__init__ so the Trainer sees the
        # peft model from the start and sets up the optimizer correctly.
        # ------------------------------------------------------------------
        args_list = list(args)
        model = kwargs.get("model", None)
        if model is None and args_list:
            model = args_list[0]

        if model is not None and self._lora_r > 0:
            model = self._wrap_with_lora(model)
            if "model" in kwargs:
                kwargs["model"] = model
            elif args_list:
                args_list[0] = model

        # ------------------------------------------------------------------
        # Initialise the base Trainer.
        # ------------------------------------------------------------------
        super().__init__(*args_list, **kwargs)

        # ------------------------------------------------------------------
        # Install ablation forward hooks on decoder layers.
        # ------------------------------------------------------------------
        self._hooks: List[_AblationHook] = []
        self._hook_handles: List[Any] = []
        self._install_hooks()

        # ------------------------------------------------------------------
        # Build infinitely cycling iterators for the two datasets.
        # ------------------------------------------------------------------
        bs = int(getattr(self.args, "per_device_train_batch_size", 1))
        self._harmful_iter: Optional[Iterator] = (
            _make_dataloader(harmful_dataset, bs) if harmful_dataset is not None else None
        )
        self._benign_iter: Optional[Iterator] = (
            _make_dataloader(benign_dataset, bs) if benign_dataset is not None else None
        )

    # ------------------------------------------------------------------
    # LoRA wrapping
    # ------------------------------------------------------------------

    def _wrap_with_lora(self, model: Any) -> Any:
        """Apply LoRA adapters to *model* and return the peft model.

        Falls back to the raw model (full fine-tuning) and logs a warning when
        ``peft`` is not installed.
        """
        if _PEFT_IMPORT_ERROR is not None:
            logger.warning(
                "DeepRefusalTrainer: peft is not installed (%s). "
                "Falling back to full-parameter fine-tuning without LoRA. "
                "Install peft (`pip install peft`) for the faithful algorithm.",
                _PEFT_IMPORT_ERROR,
            )
            return model

        lora_cfg = LoraConfig(
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            target_modules=self._target_modules,  # None → peft auto-selects
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        try:
            peft_model = get_peft_model(model, lora_cfg)
            peft_model.print_trainable_parameters()
            return peft_model
        except Exception as exc:
            logger.warning(
                "DeepRefusalTrainer: get_peft_model failed (%s). "
                "Falling back to full-parameter fine-tuning without LoRA.",
                exc,
            )
            return model

    # ------------------------------------------------------------------
    # Ablation hooks
    # ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        """Register a post-forward ablation hook on every decoder layer."""
        layers = _get_decoder_layers(self.model)
        if not layers:
            logger.warning(
                "DeepRefusalTrainer: could not locate decoder layers "
                "(expected model.model.language_model.layers [Gemma-3], "
                "model.model.layers [LLaMA/Mistral/Qwen/Gemma], or "
                "model.transformer.h [GPT-style]). "
                "Ablation hooks will not be installed; the trainer will "
                "still run but behaves like plain SFT (no ablation)."
            )
            return

        for layer in layers:
            hook_obj = _AblationHook(self._d_norm, self._ablation_prob)
            handle = layer.register_forward_hook(hook_obj)
            self._hooks.append(hook_obj)
            self._hook_handles.append(handle)

        logger.info(
            "DeepRefusalTrainer: installed ablation hooks on %d decoder layers "
            "(ablation_prob=%.2f).",
            len(self._hooks),
            self._ablation_prob,
        )

    def _select_active_layers(self) -> None:
        """Randomly select a subset of decoder layers to ablate this step.

        Samples ``k ~ Uniform{1, num_layers}`` distinct layer indices without
        replacement (paper Section 3.2).  Only the selected hooks have their
        ``active`` flag set; all others are deactivated.
        """
        n = len(self._hooks)
        if n == 0:
            return
        k = random.randint(1, n)
        selected = set(random.sample(range(n), k))
        for i, hook in enumerate(self._hooks):
            hook.active = (i in selected)

    def _deactivate_all_hooks(self) -> None:
        """Set all ablation hooks to inactive (no-op forward pass)."""
        for hook in self._hooks:
            hook.active = False

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _forward_ce(self, model: Any, batch: Any) -> Any:
        """Run *model* on *batch* and return the cross-entropy loss scalar.

        Passes ``input_ids``, ``attention_mask``, ``labels``, and
        ``position_ids`` (when present) to the model, then:

        * Returns ``outputs.loss`` when the model computed it (HF default when
          ``labels`` are provided).
        * Falls back to :func:`_ce_loss_from_logits` when ``outputs.loss`` is
          ``None``.
        """
        try:
            device = self.model.device
        except AttributeError:
            device = next(self.model.parameters()).device

        batch = _prepare_batch(batch, device)
        valid_keys = {"input_ids", "attention_mask", "labels", "position_ids"}
        filtered = {k: v for k, v in batch.items() if k in valid_keys}

        outputs = model(**filtered)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if hasattr(outputs, "logits") and "labels" in filtered:
            return _ce_loss_from_logits(outputs.logits, filtered["labels"])
        raise RuntimeError(
            "DeepRefusalTrainer._forward_ce: model output has neither "
            ".loss nor .logits; cannot compute cross-entropy."
        )

    # ------------------------------------------------------------------
    # compute_loss (core of the DeepRefusal training loop)
    # ------------------------------------------------------------------

    def compute_loss(  # type: ignore[override]
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        """Compute the DeepRefusal two-part loss (Eq. 1 of arXiv:2509.15202).

        Steps per training step:

        1. Sample a random subset of decoder layers to activate for ablation.
        2. Compute ``L_harmful`` on a batch from ``harmful_dataset`` with
           ablation hooks active (stochastic refusal-direction erasure).
        3. Deactivate all hooks; compute ``L_benign`` on a batch from
           ``benign_dataset`` with no ablation (benign data is unperturbed).
        4. Return ``alpha * L_harmful + (1 - alpha) * L_benign``.

        The Trainer-provided ``inputs`` (from the ``train_dataset`` DataLoader)
        serve as fallbacks when a dedicated dataset iterator is not available.
        """
        # ------------------------------------------------------------------
        # 1. Select layers to ablate this step.
        # ------------------------------------------------------------------
        self._select_active_layers()

        # ------------------------------------------------------------------
        # 2. Harmful loss — hooks active.
        # ------------------------------------------------------------------
        loss_harmful: Optional[Any] = None
        if self._harmful_iter is not None:
            harmful_batch = _next_batch(self._harmful_iter)
            if harmful_batch is not None:
                try:
                    loss_harmful = self._forward_ce(model, harmful_batch)
                except Exception as exc:
                    logger.warning(
                        "DeepRefusalTrainer: L_harmful computation failed "
                        "(%s); skipping this term for the current step.",
                        exc,
                    )

        if loss_harmful is None and inputs is not None:
            # Fallback: no dedicated harmful iterator — use the DataLoader batch.
            try:
                loss_harmful = self._forward_ce(model, inputs)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 3. Benign loss — hooks deactivated (benign data is never ablated).
        # ------------------------------------------------------------------
        self._deactivate_all_hooks()

        loss_benign: Optional[Any] = None
        if self._benign_iter is not None:
            benign_batch = _next_batch(self._benign_iter)
            if benign_batch is not None:
                try:
                    loss_benign = self._forward_ce(model, benign_batch)
                except Exception as exc:
                    logger.warning(
                        "DeepRefusalTrainer: L_benign computation failed "
                        "(%s); skipping this term for the current step.",
                        exc,
                    )

        if loss_benign is None and self._benign_iter is None:
            # No dedicated benign iterator was configured.  How we recover the
            # retention term depends on what the DataLoader ``inputs`` actually
            # contain:
            if self._harmful_iter is None:
                # Neither dataset was dedicated, so ``train_dataset`` (and hence
                # ``inputs``) is the caller's generic batch — NOT the ablated
                # harmful batch.  Using it for the benign retention term is the
                # documented ``inputs``-for-both-terms fallback.
                if inputs is not None:
                    try:
                        loss_benign = self._forward_ce(model, inputs)
                    except Exception:
                        pass
            else:
                # Only a harmful dataset was supplied, so ``inputs`` IS the
                # harmful batch.  Computing "benign" retention on harmful data
                # would defeat the objective (0.8·L_harm(unablated) pulling the
                # model toward the harmful distribution), so we skip the term.
                if not getattr(self, "_warned_no_benign", False):
                    logger.warning(
                        "DeepRefusalTrainer: no benign_dataset supplied; the "
                        "benign retention term is skipped (only the ablated "
                        "harmful loss is optimized). Provide benign_dataset for "
                        "the full DeepRefusal objective."
                    )
                    self._warned_no_benign = True

        # Ensure hooks are always off after the step (safety guard).
        self._deactivate_all_hooks()

        # ------------------------------------------------------------------
        # 4. Combine: L = alpha * L_harmful + (1 - alpha) * L_benign
        # ------------------------------------------------------------------
        alpha = self._alpha
        if loss_harmful is not None and loss_benign is not None:
            loss = alpha * loss_harmful + (1.0 - alpha) * loss_benign
        elif loss_harmful is not None:
            loss = alpha * loss_harmful
        elif loss_benign is not None:
            loss = (1.0 - alpha) * loss_benign
        else:
            # No data at all; return a differentiable zero so autograd stays healthy.
            dummy = next(iter(model.parameters()))
            loss = dummy.sum() * 0.0

        if return_outputs:
            return loss, None
        return loss

    # ------------------------------------------------------------------
    # Post-training LoRA merge
    # ------------------------------------------------------------------

    def _maybe_merge_lora(self) -> None:
        """Merge LoRA adapter weights into the base model if configured.

        Calls ``model.merge_and_unload()`` (standard peft API).  After a
        successful merge ``self.model`` is a plain ``nn.Module`` with no
        LoRA layers.
        """
        if not self._merge_after_training:
            return
        if _PEFT_IMPORT_ERROR is not None:
            return  # peft was never used; nothing to merge

        merge_fn = getattr(self.model, "merge_and_unload", None)
        if merge_fn is None:
            logger.warning(
                "DeepRefusalTrainer: model has no merge_and_unload() method; "
                "skipping LoRA merge (model may not be a peft model)."
            )
            return
        try:
            merged = merge_fn()
            self.model = merged
            logger.info("DeepRefusalTrainer: LoRA adapter merged into base model.")
        except Exception as exc:
            logger.warning(
                "DeepRefusalTrainer: merge_and_unload() failed (%s); "
                "LoRA adapter remains separate.",
                exc,
            )

    def train(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Run training and optionally merge the LoRA adapter afterward."""
        result = super().train(*args, **kwargs)
        self._maybe_merge_lora()
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Remove all forward hooks when the trainer is garbage-collected."""
        for handle in getattr(self, "_hook_handles", []):
            try:
                handle.remove()
            except Exception:
                pass


__all__ = ["DeepRefusalConfig", "DeepRefusalTrainer"]
