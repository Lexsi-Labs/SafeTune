"""
RepNoise Trainer — faithful standalone implementation.

Implements "Representation Noising: A Defence Mechanism Against Harmful
Finetuning" (Rosati et al., arXiv:2405.14577, NeurIPS 2024).  This module is a
faithful port of the reference objective ``rep_noise_loss`` from the official
repository ``domenicrosati/representation-noising`` (``representation_noising/
loss.py``).

RepNoise protects a model during alignment by training with a three-part
compound loss that destroys recoverable harmful representations while
preserving benign utility.  Using the symbol names of the reference repo
(``beta`` for the noise weight, ``alpha`` for the ascent weight):

    L = L_retain  +  beta * L_noise  -  alpha * log(L_harmful)

  1. **Retain loss** (``L_retain``): standard (masked) token cross-entropy on a
     benign / harmless batch X_b.  Preserves general capability.

  2. **Noise loss** (``L_noise``): a *layer-wise distributional* multi-kernel
     Gaussian Maximum Mean Discrepancy (MMD) between the harmful-prompt hidden
     representations and sampled Gaussian noise, summed over **all** hidden
     layers and averaged.  This pushes the per-layer distribution of harmful
     activations toward isotropic Gaussian noise rather than matching a single
     fixed vector.  The MMD uses a sum of 5 Gaussian kernels with bandwidths
     spaced by ``kernel_mul=2.0`` around the median pairwise L2 distance
     (``kernel_num=5``), exactly as in the reference ``MMD_loss`` /
     ``guassian_kernel``.

  3. **Ascent loss** (``-alpha * log(L_harmful)``): *gradient ascent* on the
     harmful cross-entropy.  ``L_harmful`` is the (masked) harmful-token CE,
     accumulated with the CE obtained by re-projecting every harmful hidden
     state through the model's final norm and output embeddings, then
     normalised.  Because the term is ``-alpha * log(L_harmful)``, minimising L
     *increases* the harmful CE — the model is driven away from predicting
     harmful continuations.  This replaces the older max-entropy formulation.

Reference loss (``representation_noising/loss.py``)::

    loss = harmless_losses + beta * noise_loss - alpha * torch.log(harmful_losses)
    # beta = 0.001 (noise/MMD weight), alpha = 1 (ascent weight)

Masking: the reference computes a token mask of positions where the paired
harmful and harmless ``input_ids`` differ (``~torch.eq(harmful, harmless)``) and
applies it to both the activations (for MMD) and the CE losses.  When a paired
harmless batch is not available, this implementation falls back to the harmful
``attention_mask`` so the term remains well defined.

Public API mapping (kept stable; field names unchanged):

  * ``repnoise_beta1``  -> ``alpha``  (ascent weight on harmful CE)
  * ``repnoise_beta2``  -> retain (benign CE) weight
  * ``repnoise_beta3``  -> ``beta``   (noise/MMD weight)
  * ``repnoise_noise_seed`` -> seed for the sampled Gaussian noise (MMD target)

Usage::

    from safetune.harden.repnoise import RepNoiseTrainer, RepNoiseConfig

    config = RepNoiseConfig(
        output_dir="./repnoise_output",
        repnoise_beta1=1.0,     # alpha (ascent)
        repnoise_beta2=1.0,     # retain CE weight
        repnoise_beta3=0.001,   # beta  (MMD noise)
        repnoise_noise_seed=42,
    )
    trainer = RepNoiseTrainer(
        model=model,
        args=config,
        train_dataset=benign_dataset,
        harmful_dataset=harmful_dataset,     # iterable; cycled per step
        benign_dataset=benign_dataset,       # iterable; cycled per step
        tokenizer=tokenizer,
    )
    trainer.train()

If ``harmful_dataset`` is ``None``, the noise and ascent losses are both
skipped and the trainer falls back to plain benign SFT (graceful degradation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import Trainer, TrainingArguments
    _TRAINER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = object  # type: ignore[assignment,misc]
    _TRAINER_IMPORT_ERROR = _e

try:
    import torch
    import torch.nn.functional as F
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e


if _TRAINER_IMPORT_ERROR is None:
    @dataclass
    class RepNoiseConfig(TrainingArguments):  # type: ignore[misc]
        """TrainingArguments subclass exposing the RepNoise hyper-parameters.

        All fields have defaults so the public construction signature is
        unchanged from :class:`transformers.TrainingArguments`.  Field names are
        kept stable for backward compatibility; their semantics are documented
        below and map onto the reference ``rep_noise_loss`` coefficients.

        Attributes:
            repnoise_beta1: ``alpha`` — weight of the gradient-*ascent* term on
                harmful cross-entropy, ``-alpha * log(L_harmful)``.  Reference
                default: 1.0.
            repnoise_beta2: weight of the retain (benign cross-entropy) loss
                ``L_retain``.  Reference default: 1.0.
            repnoise_beta3: ``beta`` — weight of the layer-wise MMD noise loss
                ``L_noise``.  Reference default: 0.001.
            repnoise_noise_seed: Seed used to make the sampled Gaussian noise
                (the MMD target) reproducible.  Default: 42.
        """

        repnoise_beta1: float = 1.0    # alpha (ascent weight)
        repnoise_beta2: float = 1.0    # retain CE weight
        repnoise_beta3: float = 0.001  # beta  (MMD noise weight)
        repnoise_noise_seed: int = 42
else:  # pragma: no cover
    class RepNoiseConfig(object):  # type: ignore[assignment]
        pass


if _TORCH_IMPORT_ERROR is None:

    class MMD_loss(torch.nn.Module):  # type: ignore[misc]
        """Multi-kernel (multi-bandwidth) Gaussian Maximum Mean Discrepancy.

        Faithful port of ``MMD_loss`` from the official RepNoise repo
        (``representation_noising/loss.py``).  Uses a sum of ``kernel_num``
        Gaussian kernels whose bandwidths are geometrically spaced by
        ``kernel_mul`` around the median pairwise squared-L2 distance.
        """

        def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5) -> None:
            super().__init__()
            self.kernel_num = kernel_num
            self.kernel_mul = kernel_mul
            self.fix_sigma = None

        def guassian_kernel(
            self,
            source: "torch.Tensor",
            target: "torch.Tensor",
            kernel_mul: float = 2.0,
            kernel_num: int = 5,
            fix_sigma: Optional[float] = None,
        ) -> "torch.Tensor":
            n_samples = int(source.size()[0]) + int(target.size()[0])
            total = torch.cat([source, target], dim=0)

            total0 = total.unsqueeze(0).expand(
                int(total.size(0)), int(total.size(0)), int(total.size(1))
            )
            total1 = total.unsqueeze(1).expand(
                int(total.size(0)), int(total.size(0)), int(total.size(1))
            )
            L2_distance = ((total0 - total1) ** 2).sum(2)
            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
            bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
            bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
            kernel_val = [
                torch.exp(-L2_distance / bandwidth_temp)
                for bandwidth_temp in bandwidth_list
            ]
            return sum(kernel_val)

        def forward(
            self,
            source: "torch.Tensor",
            target: "torch.Tensor",
            xy_only: bool = False,
        ) -> "torch.Tensor":
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source,
                target,
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma,
            )
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            loss = torch.mean(XX + YY - XY - YX)
            return loss

    def masked_token_ce_loss(
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Masked next-token cross-entropy (port of the reference helper).

        Shifts logits/labels for causal LM, zeros out masked positions, maps the
        masked label positions to ``-100`` (ignored), and computes CE.
        """
        device = logits.device
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logit_mask = mask[..., :-1].contiguous()
        expanded_mask = shift_logit_mask.unsqueeze(-1).expand(
            -1, -1, shift_logits.size(-1)
        )
        shift_label_mask = mask[..., 1:].contiguous()
        shift_logits = shift_logits * expanded_mask
        shift_labels = shift_labels * shift_label_mask
        shift_labels = shift_labels.clone()
        shift_labels[shift_labels == 0] = -100
        shift_labels = shift_labels.long()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)).to(device),
            shift_labels.view(-1).to(device),
        )
        return loss

else:  # pragma: no cover
    MMD_loss = object  # type: ignore[assignment,misc]

    def masked_token_ce_loss(*args: Any, **kwargs: Any):  # type: ignore[misc]
        raise ImportError("torch is required for masked_token_ce_loss")


class RepNoiseTrainer(Trainer if _TRAINER_IMPORT_ERROR is None else object):  # type: ignore[misc]
    """HuggingFace :class:`~transformers.Trainer` with the RepNoise compound loss.

    Faithful port of ``rep_noise_loss`` (Rosati et al. arXiv:2405.14577):

        L = L_retain + beta * L_noise - alpha * log(L_harmful)

    * ``L_retain``: masked benign cross-entropy (utility preservation).
    * ``L_noise``: layer-wise multi-kernel Gaussian MMD between harmful hidden
      states (all layers) and sampled Gaussian noise.
    * ``-alpha * log(L_harmful)``: gradient ascent on harmful cross-entropy.

    Args:
        harmful_dataset: An iterable of batches of harmful data (``X_h``).
            Cycled indefinitely; one batch per step drives the MMD-noise and
            harmful-CE-ascent terms.  When ``None``, both are skipped (graceful
            degradation to benign SFT).
        benign_dataset: An iterable of batches of benign / harmless data
            (``X_b``).  Cycled indefinitely; one batch per step provides
            ``L_retain`` and (when shaped-compatibly) the harmful/benign token
            *difference* mask used by the reference.  When ``None``, ``inputs``
            (the standard trainer batch) is used as the benign batch.
        **kwargs: Forwarded verbatim to :class:`transformers.Trainer`.
    """

    def __init__(
        self,
        *args: Any,
        harmful_dataset: Optional[Iterable] = None,
        benign_dataset: Optional[Iterable] = None,
        **kwargs: Any,
    ) -> None:
        if _TRAINER_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for RepNoiseTrainer"
            ) from _TRAINER_IMPORT_ERROR
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "torch is required for RepNoiseTrainer"
            ) from _TORCH_IMPORT_ERROR

        super().__init__(*args, **kwargs)

        # repnoise_beta1 -> alpha (ascent), repnoise_beta3 -> beta (MMD noise),
        # repnoise_beta2 -> retain CE weight.  See module docstring.
        self._alpha = float(getattr(self.args, "repnoise_beta1", 1.0))
        self._retain_weight = float(getattr(self.args, "repnoise_beta2", 1.0))
        self._beta = float(getattr(self.args, "repnoise_beta3", 0.001))
        self._noise_seed = int(getattr(self.args, "repnoise_noise_seed", 42))

        self._mmd = MMD_loss()

        # Iterators over the harmful and benign datasets (cycled per step).
        self._harmful_dataset = harmful_dataset
        self._harmful_iter: Optional[Iterator] = (
            cycle(iter(harmful_dataset)) if harmful_dataset is not None else None
        )

        self._benign_dataset = benign_dataset
        self._benign_iter: Optional[Iterator] = (
            cycle(iter(benign_dataset)) if benign_dataset is not None else None
        )

    # ------------------------------------------------------------------
    # Dataset iteration helpers
    # ------------------------------------------------------------------

    def _next_batch(
        self,
        dataset: Optional[Iterable],
        iterator: Optional[Iterator],
        name: str,
    ) -> Optional[Any]:
        """Advance ``iterator`` and return the next batch, cycling as needed."""
        if iterator is None:
            return None
        try:
            return next(iterator)
        except StopIteration:
            if dataset is None:
                return None
            new_iter = cycle(iter(dataset))
            setattr(self, f"_{name}_iter", new_iter)
            return next(new_iter)

    def _next_harmful_batch(self) -> Optional[Any]:
        return self._next_batch(self._harmful_dataset, self._harmful_iter, "harmful")

    def _next_benign_batch(self) -> Optional[Any]:
        return self._next_batch(self._benign_dataset, self._benign_iter, "benign")

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch(self, batch: Any, model: Any) -> dict:
        """Move a batch dict to the model's device, applying _prepare_inputs."""
        if hasattr(self, "_prepare_inputs"):
            try:
                batch = self._prepare_inputs(batch)
            except Exception:
                pass
        if isinstance(batch, dict):
            device = next(model.parameters()).device
            batch = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
        return batch

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------

    @staticmethod
    def _diff_mask(harmful_batch: dict, benign_batch: Optional[dict]) -> "torch.Tensor":
        """Token mask of positions where harmful and benign input_ids differ.

        Faithful to the reference (``~torch.eq(harmful, harmless)``).  Requires
        the two batches to share shape; otherwise (or when no benign batch is
        available) falls back to the harmful ``attention_mask`` (or all-ones).
        """
        h_ids = harmful_batch["input_ids"]
        b_ids = benign_batch.get("input_ids") if isinstance(benign_batch, dict) else None
        if b_ids is not None and tuple(b_ids.shape) == tuple(h_ids.shape):
            return (~torch.eq(h_ids, b_ids.to(h_ids.device)))
        am = harmful_batch.get("attention_mask", None)
        if am is not None:
            return am.bool()
        return torch.ones_like(h_ids, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Noise (MMD) loss over all hidden layers
    # ------------------------------------------------------------------

    def _noise_loss(
        self,
        harmful_hidden_states,
        mask: "torch.Tensor",
        device: Any,
    ) -> "torch.Tensor":
        """L_noise: layer-wise multi-kernel Gaussian MMD to Gaussian noise.

        For every hidden layer, mask the harmful activations, sample matching
        Gaussian noise (masked identically), flatten each per-example, and
        accumulate the MMD between the two distributions.  Averaged over layers.
        Faithful to the reference noise-loop in ``rep_noise_loss``.
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._noise_seed)

        noise_loss = torch.zeros((), device=device)
        n_layers = len(harmful_hidden_states)
        for hidden in harmful_hidden_states:
            hidden = hidden.to(device)
            hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
            hiddens = hidden * hiddens_mask
            # Reproducible Gaussian noise (sampled on CPU, moved to device) so
            # the MMD target distribution is deterministic across runs.
            gaussian = torch.randn(
                hidden.shape, generator=gen, dtype=hidden.dtype
            ).to(hidden.device)
            gaussian = gaussian * hiddens_mask
            noise_loss = noise_loss + self._mmd(
                hiddens.view(hiddens.size(0), -1),
                gaussian.view(gaussian.size(0), -1),
            ).to(device)
        noise_loss = noise_loss / max(n_layers, 1)
        return noise_loss

    # ------------------------------------------------------------------
    # Harmful CE (for ascent term)
    # ------------------------------------------------------------------

    def _harmful_ce(
        self,
        model: Any,
        harmful_outputs: Any,
        harmful_input_ids: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """L_harmful: masked harmful-token CE accumulated over hidden states.

        Faithful to the reference: start from the masked CE on the final logits,
        then add the CE obtained by re-projecting every hidden state through the
        model's final norm and output embeddings, and normalise by the number
        of hidden states.
        """
        device = harmful_outputs.logits.device
        harmful_losses = masked_token_ce_loss(
            harmful_outputs.logits, harmful_input_ids, mask
        )

        hidden_states = getattr(harmful_outputs, "hidden_states", None)
        output_embeddings = self._get_output_embeddings(model)
        norm = self._get_final_norm(model)

        if hidden_states is not None and output_embeddings is not None and norm is not None:
            for h in hidden_states:
                out = output_embeddings(norm(h.to(device)))
                harmful_losses = harmful_losses + masked_token_ce_loss(
                    out.to(device), harmful_input_ids.to(device), mask
                )
            harmful_losses = harmful_losses / len(hidden_states)
        return harmful_losses

    @staticmethod
    def _get_output_embeddings(model: Any):
        if hasattr(model, "get_output_embeddings"):
            try:
                emb = model.get_output_embeddings()
                if emb is not None:
                    return emb
            except Exception:
                pass
        for attr in ("lm_head",):
            if hasattr(model, attr):
                return getattr(model, attr)
        return None

    @staticmethod
    def _get_final_norm(model: Any):
        # Common locations: model.model.norm (Qwen/Llama), model.base_model.norm.
        for base_attr in ("model", "base_model"):
            base = getattr(model, base_attr, None)
            if base is not None and hasattr(base, "norm"):
                return base.norm
        if hasattr(model, "norm"):
            return model.norm
        return None

    # ------------------------------------------------------------------
    # Trainer hook
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        """Compute the RepNoise compound loss.

        Faithful three-term objective (reference ``rep_noise_loss``)::

            L = L_retain + beta * L_noise - alpha * log(L_harmful)

        When ``harmful_dataset`` is ``None`` the harmful terms (noise + ascent)
        are skipped and the loss reduces to the retain CE (graceful degradation
        to plain benign SFT).
        """
        device = next(model.parameters()).device

        # ------------------------------------------------------------------
        # Benign / retain batch (also supplies the diff mask reference)
        # ------------------------------------------------------------------
        benign_batch = self._next_benign_batch()
        if benign_batch is not None:
            benign_batch = self._prepare_batch(benign_batch, model)
            retain_inputs = benign_batch
        else:
            retain_inputs = inputs

        # ------------------------------------------------------------------
        # Harmful terms (L_noise + ascent on harmful CE)
        # ------------------------------------------------------------------
        harmful_batch = self._next_harmful_batch()

        if harmful_batch is not None:
            harmful_batch = self._prepare_batch(harmful_batch, model)

            mask = self._diff_mask(harmful_batch, retain_inputs).to(device)

            harmful_outputs = model(
                input_ids=harmful_batch["input_ids"],
                attention_mask=harmful_batch.get("attention_mask"),
                output_hidden_states=True,
            )

            # L_noise: layer-wise multi-kernel Gaussian MMD to Gaussian noise.
            hidden_states = getattr(harmful_outputs, "hidden_states", None)
            if hidden_states is not None and len(hidden_states) > 0:
                l_noise = self._noise_loss(hidden_states, mask, device)
            else:  # pragma: no cover - model without hidden states
                l_noise = torch.zeros((), device=device)

            # L_harmful (CE) and the ascent term -alpha * log(L_harmful).
            mask_f = mask.float().to(device)
            l_harmful = self._harmful_ce(
                model, harmful_outputs, harmful_batch["input_ids"], mask_f
            )
            # Clamp inside log for numerical stability (does not change the
            # reference behaviour for positive CE values).
            l_ascent = -self._alpha * torch.log(l_harmful.clamp(min=1e-8))
        else:
            l_noise = torch.zeros((), device=device)
            l_harmful = torch.zeros((), device=device)
            l_ascent = torch.zeros((), device=device)

        # ------------------------------------------------------------------
        # Retain (benign cross-entropy) term
        # ------------------------------------------------------------------
        retain_outputs = model(**retain_inputs)
        l_retain = retain_outputs.loss

        # ------------------------------------------------------------------
        # Combined loss:  L_retain + beta * L_noise - alpha * log(L_harmful)
        # ------------------------------------------------------------------
        loss = (
            self._retain_weight * l_retain
            + self._beta * l_noise
            + l_ascent
        )

        if return_outputs:
            return (loss, retain_outputs)
        return loss


__all__ = ["RepNoiseConfig", "RepNoiseTrainer", "MMD_loss", "masked_token_ce_loss"]
