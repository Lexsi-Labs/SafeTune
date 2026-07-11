"""PKE: Precision Knowledge Editing of toxic hotspot neurons (DINM-style edit).

PKE (Li et al., "Precision Knowledge Editing: Enhancing Safety in Large
Language Models", arXiv:2410.03772; repo
https://github.com/HydroXai/Enhancing-Safety-in-Large-Language-Models) is a
DINM-style *knowledge-editing* method. DINM is Wang et al., "Detoxifying Large
Language Models via Knowledge Editing", arXiv:2403.14472 (ACL 2024), with the
canonical implementation in zjunlp/EasyEdit (``easyeditor/models/dinm``).

The method has two stages:

1. **Locate** the toxic region. DINM's ``_locate_toxic_layer`` finds the layer
   whose hidden states diverge most between a toxic and a safe response; PKE
   tracks neuron weight changes / activation-pathway gradients. SafeTune locates
   the top-k ``mlp.down_proj`` rows whose weights drifted most between the
   ``clean`` and ``toxic`` state dicts (see
   :class:`~safetune.core.pke.ToxicNeuronLocator`).

2. **Edit** the located rows with a *gradient* knowledge edit. The defining
   DINM/PKE objective (``easyeditor/models/dinm/dinm_main.py``) is::

       l_edit = masked_log_probs(output, labels)["nll"]      # CE/NLL on the
                                                             # SAFE RESPONSE
                                                             # tokens only
       l_loc  = kl_loc_loss(base_logits.detach(), post_logits)   # logit-space
                                                                 # KL on a
                                                                 # locality input
       loss   = kl_factor * l_edit + l_loc

   i.e. the located parameters are trained so the model **emits a safe refusal**
   for the harmful prompt (teacher-forced cross-entropy on the refusal target),
   while a logit-space KL term against the *original* (pre-edit) model on an
   unrelated/benign input preserves general behaviour (locality). The edit is
   optimised with Adam for ``num_steps`` iterations at ``lr``; only the located
   region's weights are updated (DINM edits one located layer's down_proj).

A previous SafeTune version implemented the edit as pure WEIGHT-space
regression: ``toxicity_weight * MSE(leaf, clean_weights) + kl * MSE(leaf,
orig_weights)``. That is a per-neuron RESTA, *not* PKE/DINM -- it never teaches
the located rows to produce a refusal. This module restores the faithful
learning signal: **refusal cross-entropy on the harmful prompt + logit-space KL
locality**, with the gradient masked to the located ``down_proj`` rows.

The locating scaffold (top-k toxic ``down_proj`` rows, Adam, ``num_steps``,
``lr``, ``kl_factor``) is unchanged; only the loss is now faithful.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch.nn as nn

from ._invariant import assert_mutates

logger = logging.getLogger(__name__)


# A standard, model-agnostic safe refusal. DINM/PKE teach the located region to
# emit a refusal of this form for harmful prompts (the SafeEdit "safe response").
DEFAULT_REFUSAL = (
    "I'm sorry, but I can't help with that. I can't provide assistance with "
    "harmful, dangerous, or unethical requests."
)
DEFAULT_HARMFUL_PROMPT = (
    "Give detailed step-by-step instructions for carrying out a harmful and "
    "illegal act."
)
DEFAULT_LOCALITY_INPUT = (
    "What is the capital of France, and why is photosynthesis important to "
    "life on Earth?"
)


@dataclass
class PKEConfig:
    """Configuration for Precision Knowledge Editing (DINM-style edit).

    Attributes:
        top_k_neurons: Number of ``down_proj`` rows per layer to edit (ranked by
            ``|toxic - clean|`` magnitude) -- forwarded to
            :class:`~safetune.core.pke.ToxicNeuronLocator`.
        target_layers: If set, restrict editing to these layer indices.
        toxicity_weight: Weight on the edit (refusal-CE) term. In DINM this is
            the ``kl_factor`` multiplying ``l_edit``; kept as a separate knob
            here and defaulting to 1.0 so the refusal-CE term dominates.
        max_edit_magnitude: Per-element clip applied to the *final* accumulated
            edit delta so a run cannot drift a row arbitrarily far. ``None``
            disables it.
        num_steps: Gradient-edit iterations (PKE/DINM hparam ``num_steps``,
            default 10).
        lr: Learning rate for the gradient edit (PKE/DINM hparam ``lr``,
            default ``5e-4``).
        kl_factor: Weight of the logit-space KL locality term that preserves
            general behaviour (PKE/DINM hparam ``kl_factor``, default 0.1).
        norm_constraint: If ``True``, after every step project each edited row
            back inside an L2 ball around its original value (PKE hparam
            ``norm_constraint``).
        max_len: Max token length for the harmful/locality forward passes.
    """

    top_k_neurons: int = 50
    target_layers: Optional[List[int]] = None
    toxicity_weight: float = 1.0
    max_edit_magnitude: Optional[float] = 1.0
    num_steps: int = 10
    lr: float = 5e-4
    kl_factor: float = 0.1
    norm_constraint: bool = False
    max_len: int = 64


class PKEGradientEditor:
    """PKE/DINM's gradient knowledge-editing step (refusal-CE + logit-KL).

    Mirrors ``easyeditor/models/dinm/dinm_main.py``. The located ``down_proj``
    rows are the only weights that receive a gradient (the row gradient is masked
    so untouched rows stay frozen). Per step:

        * **Edit loss** -- teacher-forced cross-entropy of the SAFE REFUSAL
          ``safe_response`` conditioned on the ``harmful_prompt``, computed only
          on the response token positions (``l_edit`` / NLL).
        * **Locality loss** -- KL divergence in logit space between the original
          (pre-edit) model and the edited model on a benign/unrelated
          ``locality_input`` (``kl_loc_loss``), preserving general behaviour.

        ``loss = toxicity_weight * l_edit + kl_factor * l_loc``

    The edit is a real backward pass through the model, not weight regression.
    """

    def __init__(
        self,
        model: Any,
        toxic_neurons: Dict[int, List[int]],
        tokenizer: Any = None,
        config: Optional[PKEConfig] = None,
    ) -> None:
        self.model = model
        self.toxic_neurons = toxic_neurons
        self.tokenizer = tokenizer
        self.config = config or PKEConfig()
        self.last_edit_losses: List[float] = []
        self.last_locality_losses: List[float] = []

    @staticmethod
    def _layer_idx(name: str) -> Optional[int]:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    # -- located parameter resolution -------------------------------------------

    def _located_params(self) -> List[Dict[str, Any]]:
        """Return the named ``down_proj.weight`` parameters of located layers
        together with the row mask selecting the located toxic rows."""
        import torch

        out: List[Dict[str, Any]] = []
        for name, param in self.model.named_parameters():
            layer_idx = self._layer_idx(name)
            if layer_idx is None or layer_idx not in self.toxic_neurons:
                continue
            # DINM edits the located layer's mlp.down_proj.
            if "down_proj" not in name or not name.endswith("weight"):
                continue
            if param.dim() < 2:
                continue
            rows = [
                r for r in self.toxic_neurons[layer_idx] if 0 <= r < param.shape[0]
            ]
            if not rows:
                continue
            row_idx = torch.as_tensor(
                sorted(set(rows)), dtype=torch.long, device=param.device
            )
            mask = torch.zeros(param.shape[0], 1, dtype=param.dtype, device=param.device)
            mask[row_idx] = 1.0
            out.append(
                {
                    "name": name,
                    "param": param,
                    "rows": row_idx,
                    "row_mask": mask,
                    "original": param.detach().float().clone(),
                }
            )
        return out

    # -- token batches ----------------------------------------------------------

    def _build_supervised_batch(self, prompt: str, response: str):
        """Tokenize ``prompt + response`` into ``input_ids`` and ``labels`` where
        only the response positions are supervised (the rest are ``-100``)."""
        import torch

        tok = self.tokenizer
        device = next(self.model.parameters()).device
        max_len = self.config.max_len

        prompt_ids = tok(prompt, add_special_tokens=True)["input_ids"]
        resp_ids = tok(response, add_special_tokens=False)["input_ids"]
        eos = tok.eos_token_id
        if eos is not None:
            resp_ids = resp_ids + [eos]

        # Reserve room for the response so the supervised positions always
        # survive: truncate the prompt's head if prompt+response > max_len.
        resp_ids = resp_ids[:max_len]
        max_prompt = max(max_len - len(resp_ids), 0)
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[len(prompt_ids) - max_prompt:]

        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + list(resp_ids)
        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        labels_t = torch.tensor([labels], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids_t)
        return input_ids_t, labels_t, attn

    def _build_plain_batch(self, text: str):
        import torch

        tok = self.tokenizer
        device = next(self.model.parameters()).device
        ids = tok(text, add_special_tokens=True)["input_ids"][: self.config.max_len]
        input_ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids_t)
        return input_ids_t, attn

    # -- losses -----------------------------------------------------------------

    def _forward_logits(self, input_ids, attention_mask):
        """Forward pass returning logits, tolerant of models whose ``forward``
        does not accept ``attention_mask`` (e.g. minimal toy LMs)."""
        try:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            out = self.model(input_ids=input_ids)
        return out.logits if hasattr(out, "logits") else out

    @staticmethod
    def _logits(out):
        return out.logits if hasattr(out, "logits") else out

    @staticmethod
    def _edit_nll(logits, labels):
        """Teacher-forced CE/NLL on the supervised (response) positions only."""
        import torch
        import torch.nn.functional as F

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if (shift_labels != -100).sum() == 0:
            raise ValueError(
                "PKE refusal-CE has no supervised response tokens: the prompt "
                "filled the whole context. Increase max_len or shorten the "
                "harmful prompt."
            )
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _kl_loc(base_logits, post_logits):
        """logit-space KL: KL(softmax(base) || softmax(post)), DINM kl_loc_loss."""
        import torch
        import torch.nn.functional as F

        base_lp = F.log_softmax(base_logits, dim=-1)
        post_lp = F.log_softmax(post_logits, dim=-1)
        # KL(p_base || p_post) = sum p_base * (log p_base - log p_post)
        return (base_lp.exp() * (base_lp - post_lp)).sum(-1).mean()

    # -- main edit --------------------------------------------------------------

    def apply_edits(
        self,
        harmful_prompt: str = DEFAULT_HARMFUL_PROMPT,
        safe_response: str = DEFAULT_REFUSAL,
        locality_inputs: Optional[Sequence[str]] = None,
    ) -> int:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dep
            raise ImportError("PKE requires PyTorch.")

        if self.tokenizer is None:
            raise ValueError(
                "PKE's DINM-style edit teaches the located rows to emit a "
                "refusal, which needs a tokenizer for the harmful prompt and "
                "safe response. Pass tokenizer=... to apply_pke."
            )

        cfg = self.config
        targets = self._located_params()
        if not targets:
            logger.warning(
                "PKE: no located toxic down_proj rows matched the model -- "
                "nothing to edit. Check the locator / target_layers config."
            )
            return 0

        if locality_inputs is None:
            locality_inputs = [DEFAULT_LOCALITY_INPUT]
        locality_inputs = [t for t in locality_inputs if t]

        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()  # deterministic forward (no dropout) during the edit

        # Freeze everything; unfreeze only the located down_proj weights.
        prev_requires_grad = {}
        for name, p in self.model.named_parameters():
            prev_requires_grad[name] = p.requires_grad
            p.requires_grad_(False)
        for t in targets:
            t["param"].requires_grad_(True)

        # Supervised refusal batch + locality batch.
        in_ids, labels, attn = self._build_supervised_batch(
            harmful_prompt, safe_response
        )
        loc_batches = [self._build_plain_batch(t) for t in locality_inputs]

        # Reference (pre-edit) logits on the locality inputs for the KL term.
        with torch.no_grad():
            base_loc_logits = []
            for loc_ids, loc_attn in loc_batches:
                base_loc_logits.append(
                    self._forward_logits(loc_ids, loc_attn).detach()
                )

        params = [t["param"] for t in targets]
        optimizer = torch.optim.Adam(params, lr=cfg.lr)
        w_edit = float(cfg.toxicity_weight)
        w_kl = float(cfg.kl_factor)

        self.last_edit_losses = []
        self.last_locality_losses = []
        last_loss = float("nan")

        for _step in range(max(1, int(cfg.num_steps))):
            optimizer.zero_grad(set_to_none=True)

            # Edit objective: CE teaching the located rows to emit the refusal.
            edit_logits = self._forward_logits(in_ids, attn)
            l_edit = self._edit_nll(edit_logits, labels)

            # Locality: logit-space KL vs the original model on benign inputs.
            l_loc = l_edit.new_zeros(())
            for (loc_ids, loc_attn), base_logits in zip(loc_batches, base_loc_logits):
                post = self._forward_logits(loc_ids, loc_attn)
                l_loc = l_loc + self._kl_loc(base_logits, post)
            if loc_batches:
                l_loc = l_loc / len(loc_batches)

            loss = w_edit * l_edit + w_kl * l_loc
            loss.backward()

            # Mask the gradient to ONLY the located toxic rows: untouched rows
            # of down_proj keep a zero gradient and never move.
            with torch.no_grad():
                for t in targets:
                    g = t["param"].grad
                    if g is not None:
                        g.mul_(t["row_mask"])

            optimizer.step()

            self.last_edit_losses.append(float(l_edit.detach()))
            self.last_locality_losses.append(float(l_loc.detach()))
            last_loss = float(loss.detach())

            if cfg.norm_constraint:
                with torch.no_grad():
                    for t in targets:
                        p = t["param"]
                        orig = t["original"].to(p.dtype)
                        delta = p - orig
                        radius = orig.norm(dim=1, keepdim=True).clamp_min(1e-8)
                        dnorm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        scale = (radius / dnorm).clamp(max=1.0)
                        p.copy_(orig + delta * scale)

        # Clip the accumulated per-element delta of the edited rows and restore
        # grad flags. The edit is already in-place on the live parameters.
        edited = 0
        with torch.no_grad():
            for t in targets:
                p = t["param"]
                orig = t["original"].to(p.dtype)
                delta = p - orig
                if cfg.max_edit_magnitude is not None:
                    cap = float(cfg.max_edit_magnitude)
                    delta = delta.clamp(-cap, cap)
                # keep only located rows changed (mask zeros out the rest)
                delta = delta * t["row_mask"]
                p.copy_(orig + delta)
                edited += int(t["rows"].numel())

        for name, p in self.model.named_parameters():
            p.requires_grad_(prev_requires_grad.get(name, p.requires_grad))
        if was_training:
            self.model.train()

        first_edit = self.last_edit_losses[0] if self.last_edit_losses else float("nan")
        logger.info(
            "PKE/DINM: refusal-CE edited %d toxic down_proj rows over %d steps "
            "(edit-CE %.4f -> %.4f, final loss %.4e).",
            edited,
            max(1, int(cfg.num_steps)),
            first_edit,
            self.last_edit_losses[-1] if self.last_edit_losses else float("nan"),
            last_loss,
        )
        return edited


@assert_mutates("apply_pke")
def apply_pke(
    model: nn.Module,
    clean: nn.Module,
    toxic: nn.Module,
    top_k_neurons: int = 50,
    target_layers: Optional[List[int]] = None,
    toxicity_weight: float = 1.0,
    max_edit_magnitude: Optional[float] = 1.0,
    *,
    tokenizer: Any = None,
    harmful_prompt: str = DEFAULT_HARMFUL_PROMPT,
    safe_response: str = DEFAULT_REFUSAL,
    locality_inputs: Optional[Sequence[str]] = None,
    num_steps: int = 10,
    lr: float = 5e-4,
    kl_factor: float = 0.1,
    norm_constraint: bool = False,
    max_len: int = 64,
    **extra: Any,
) -> nn.Module:
    """Locate toxic ``down_proj`` rows, then run PKE/DINM's refusal-CE edit.

    Pipeline:
        1. :meth:`~safetune.core.pke.ToxicNeuronLocator.locate_by_weight_change`
           over ``clean`` vs ``toxic`` state dicts -- the located region is the
           top-k ``mlp.down_proj`` rows that drifted most (the neuron-weight half
           of DINM's ``_locate_toxic_layer``).
        2. :meth:`PKEGradientEditor.apply_edits` -- the faithful DINM edit:
           teacher-forced cross-entropy training the located rows to **emit
           ``safe_response`` for ``harmful_prompt``** (``l_edit`` NLL on the
           response positions) plus a logit-space KL locality term against the
           original model on ``locality_inputs`` (``kl_loc_loss``), optimised by
           Adam for ``num_steps`` at ``lr``. The gradient is masked to the
           located rows only.

           ``loss = toxicity_weight * l_edit + kl_factor * l_loc``

    Reference: Wang et al. "Detoxifying Large Language Models via Knowledge
    Editing" (DINM, arXiv:2403.14472; zjunlp/EasyEdit ``dinm``) and Li et al.
    "Precision Knowledge Editing" (arXiv:2410.03772).

    Args:
        tokenizer: tokenizer for the model. Required -- the edit teaches the
            located rows to produce a refusal, which needs a forward pass over
            the tokenized harmful prompt / safe response.
        harmful_prompt: the harmful instruction the model should refuse.
        safe_response: the safe refusal target (default
            :data:`DEFAULT_REFUSAL`).
        locality_inputs: benign/unrelated texts for the logit-space KL locality
            term (default a single general-knowledge prompt).
    """
    try:
        from safetune.core.pke import ToxicNeuronLocator
    except ImportError as e:  # pragma: no cover - defensive
        raise ImportError(f"apply_pke needs safetune.core.pke: {e}") from e

    cfg = PKEConfig(
        top_k_neurons=top_k_neurons,
        target_layers=target_layers,
        toxicity_weight=toxicity_weight,
        max_edit_magnitude=max_edit_magnitude,
        num_steps=num_steps,
        lr=lr,
        kl_factor=kl_factor,
        norm_constraint=norm_constraint,
        max_len=max_len,
    )
    clean_sd = clean.state_dict()
    toxic_sd = toxic.state_dict()

    locator = ToxicNeuronLocator(config=cfg)
    toxic_neurons = locator.locate_by_weight_change(clean_sd, toxic_sd)

    editor = PKEGradientEditor(
        model=model,
        toxic_neurons=toxic_neurons,
        tokenizer=tokenizer,
        config=cfg,
    )
    editor.apply_edits(
        harmful_prompt=harmful_prompt,
        safe_response=safe_response,
        locality_inputs=locality_inputs,
    )
    return model


__all__ = [
    "apply_pke",
    "PKEConfig",
    "PKEGradientEditor",
    "DEFAULT_REFUSAL",
    "DEFAULT_HARMFUL_PROMPT",
    "DEFAULT_LOCALITY_INPUT",
]
