"""
SafeSwitch: Steering Unsafe LLM Behavior via Internal Activation Signals.
Hanpx20/SafeSwitch

SafeSwitch adds two learned components on top of a base LLM:
1. Safety Prober: a small probe trained on internal activations to predict
   whether a prompt is likely to produce unsafe output.
2. Refusal Head: an auxiliary LM head (or logit-bias layer) that overrides the
   base LM head's token distribution toward refusals when the prober fires.

This module provides:
- SafetyProber: linear probe trained on pooled hidden states.
- RefusalHeadWrapper: wraps a model to add the plug-in refusal head.
- SafeSwitchRunner: end-to-end inference wrapper that activates the refusal head
  when the prober confidence exceeds a threshold.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafeSwitchConfig:
    """Configuration for SafeSwitch."""
    # Layer index from which to extract hidden states for the prober
    probe_layer: int = -1
    # Hidden dimension of the probe layer (must match model's hidden size)
    hidden_size: int = 4096
    # Probability threshold above which the refusal head is activated
    unsafe_threshold: float = 0.7
    # Token IDs to boost when the refusal head is triggered (e.g., refusal start tokens)
    # If empty, we apply a uniform logit penalty to all non-EOS tokens.
    refusal_token_ids: List[int] = None
    # Logit bonus applied to refusal tokens when activated
    refusal_logit_bonus: float = 10.0

    def __post_init__(self):
        if self.refusal_token_ids is None:
            self.refusal_token_ids = []


class SafetyProber:
    """
    A lightweight linear probe that classifies internal activations as safe/unsafe.

    Requires sklearn or a compatible classifier with .fit() and .predict_proba().
    """

    def __init__(self, hidden_size: int, layer_idx: int = -1) -> None:
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self._clf = None  # populated by train()

    def _extract_features(self, hidden_states: Any, attention_mask: Any = None) -> Any:
        """
        Pool hidden states to one feature vector per prompt.
        hidden_states: tuple of tensors, one per layer, each (batch, seq, hidden).
        attention_mask: (batch, seq); padding positions are left out of the mean.
        """
        hs = hidden_states[self.layer_idx].float()  # (batch, seq, hidden)
        if attention_mask is None:
            pooled = hs.mean(dim=1)
        else:
            m = attention_mask.to(hs.device, hs.dtype).unsqueeze(-1)
            pooled = (hs * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
        return pooled.detach().cpu().numpy()

    def train(
        self,
        model: Any,
        safe_inputs: List[Dict[str, Any]],
        unsafe_inputs: List[Dict[str, Any]],
    ) -> None:
        """
        Train the probe by collecting activations from safe and unsafe inputs.

        Args:
            model: HuggingFace causal LM with output_hidden_states support.
            safe_inputs: List of model input dicts (input_ids, attention_mask, ...).
            unsafe_inputs: List of model input dicts.
        """
        try:
            import torch
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError("SafetyProber requires PyTorch and scikit-learn.")

        features, labels = [], []

        for batch, label in [(safe_inputs, 0), (unsafe_inputs, 1)]:
            for inp in batch:
                with torch.no_grad():
                    out = model(**inp, output_hidden_states=True)
                feats = self._extract_features(out.hidden_states)
                features.append(feats[0])  # batch size 1
                labels.append(label)

        import numpy as np
        X = np.stack(features)
        y = np.array(labels)

        self._clf = LogisticRegression(max_iter=500)
        self._clf.fit(X, y)
        logger.info("SafetyProber trained on %d examples.", len(labels))

    def predict_unsafe_probabilities(self, hidden_states: Any, attention_mask: Any = None) -> List[float]:
        """P(unsafe) for each prompt of the batch."""
        if self._clf is None:
            raise RuntimeError("SafetyProber must be trained before use. Call train() first.")
        feats = self._extract_features(hidden_states, attention_mask)
        return [float(p) for p in self._clf.predict_proba(feats)[:, 1]]

    def predict_unsafe_probability(self, hidden_states: Any, attention_mask: Any = None) -> float:
        """P(unsafe) for a single prompt; see ``predict_unsafe_probabilities`` for a batch."""
        return _single(self.predict_unsafe_probabilities(hidden_states, attention_mask))

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"clf": self._clf, "hidden_size": self.hidden_size, "layer": self.layer_idx}, f)

    @classmethod
    def load(cls, path: str) -> "SafetyProber":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        prober = cls(hidden_size=data["hidden_size"], layer_idx=data["layer"])
        prober._clf = data["clf"]
        return prober


def _single(probs: List[float]) -> float:
    if len(probs) != 1:
        raise ValueError(f"predict_unsafe_probability takes one prompt, got {len(probs)}; "
                         "use predict_unsafe_probabilities for a batch")
    return probs[0]


def prober_probs(prober: Any, hidden_states: Any, attention_mask: Any = None) -> List[float]:
    """P(unsafe) per prompt from any prober. One that only scores a single prompt
    (``predict_unsafe_probability``) is called once per row."""
    if hasattr(prober, "predict_unsafe_probabilities"):
        return list(prober.predict_unsafe_probabilities(hidden_states, attention_mask))
    rows = range(hidden_states[0].shape[0])
    return [float(prober.predict_unsafe_probability(tuple(h[i:i + 1] for h in hidden_states)))
            for i in rows]


class SafeSwitchRunner:
    """
    Inference-time SafeSwitch: combines the SafetyProber and a Refusal Head.

    During generate(), the prober is evaluated on the prefill (prompt) hidden states.
    If P(unsafe) >= threshold, logit biases are applied to steer toward refusal tokens.
    """

    def __init__(
        self,
        model: Any,
        prober: SafetyProber,
        config: Optional[SafeSwitchConfig] = None,
    ) -> None:
        self.model = model
        self.prober = prober
        self.config = config or SafeSwitchConfig()

    def _check_prompt_safety(self, input_ids: Any, attention_mask: Any = None) -> List[float]:
        """Run a prefill forward pass and get P(unsafe) for each prompt."""
        try:
            import torch
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
            return prober_probs(self.prober, out.hidden_states, attention_mask)
        except Exception as e:
            logger.error("SafeSwitch: prober failed: %s", e)
            return [0.0] * len(input_ids)

    def generate(self, input_ids: Any, **kwargs: Any) -> Any:
        """
        Safe generation: probe each prompt, then generate normally for the safe
        ones and apply the refusal logit bias to the unsafe ones.
        """
        try:
            import torch
        except ImportError:
            return self.model.generate(input_ids=input_ids, **kwargs)

        p_unsafe = self._check_prompt_safety(input_ids, kwargs.get("attention_mask"))
        unsafe = torch.tensor([p >= self.config.unsafe_threshold for p in p_unsafe])
        logger.debug("SafeSwitch: P(unsafe) = %s (threshold = %.4f)", p_unsafe, self.config.unsafe_threshold)

        if not unsafe.any():
            # Safe: generate as normal
            return self.model.generate(input_ids=input_ids, **kwargs)

        logger.warning(
            "SafeSwitch: unsafe intent detected in %d of %d prompts. Activating refusal head.",
            int(unsafe.sum()), len(p_unsafe))

        if self.config.refusal_token_ids:
            # Build logit_processor that boosts refusal token IDs on the unsafe prompts
            def _refusal_processor(input_ids_gen: Any, scores: Any) -> Any:
                rows = unsafe.to(scores.device)
                for tok_id in self.config.refusal_token_ids:
                    scores[rows, tok_id] += self.config.refusal_logit_bonus
                return scores

            existing = list(kwargs.pop("logits_processor", []))
            existing.append(_refusal_processor)
            return self.model.generate(
                input_ids=input_ids, logits_processor=existing, **kwargs
            )
        if unsafe.all():
            # No specific refusal tokens configured: return input (abort generation)
            return input_ids
        # Mixed batch: the unsafe prompts get no new tokens (padding after the prompt).
        out = self.model.generate(input_ids=input_ids, **kwargs)
        gc = getattr(self.model, "generation_config", None)
        pad = next((t for t in (kwargs.get("pad_token_id"), getattr(gc, "pad_token_id", None),
                                getattr(gc, "eos_token_id", None)) if t is not None), 0)
        out[unsafe.to(out.device), input_ids.shape[1]:] = pad[0] if isinstance(pad, list) else pad
        return out
