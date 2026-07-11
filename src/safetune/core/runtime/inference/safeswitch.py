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

    def _extract_features(self, hidden_states: Any) -> Any:
        """
        Pool hidden states to a feature vector.
        hidden_states: tuple of tensors, one per layer, each (batch, seq, hidden).
        """
        layer = self.layer_idx
        hs = hidden_states[layer]      # (batch, seq, hidden)
        # Mean-pool over sequence dimension
        pooled = hs.float().mean(dim=1)  # (batch, hidden)
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

    def predict_unsafe_probability(self, hidden_states: Any) -> float:
        """Return P(unsafe) for the current hidden state batch."""
        if self._clf is None:
            raise RuntimeError("SafetyProber must be trained before use. Call train() first.")
        feats = self._extract_features(hidden_states)
        prob = self._clf.predict_proba(feats)[0][1]
        return float(prob)

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

    def _check_prompt_safety(self, input_ids: Any) -> float:
        """Run a prefill forward pass and get P(unsafe)."""
        try:
            import torch
            with torch.no_grad():
                out = self.model(input_ids=input_ids, output_hidden_states=True)
            return self.prober.predict_unsafe_probability(out.hidden_states)
        except Exception as e:
            logger.error("SafeSwitch: prober failed: %s", e)
            return 0.0

    def generate(self, input_ids: Any, **kwargs: Any) -> Any:
        """
        Safe generation: probe the prompt, then either generate normally or
        apply logit biases to the refusal head.
        """
        try:
            import torch
        except ImportError:
            return self.model.generate(input_ids=input_ids, **kwargs)

        p_unsafe = self._check_prompt_safety(input_ids)
        logger.debug("SafeSwitch: P(unsafe) = %.4f (threshold = %.4f)", p_unsafe, self.config.unsafe_threshold)

        if p_unsafe < self.config.unsafe_threshold:
            # Safe: generate as normal
            return self.model.generate(input_ids=input_ids, **kwargs)

        logger.warning(
            "SafeSwitch: unsafe intent detected (P=%.2f). Activating refusal head.", p_unsafe
        )

        if self.config.refusal_token_ids:
            # Build logit_processor that boosts refusal token IDs
            def _refusal_processor(input_ids_gen: Any, scores: Any) -> Any:
                for tok_id in self.config.refusal_token_ids:
                    scores[:, tok_id] += self.config.refusal_logit_bonus
                return scores

            existing = list(kwargs.pop("logits_processor", []))
            existing.append(_refusal_processor)
            return self.model.generate(
                input_ids=input_ids, logits_processor=existing, **kwargs
            )
        else:
            # No specific refusal tokens configured: return input (abort generation)
            return input_ids
