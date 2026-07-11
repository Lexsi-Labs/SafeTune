"""
LLM Safeguard (Inference Time Predictor)

Based on THU-KEG/SafetyNeuron predict_before_gen concept.
Uses the activations of safety neurons to predict if a model is about to
generate harmful content, intercepting the prompt early to save compute.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMSafeguardPredictor:
    """
    Intercepts the forward pass of a model to monitor specific neuron activations.
    If the activations exceed learned thresholds, the predictor can flag the 
    generation as unsafe before full decoding occurs.
    """

    def __init__(
        self,
        model: Any,
        classifier: Any,
        target_indices: Dict[str, List[int]],
        threshold: float = 0.5,
    ):
        """
        Args:
            model: The base Hugging Face model to monitor.
            classifier: An sklearn/PyTorch model with a `.predict_proba(X)` style interface.
                        It takes a flattened vector of activation values for the tracked neurons.
            target_indices: Mapping from module_name (e.g. 'model.layers.10.mlp.up_proj') 
                            to the list of neuron/feature indices the classifier expects.
                            Order here must exactly match the classifier's feature expectation.
            threshold: Confidence threshold for predicting "unsafe".
        """
        self.model = model
        self.classifier = classifier
        self.target_indices = target_indices
        self.threshold = threshold
        
        self._hooks: List[Any] = []
        self._is_active = False

        # State tracking during forward passes
        self.last_unsafe_probability: float = 0.0
        self.is_currently_unsafe: bool = False
        
        # We need an ordered list of targets to ensure consistent feature construction
        self._ordered_modules = list(self.target_indices.keys())

    def _build_feature_vector(self, cache: Dict[str, Any]) -> Optional[Any]:
        """Flatten the target neuron activations for the classifier."""
        import numpy as np

        features = []
        for mod in self._ordered_modules:
            tensor = cache.get(mod)
            if tensor is None:
                return None  # Missing data, can't predict
            
            # Usually shape is (batch_size, seq_len, hidden_size).
            # SafetyNeuron typically pools the sequence dimension (e.g., mean or last token).
            # We use the mean across the sequence dimension for prediction.
            # tensor: (batch, seq, hidden) -> mean: (batch, hidden)
            seq_mean = tensor.float().mean(dim=1)  # average over sequence tokens
            
            indices = self.target_indices[mod]
            # Gather specific neurons: (batch, len(indices))
            selected = seq_mean[:, indices]
            features.append(selected.detach().cpu().numpy())

        if not features:
            return None
        
        # Concatenate along the feature dimension -> (batch_size, total_neurons)
        return np.concatenate(features, axis=1)

    def _register_hooks(self):
        try:
            import torch
        except ImportError:
            pass

        # We need a shared dict for a single forward pass, but forward hooks fire sequentially.
        # We process the classification on the *last* module activated.
        last_module = self._ordered_modules[-1] if self._ordered_modules else None
        
        # Use a dictionary to store activations for the current pass
        pass_cache: Dict[str, Any] = {}

        def _make_hook(module_name: str):
            def _hook(module, args, output):
                # output can be a tensor or a tuple
                tensor = output if isinstance(output, torch.Tensor) else (
                    output[0] if isinstance(output, (tuple, list)) and len(output) > 0 else None
                )
                if tensor is not None:
                    # SafetyNeuron computes raw magnitude
                    pass_cache[module_name] = tensor.abs()

                # If this is the final module in our ordered list, we have all features!
                if module_name == last_module:
                    X = self._build_feature_vector(pass_cache)
                    if X is not None:
                        # Assuming the classifier outputs [prob_safe, prob_unsafe]
                        # or similar standardized interface. We take column 1 as prob_unsafe.
                        try:
                            probs = self.classifier.predict_proba(X)
                            prob_unsafe = float(probs[0][1])  # taking batch 0 for simplicity
                            
                            self.last_unsafe_probability = prob_unsafe
                            self.is_currently_unsafe = prob_unsafe >= self.threshold
                            
                            if self.is_currently_unsafe:
                                logger.warning(
                                    f"Safeguard intercepted unsafe prompt! "
                                    f"(confidence: {prob_unsafe:.2f} >= {self.threshold})"
                                )
                        except Exception as e:
                            logger.error(f"Safeguard classifier failed: {e}")
                    
                    # Clear cache for the next forward pass
                    pass_cache.clear()

            return _hook

        for name, module in self.model.named_modules():
            if name in self.target_indices.keys():
                h = module.register_forward_hook(_make_hook(name))
                self._hooks.append(h)

    def activate(self):
        """Enable safeguard prediction on forward passes."""
        if self._is_active:
            return
        self._register_hooks()
        self._is_active = True
        self.is_currently_unsafe = False
        self.last_unsafe_probability = 0.0

    def deactivate(self):
        """Disable safeguard prediction."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._is_active = False

    def generate_safe(self, *args, **kwargs):
        """
        A wrapper around `model.generate()`.
        If the safeguard detects unsafe intent on the prompt's initial forward pass,
        it aborts generation and returns a stylized refusal tensor.
        NOTE: This requires `kwargs` to at least contain `input_ids`.
        """
        import torch

        self.activate()
        try:
            # We do a tiny dummy forward pass on just the input prompt to trigger our safeguard
            input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
            if input_ids is not None:
                with torch.no_grad():
                    # This populates self.is_currently_unsafe based on the prompt
                    self.model(input_ids=input_ids)
            
            if self.is_currently_unsafe:
                # Prompt intercepted. Return the input_ids + a mocked refusal token ID.
                # Since we don't have tokenizer here, returning the input is the safest fallback.
                return input_ids

            return self.model.generate(*args, **kwargs)
        finally:
            self.deactivate()

