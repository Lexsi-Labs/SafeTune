"""
RRFA: Representation Rerouting for Agentic Safety.
memo-ozdincer/RRFA

Defends LLM-based agents against indirect prompt injection by
rerouting representations when the model's hidden state indicates
it is following injected instructions rather than the user's original
intent.  Uses a triplet-loss trained classifier on hidden states to
distinguish between legitimate tool outputs and injected prompts.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RRFAConfig:
    """Configuration for Representation Rerouting for Agentic Safety."""

    monitor_layers: List[int] = field(default_factory=lambda: list(range(16, 28)))
    triplet_margin: float = 1.0
    detection_threshold: float = 0.6
    reroute_to_refusal: bool = True
    embedding_dim: int = 256
    max_context_window: int = 4096


class InjectionDetector:
    """Detects prompt injection via hidden-state analysis.

    Trained with triplet loss:
      - anchor   = hidden state from legitimate user prompt
      - positive = hidden state from legitimate tool output
      - negative = hidden state from injected adversarial prompt
    """

    def __init__(self, config: Optional[RRFAConfig] = None) -> None:
        self.config = config or RRFAConfig()
        self._projection: Optional[Any] = None
        self._anchor_centroid: Optional[Any] = None

    def train_projector(
        self,
        anchor_states: Any,
        positive_states: Any,
        negative_states: Any,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """Train a linear projection that separates injected from legitimate."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("RRFA requires PyTorch.")

        input_dim = anchor_states.shape[-1]
        proj = nn.Linear(input_dim, self.config.embedding_dim, bias=False)
        proj = proj.to(anchor_states.device)
        optimiser = torch.optim.Adam(proj.parameters(), lr=lr)
        triplet_loss = nn.TripletMarginLoss(margin=self.config.triplet_margin)

        losses = []
        for epoch in range(epochs):
            optimiser.zero_grad()
            a = proj(anchor_states.float())
            p = proj(positive_states.float())
            n = proj(negative_states.float())
            loss = triplet_loss(a, p, n)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())

        self._projection = proj
        # store centroid of anchor embeddings for detection
        with torch.no_grad():
            self._anchor_centroid = proj(anchor_states.float()).mean(dim=0)

        final_loss = losses[-1] if losses else float("inf")
        logger.info("RRFA: trained projector for %d epochs, final loss=%.4f", epochs, final_loss)
        return {"final_loss": final_loss, "epochs": epochs}

    def detect_injection(self, hidden_state: Any) -> Tuple[bool, float]:
        """Detect whether a hidden state was produced by an injected prompt.

        Returns (is_injection, confidence_score).
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("RRFA requires PyTorch.")

        if self._projection is None or self._anchor_centroid is None:
            raise RuntimeError("Call train_projector() first.")

        with torch.no_grad():
            embedded = self._projection(hidden_state.float())
            if embedded.dim() > 1:
                embedded = embedded.mean(dim=0)
            sim = F.cosine_similarity(
                embedded.unsqueeze(0),
                self._anchor_centroid.unsqueeze(0),
                dim=-1,
            ).item()

        # low similarity to anchor centroid → likely injection
        is_injection = sim < self.config.detection_threshold
        confidence = 1.0 - sim if is_injection else sim
        return is_injection, confidence

    def save(self, path: str) -> None:
        """Save the trained projector and centroid."""
        try:
            import torch
        except ImportError:
            raise ImportError("RRFA requires PyTorch.")
        torch.save({
            "projection": self._projection.state_dict() if self._projection else None,
            "anchor_centroid": self._anchor_centroid,
            "config": self.config,
        }, path)
        logger.info("RRFA: saved detector to %s", path)

    @classmethod
    def load(cls, path: str) -> "InjectionDetector":
        """Load a trained detector."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("RRFA requires PyTorch.")
        data = torch.load(path, map_location="cpu", weights_only=False)
        config = data["config"]
        detector = cls(config)
        detector._anchor_centroid = data["anchor_centroid"]
        if data["projection"] is not None:
            dim_in = list(data["projection"].values())[0].shape[1]
            proj = nn.Linear(dim_in, config.embedding_dim, bias=False)
            proj.load_state_dict(data["projection"])
            detector._projection = proj
        return detector


class RRFAWrapper:
    """Full RRFA pipeline: detect injection → reroute to refusal."""

    def __init__(
        self,
        model: Any,
        detector: Optional[InjectionDetector] = None,
        config: Optional[RRFAConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or RRFAConfig()
        self.detector = detector or InjectionDetector(self.config)
        self._hooks: List[Any] = []
        self._last_detection: Optional[Tuple[bool, float]] = None

    def check_and_reroute(
        self,
        hidden_states: Dict[int, Any],
    ) -> Tuple[bool, float, Dict[int, Any]]:
        """Check for injection across monitored layers and reroute if detected.

        Returns (injection_detected, max_confidence, rerouted_states).
        """
        try:
            import torch
        except ImportError:
            raise ImportError("RRFA requires PyTorch.")

        max_confidence = 0.0
        detected = False
        rerouted = dict(hidden_states)

        for layer_idx in self.config.monitor_layers:
            if layer_idx not in hidden_states:
                continue
            h = hidden_states[layer_idx]
            if h.dim() == 3:
                h_pooled = h.mean(dim=1).mean(dim=0)
            elif h.dim() == 2:
                h_pooled = h.mean(dim=0)
            else:
                h_pooled = h

            is_inj, conf = self.detector.detect_injection(h_pooled)
            if is_inj and conf > max_confidence:
                max_confidence = conf
                detected = True

            if is_inj and self.config.reroute_to_refusal:
                # zero out the hidden state to trigger refusal behaviour
                rerouted[layer_idx] = torch.zeros_like(hidden_states[layer_idx])

        self._last_detection = (detected, max_confidence)
        if detected:
            logger.warning("RRFA: injection detected (confidence=%.3f), rerouting.", max_confidence)
        return detected, max_confidence, rerouted
