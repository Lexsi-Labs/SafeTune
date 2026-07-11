"""
Steering Vector Extraction Pipeline.

Extracts safety steering vectors from contrast datasets (safe vs. unsafe
prompts) for use with AdaSteer, SCANS, and STA inference-time steering.
Automates the manual process of computing mean activation differences
across transformer layers.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class VectorExtractionConfig:
    """Configuration for steering vector extraction."""

    target_layers: List[int] = field(default_factory=lambda: list(range(8, 28)))
    batch_size: int = 16
    pool_method: str = "mean"  # "mean", "last_token", "max"
    normalize: bool = True
    save_format: str = "pt"  # "pt", "npz", "safetensors"


class SteeringVectorExtractor:
    """Extract safety steering vectors from contrast datasets.

    Given paired safe/unsafe prompts, runs them through the model,
    captures hidden states at each target layer, and computes the
    mean difference vector: direction = mean(safe) - mean(unsafe).

    These vectors can then be loaded by AdaSteer, SCANS, or STA for
    inference-time safety steering.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[VectorExtractionConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or VectorExtractionConfig()
        self._hooks: List[Any] = []
        self._activations: Dict[int, List[Any]] = {}

    def _get_layers(self) -> list:
        if (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "language_model")
            and hasattr(self.model.model.language_model, "layers")
        ):
            return list(self.model.model.language_model.layers)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        return []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out
            self._activations.setdefault(layer_idx, []).append(h.detach().cpu())
        return hook_fn

    def _register_hooks(self) -> None:
        # Clear any prior state, then attach fresh hooks. We do NOT clear
        # ``self._activations`` inside ``_remove_hooks`` because removal is
        # called in the ``finally`` block of ``_collect_activations``, after
        # which we still need to aggregate the captured tensors.
        self._remove_hooks()
        self._activations.clear()
        layers = self._get_layers()
        for idx in self.config.target_layers:
            if idx < len(layers):
                handle = layers[idx].register_forward_hook(self._make_hook(idx))
                self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _pool(self, hidden: Any) -> Any:
        """Pool hidden states along the sequence dimension."""
        try:
            import torch
        except ImportError:
            raise ImportError("Vector extraction requires PyTorch.")

        if self.config.pool_method == "last_token":
            return hidden[:, -1, :]
        elif self.config.pool_method == "max":
            return hidden.max(dim=1).values
        else:  # mean
            return hidden.mean(dim=1)

    def _collect_activations(
        self, prompts: List[str], desc: str = "activations"
    ) -> Dict[int, Any]:
        """Run prompts through model and collect pooled activations per layer."""
        try:
            import torch
        except ImportError:
            raise ImportError("Vector extraction requires PyTorch.")

        self._register_hooks()
        self.model.eval()

        use_chat = getattr(self.tokenizer, "chat_template", None) is not None

        def _fmt(p: str) -> str:
            if use_chat:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
            return p

        _prev_side = getattr(self.tokenizer, "padding_side", None)
        if _prev_side is not None:
            self.tokenizer.padding_side = "left"
        n_batches = (len(prompts) + self.config.batch_size - 1) // self.config.batch_size
        try:
            bar = tqdm(
                range(0, len(prompts), self.config.batch_size),
                total=n_batches,
                desc=f"Calibrate [{desc}]",
                unit="batch",
                leave=False,
            )
            for i in bar:
                batch = [_fmt(p) for p in prompts[i : i + self.config.batch_size]]
                tok_kwargs = dict(return_tensors="pt", padding=True, truncation=True)
                if use_chat:
                    tok_kwargs["add_special_tokens"] = False
                inputs = self.tokenizer(batch, **tok_kwargs)
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    self.model(**inputs)
        finally:
            if _prev_side is not None:
                self.tokenizer.padding_side = _prev_side
            self._remove_hooks()

        # aggregate: pool each batch, THEN concat. Batches are tokenized
        # independently so their padded sequence lengths differ; pooling
        # (which reduces the sequence dim) before the concat avoids a
        # dim-1 size mismatch when there is more than one batch.
        result: Dict[int, Any] = {}
        for layer_idx, act_list in self._activations.items():
            pooled = [self._pool(a) for a in act_list]   # each: (batch, d)
            result[layer_idx] = torch.cat(pooled, dim=0)  # (N, d)

        self._activations.clear()
        return result

    def extract(
        self,
        safe_prompts: List[str],
        unsafe_prompts: List[str],
    ) -> Dict[int, Any]:
        """Extract steering vectors from contrast datasets.

        Args:
            safe_prompts: list of safe/benign prompt strings.
            unsafe_prompts: list of unsafe/harmful prompt strings.

        Returns:
            dict mapping layer_idx to steering vector (1D tensor).
        """
        try:
            import torch
        except ImportError:
            raise ImportError("Vector extraction requires PyTorch.")

        logger.info(
            "Extracting steering vectors from %d safe + %d unsafe prompts across %d layers.",
            len(safe_prompts), len(unsafe_prompts), len(self.config.target_layers),
        )

        safe_acts = self._collect_activations(safe_prompts, desc="harmless")
        unsafe_acts = self._collect_activations(unsafe_prompts, desc="harmful")

        vectors: Dict[int, Any] = {}
        for layer_idx in self.config.target_layers:
            if layer_idx not in safe_acts or layer_idx not in unsafe_acts:
                continue
            safe_mean = safe_acts[layer_idx].float().mean(dim=0)
            unsafe_mean = unsafe_acts[layer_idx].float().mean(dim=0)
            direction = safe_mean - unsafe_mean

            if self.config.normalize:
                norm = direction.norm()
                if norm > 1e-8:
                    direction = direction / norm

            vectors[layer_idx] = direction

        logger.info("Extracted %d steering vectors.", len(vectors))
        return vectors

    def save_vectors(self, vectors: Dict[int, Any], path: str) -> None:
        """Save extracted vectors to disk."""
        try:
            import torch
        except ImportError:
            raise ImportError("Saving requires PyTorch.")

        torch.save(vectors, path)
        logger.info("Saved %d steering vectors to %s", len(vectors), path)

    @staticmethod
    def load_vectors(path: str) -> Dict[int, Any]:
        """Load steering vectors from disk."""
        try:
            import torch
        except ImportError:
            raise ImportError("Loading requires PyTorch.")
        vectors = torch.load(path, map_location="cpu", weights_only=True)
        logger.info("Loaded %d steering vectors from %s", len(vectors), path)
        return vectors
