"""
Generator: orchestrate (benchmark, backend) -> JSONL of completions.

The generator is the single public class users hit when they want to run a
model over a benchmark and get responses back. It does *not* score them; that
is the job of :mod:`safetune.core.eval.pipeline.scorer`.

Design choice: this is a class, not a CLI, so it composes with the SafeTune
library API. There is a ``__main__`` shim in ``safetune.cli`` for shell users.

Usage::

    from safetune.core.eval.pipeline import Generator, GenerationConfig, load_prompts

    prompts = load_prompts("harmbench", {"max_prompts": 20})
    gen = Generator(
        backend="vllm",
        model="meta-llama/Llama-3.2-1B-Instruct",
        config=GenerationConfig(max_new_tokens=256, temperature=0.0),
    )
    rows = gen.run(prompts, out_path="results/harmbench_llama1b.jsonl")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .backends import GenerationConfig, InferenceBackend, make_backend

logger = logging.getLogger(__name__)


class Generator:
    """Run an inference backend over a list of prompt dicts.

    Args:
        backend: either a string (``"transformers"`` / ``"vllm"`` / ``"api"`` /
            ``"dryrun"``) or a pre-instantiated :class:`InferenceBackend`.
        model: HF id, local path, or in-memory ``nn.Module`` (transformers
            backend only). Ignored when ``backend`` is already an instance.
        config: shared :class:`GenerationConfig`.
        backend_kwargs: forwarded to the backend factory.
    """

    def __init__(
        self,
        backend: Union[str, InferenceBackend] = "transformers",
        model: Any = None,
        config: Optional[GenerationConfig] = None,
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(backend, InferenceBackend):
            self.backend: InferenceBackend = backend
        else:
            if model is None:
                raise ValueError("Generator: ``model`` is required when ``backend`` is a string.")
            self.backend = make_backend(backend, model=model, config=config, **(backend_kwargs or {}))
        self.config = config or self.backend.config

    # ------------------------------------------------------------- I/O utils

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    # ------------------------------------------------------------------ run

    def run(
        self,
        prompts: List[Dict[str, Any]],
        out_path: Optional[Union[str, Path]] = None,
        skip_existing: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate completions and (optionally) persist them.

        Args:
            prompts: list of dicts with at minimum a ``"prompt"`` key, as
                returned by :func:`safetune.core.eval.pipeline.loaders.load_prompts`.
            out_path: if set, write JSONL here. Used for resumability and
                downstream scoring.
            skip_existing: if True and ``out_path`` exists, return the cached
                rows instead of re-running. Matches cthetha-eval semantics.
            extra_fields: merged into every output row (e.g. ``{"defense":
                "abliteration", "alpha": 1.0}``).

        Returns:
            List of dicts: each input row plus a ``"response"`` key.
        """
        out_p = Path(out_path) if out_path else None
        if out_p and skip_existing and out_p.exists():
            logger.info("Generator: %s exists, skip_existing=True; reusing.", out_p)
            return self._read_jsonl(out_p)

        raw_prompts = [r.get("prompt", "") for r in prompts]
        logger.info(
            "Generator: backend=%s model=%s prompts=%d max_new_tokens=%d temperature=%.2f",
            self.backend.__class__.__name__,
            getattr(self.backend, "model", "?"),
            len(raw_prompts),
            self.config.max_new_tokens,
            self.config.temperature,
        )
        responses = self.backend.generate(raw_prompts)
        if len(responses) != len(raw_prompts):
            raise RuntimeError(
                f"Backend returned {len(responses)} responses for {len(raw_prompts)} prompts."
            )

        extras = dict(extra_fields or {})
        rows = [
            {
                **r,
                **extras,
                "model": getattr(self.backend, "model", "?"),
                "response": resp,
            }
            for r, resp in zip(prompts, responses)
        ]
        if out_p:
            self._write_jsonl(out_p, rows)
            logger.info("Generator: wrote %d rows to %s", len(rows), out_p)
        return rows


__all__ = ["Generator"]
