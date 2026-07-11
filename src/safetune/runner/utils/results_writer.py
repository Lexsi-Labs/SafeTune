"""Structured results writer for safetune/results/.

Writes per-pillar method records as JSON to safetune/src/safetune/results/summaries/,
with safety scores (7 benches) and utility scores (3 core + domain tasks).
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

# Default results root: ./results under the caller's working directory.
# (Deriving it from __file__ would place results — and checkpoints — inside
# site-packages on a pip install, which is read-only on system installs.)
DEFAULT_RESULTS_DIR = str(Path.cwd() / "results")


class ResultsWriter:
    """Write method evaluation results to the safetune results directory.

    Usage::

        writer = ResultsWriter(pillar="recover", results_dir="/path/to/results")
        writer.append({
            "method": "CThetaTrainer",
            "variant": "strength=1.0",
            "metrics": {"harmbench_refusal": 0.92, "gsm8k": 0.78, ...},
            "safety_mean": 0.88,
            "utility_mean": 0.76,
        })
    """

    def __init__(
        self,
        pillar: str,
        *,
        results_dir: str = None,
    ):
        self.pillar = pillar
        self.results_dir = results_dir or DEFAULT_RESULTS_DIR
        self._summaries_dir = os.path.join(self.results_dir, "summaries")
        os.makedirs(self._summaries_dir, exist_ok=True)

    @property
    def path(self) -> str:
        return os.path.join(self._summaries_dir, f"results_{self.pillar}.json")

    def load(self) -> list[dict]:
        if os.path.exists(self.path):
            try:
                return json.load(open(self.path))
            except Exception:
                pass
        return []

    def append(self, record: dict) -> None:
        """Append a result record, de-duplicating on (method, variant)."""
        data = self.load()
        key = (record.get("method"), record.get("variant"))
        data = [d for d in data if (d.get("method"), d.get("variant")) != key]
        data.append(record)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def write_summary_md(self, drift_task: Optional[str] = None) -> str:
        """Write a markdown summary table and return the path."""
        data = self.load()
        if not data:
            return ""
        from safetune.runner.utils.data_utils import SAFETY_BENCHES, UTILITY_TASKS_CORE

        lines = [f"# {self.pillar.capitalize()} Results\n"]
        benches = [b.replace("_v1", "") for b in SAFETY_BENCHES]
        util_keys = list(UTILITY_TASKS_CORE)
        headers = ["Method", "Variant"] + [f"{b}_refusal" for b in benches] + util_keys
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        def _fmt(v):
            # A key can be present with an explicit None (not just missing);
            # f"{None:.3f}" raises TypeError and aborts the whole summary.
            return f"{(v if isinstance(v, (int, float)) else float('nan')):.3f}"

        for rec in data:
            m = rec.get("metrics", {})
            row = [
                rec.get("method", ""),
                rec.get("variant", ""),
                *[_fmt(m.get(f"{b}_refusal")) for b in benches],
                *[_fmt(m.get(k)) for k in util_keys],
            ]
            lines.append("| " + " | ".join(row) + " |")

        md_path = os.path.join(self._summaries_dir, f"SUMMARY_{self.pillar}.md")
        with open(md_path, "w") as f:
            f.write("\n".join(lines))
        return md_path


def append_result(
    pillar: str,
    record: dict,
    *,
    results_dir: str = None,
) -> None:
    """Convenience: append a record to the pillar results file."""
    ResultsWriter(pillar, results_dir=results_dir).append(record)
