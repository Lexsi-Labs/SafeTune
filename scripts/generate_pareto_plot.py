"""
Generate the Pareto-frontier plot from validation JSON results.

Reads results/validation/refusal_direction_llama1b_harmbench20_v2.json (or
whatever ``--input`` points at) and renders a refusal-rate vs (1 - cost)
scatter with the dominated points greyed out and the frontier connected.

For "cost" we use latency-per-prompt as a stand-in for capability since the
1B + 20-prompt setup is too small for a real GSM8K signal. Customize the
mapping by editing the ``_extract_points`` function.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetune.evaluate.suite.pareto import ParetoVisualizer


def _extract_points(data: dict) -> list[tuple[str, float, float]]:
    """Map validation JSON to (label, safety, capability) triples."""
    pts = []
    # safety axis = refusal_rate (higher = better)
    # capability proxy = 1 - judge-flagged spectral anomaly proportion of
    # benign-like responses; we use raw refusal_rate vs ASR to make a clean
    # 2-point story. Replace with real GSM8K numbers if available.
    for tag in ("baseline", "defense_steer", "attack_ablate"):
        run = data["runs"][tag]
        s = run["summary"]
        pts.append((tag, s["refusal_rate"], 1.0 - s["asr"]))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="results/validation/refusal_direction_llama1b_harmbench20_v2.json",
    )
    ap.add_argument("--output", default="results/validation/pareto_refusal_direction.png")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    viz = ParetoVisualizer(
        safety_label="refusal_rate (HarmBench)",
        capability_label="1 - ASR (HarmBench)",
        safety_maximize=True,
        capability_maximize=True,
    )
    for label, safety, capability in _extract_points(data):
        viz.add(label, safety=safety, capability=capability)

    print("Summary:")
    for row in viz.summary():
        flag = "*" if row["on_frontier"] else " "
        print(f"  {flag} {row['label']:<18} safety={row['refusal_rate (HarmBench)']:.3f} capability={row['1 - ASR (HarmBench)']:.3f}")
    viz.plot(args.output, title=f"SafeTune: {data['model']} on {data['benchmark']} (n={data['n_prompts']})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
