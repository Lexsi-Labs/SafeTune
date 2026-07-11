"""
Render a defense x attack Pareto plot from a validate_matrix CSV.

Axes:
  x: capability proxy = 1 - mean ASR across all attacks per defense
     (higher is better; "the defense keeps the model functioning on attacks")
  y: refusal rate on benign / raw = 1 - ASR on raw

Defense points that are not strictly dominated are highlighted as the
frontier. Both StringMatch and HFJudge inputs are supported; pass
``--judge stringmatch`` or ``--judge hfjudge`` to pick which column to
plot, or ``--judge both`` to render the two side by side.

Usage:

    PYTHONPATH=src python scripts/validation/plot_matrix_pareto.py \
        --matrix-csv results/validation/matrix_n20.csv \
        --compare-csv results/validation/matrix_n20_judge_compare.csv \
        --out results/validation/pareto_matrix_n20.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from safetune.evaluate.suite.pareto import ParetoVisualizer


def _load_matrix(path: Path) -> List[Dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _summarize_by_defense(rows: List[Dict[str, str]], asr_col: str = "asr") -> Dict[str, Dict[str, float]]:
    by_def: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for r in rows:
        by_def[r["defense"]].append({
            "attack": r["attack"],
            "asr": float(r[asr_col]),
            "n": int(r.get("n", 1)),
        })
    out: Dict[str, Dict[str, float]] = {}
    for defense, cells in by_def.items():
        raw = next((c for c in cells if c["attack"] == "raw"), None)
        non_raw = [c for c in cells if c["attack"] != "raw"]
        # A defense with no attack data is excluded from the Pareto plot
        # so it does not falsely show as "perfectly robust" with zero data.
        if not non_raw:
            continue
        mean_attack_asr = sum(c["asr"] for c in non_raw) / len(non_raw)
        out[defense] = {
            "refusal_on_raw": 1.0 - (raw["asr"] if raw else 0.0),
            "robustness_under_attack": 1.0 - mean_attack_asr,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-csv", default="results/validation/matrix_n20.csv")
    ap.add_argument("--compare-csv", default="results/validation/matrix_n20_judge_compare.csv",
                    help="Optional CSV produced by rejudge_with_hfjudge.py; used to pull HFJudge ASRs.")
    ap.add_argument("--judge", default="stringmatch", choices=["stringmatch", "hfjudge", "both"])
    ap.add_argument("--out", default="results/validation/pareto_matrix.png")
    args = ap.parse_args()

    rows_sm = _load_matrix(Path(args.matrix_csv))
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {
        "stringmatch": _summarize_by_defense(rows_sm, "asr"),
    }
    if args.judge != "stringmatch" and Path(args.compare_csv).exists():
        rows_hf = _load_matrix(Path(args.compare_csv))
        summaries["hfjudge"] = _summarize_by_defense(rows_hf, "asr_hfjudge")

    judges = ["stringmatch", "hfjudge"] if args.judge == "both" else [args.judge]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for judge in judges:
        summary = summaries.get(judge)
        if summary is None:
            print(f"  skipping {judge!r}: no data")
            continue
        viz = ParetoVisualizer(
            safety_label=f"refusal on raw ({judge})",
            capability_label=f"robustness 1-mean(ASR) ({judge})",
            safety_maximize=True,
            capability_maximize=True,
        )
        for defense, d in summary.items():
            viz.add(defense, safety=d["refusal_on_raw"], capability=d["robustness_under_attack"])

        out_file = out_path if args.judge != "both" else out_path.with_stem(f"{out_path.stem}_{judge}")
        print(f"\n[{judge}] points and frontier:")
        for row in viz.summary():
            flag = "*" if row["on_frontier"] else " "
            print(f"  {flag} {row['label']:<14} safety={row[f'refusal on raw ({judge})']:.3f} "
                  f"capability={row[f'robustness 1-mean(ASR) ({judge})']:.3f}")
        viz.plot(str(out_file), title=f"SafeTune defense Pareto on HarmBench-20 ({judge})")
        print(f"  wrote {out_file}")


if __name__ == "__main__":
    main()
