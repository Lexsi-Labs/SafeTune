"""
Re-judge a validate_matrix JSONL output with the heavy HFJudge
(HarmBench-Mistral-7b-val-cls by default), and compare per-cell ASR
against the StringMatchJudge that the original matrix run used.

Output: a CSV with columns
    defense, attack, asr_string_match, asr_hfjudge, n

so the disagreement is visible at a glance.

Usage:

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/validation/rejudge_with_hfjudge.py \
        --in-jsonl results/validation/matrix_n6.jsonl \
        --out-csv results/validation/matrix_n6_judge_compare.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from safetune.core.eval.pipeline import HFJudge, HFJudgeConfig, asr_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default="results/validation/matrix_n6.jsonl")
    ap.add_argument("--judge-model", default="cais/HarmBench-Mistral-7b-val-cls")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-csv", default="results/validation/matrix_n6_judge_compare.csv")
    args = ap.parse_args()

    cells = []
    with open(args.in_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cells.append(json.loads(line))

    print(f"Loaded {len(cells)} cells from {args.in_jsonl}")
    print(f"Loading {args.judge_model} on {args.device} ...")
    judge = HFJudge(HFJudgeConfig(model=args.judge_model, device=args.device,
                                  torch_dtype="bfloat16", max_new_tokens=4))

    out_rows = []
    for cell in cells:
        d, a = cell["defense"], cell["attack"]
        rows = cell["rows"]
        # Re-judge with HF.
        re_rows = judge.score(rows)
        sm_summary = asr_summary(rows)  # uses existing judgement
        hf_summary = asr_summary(re_rows)
        out_rows.append({
            "defense": d, "attack": a, "n": sm_summary["n"],
            "asr_string_match": round(sm_summary["asr"], 3),
            "asr_hfjudge": round(hf_summary["asr"], 3),
            "delta": round(hf_summary["asr"] - sm_summary["asr"], 3),
        })
        print(f"  {d:<14} {a:<14} string={sm_summary['asr']:.3f} hf={hf_summary['asr']:.3f} delta={hf_summary['asr']-sm_summary['asr']:+.3f}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["defense", "attack", "n",
                                           "asr_string_match", "asr_hfjudge", "delta"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
